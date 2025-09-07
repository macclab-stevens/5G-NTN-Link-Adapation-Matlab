#!/usr/bin/env python3
"""
5G gNB SCHED/MAC Log Processor
Extracts PDSCH transmissions and correlates with HARQ-ACK feedback
Outputs CSV with MCS, TBS, ACK/NACK status over time
"""

import re
import csv
import argparse
from datetime import datetime
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

class SchedulerLogProcessor:
    def __init__(self, log_file, verbose=False):
        self.log_file = log_file
        self.verbose = verbose
        
        # Storage for parsed data
        self.pdsch_transmissions = []
        self.harq_feedback = []
        self.correlated_data = []
        
        # Regex patterns for parsing
        self.timestamp_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)')
        self.pdsch_pattern = re.compile(
            r'- UE PDSCH: ue=(\d+) c-rnti=0x([0-9a-fA-F]+) h_id=(\d+).*?tbs=(\d+) mcs=(\d+).*?nrtx=(\d+)'
        )
        self.harq_pattern = re.compile(
            r'- HARQ-ACK: ue=(\d+) rnti=0x([0-9a-fA-F]+).*?h_id=(\d+) ack=(\d+)(?:.*?tbs=(\d+))?'
        )
        
    def log(self, message):
        """Print log message if verbose mode is enabled"""
        if self.verbose:
            print(f"[LOG] {message}")
            
    def parse_timestamp(self, timestamp_str):
        """Parse timestamp string to datetime object"""
        try:
            if '.' in timestamp_str:
                # Split timestamp and microseconds
                base, microsecs = timestamp_str.split('.')
                # Truncate microseconds to 6 digits if longer
                if len(microsecs) > 6:
                    microsecs = microsecs[:6]
                # Pad with zeros if shorter than 6 digits
                microsecs = microsecs.ljust(6, '0')
                timestamp_str = f"{base}.{microsecs}"
            
            return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")
        except Exception as e:
            if self.verbose:
                print(f"[WARNING] Failed to parse timestamp '{timestamp_str}': {e}")
            return None
    
    def extract_pdsch_data(self, line, timestamp):
        """Extract PDSCH transmission data from log line"""
        match = self.pdsch_pattern.search(line)
        if match:
            ue_id, rnti, harq_id, tbs, mcs, nrtx = match.groups()
            
            pdsch_entry = {
                'timestamp': timestamp,
                'ue_id': int(ue_id),
                'rnti': f"0x{rnti}",
                'harq_id': int(harq_id),
                'tbs': int(tbs),
                'mcs': int(mcs),
                'nrtx': int(nrtx),  # Number of retransmissions
                'transmission_type': 'retx' if int(nrtx) > 0 else 'new'
            }
            
            self.pdsch_transmissions.append(pdsch_entry)
            self.log(f"PDSCH TX: UE={ue_id}, RNTI={rnti}, H_ID={harq_id}, MCS={mcs}, TBS={tbs}, NRTX={nrtx}")
            
    def extract_harq_data(self, line, timestamp):
        """Extract HARQ-ACK feedback data from log line"""
        match = self.harq_pattern.search(line)
        if match:
            ue_id, rnti, harq_id, ack_status, tbs = match.groups()
            
            harq_entry = {
                'timestamp': timestamp,
                'ue_id': int(ue_id),
                'rnti': f"0x{rnti}",
                'harq_id': int(harq_id),
                'ack_status': int(ack_status),  # 1=ACK, 0=NACK, 2=DTX
                'ack_result': 'ACK' if int(ack_status) == 1 else 'NACK' if int(ack_status) == 0 else 'DTX',
                'tbs': int(tbs) if tbs else None
            }
            
            self.harq_feedback.append(harq_entry)
            self.log(f"HARQ-ACK: UE={ue_id}, RNTI={rnti}, H_ID={harq_id}, ACK={harq_entry['ack_result']}")
    
    def correlate_pdsch_harq(self, max_time_diff_ms=50):
        """
        Correlate PDSCH transmissions with HARQ-ACK feedback
        Args:
            max_time_diff_ms: Maximum time difference in milliseconds for correlation
        """
        self.log(f"Correlating {len(self.pdsch_transmissions)} PDSCH transmissions with {len(self.harq_feedback)} HARQ-ACK entries...")
        
        # Create lookup for quick access
        harq_by_key = defaultdict(list)
        for harq in self.harq_feedback:
            if harq['timestamp']:
                key = (harq['ue_id'], harq['rnti'], harq['harq_id'])
                harq_by_key[key].append(harq)
        
        correlated_count = 0
        uncorrelated_count = 0
        
        for pdsch in self.pdsch_transmissions:
            if not pdsch['timestamp']:
                continue
                
            key = (pdsch['ue_id'], pdsch['rnti'], pdsch['harq_id'])
            found_match = False
            
            # Look for matching HARQ-ACK feedback
            if key in harq_by_key:
                for harq in harq_by_key[key]:
                    # Check if HARQ-ACK comes after PDSCH (as expected)
                    time_diff = (harq['timestamp'] - pdsch['timestamp']).total_seconds() * 1000
                    
                    if 0 <= time_diff <= max_time_diff_ms:
                        # Found a match!
                        correlation = {
                            'pdsch_timestamp': pdsch['timestamp'],
                            'harq_timestamp': harq['timestamp'],
                            'time_diff_ms': time_diff,
                            'ue_id': pdsch['ue_id'],
                            'rnti': pdsch['rnti'],
                            'harq_id': pdsch['harq_id'],
                            'mcs': pdsch['mcs'],
                            'tbs': pdsch['tbs'],
                            'nrtx': pdsch['nrtx'],
                            'transmission_type': pdsch['transmission_type'],
                            'ack_status': harq['ack_status'],
                            'ack_result': harq['ack_result']
                        }
                        
                        self.correlated_data.append(correlation)
                        found_match = True
                        correlated_count += 1
                        break
            
            if not found_match:
                uncorrelated_count += 1
                # Still record the PDSCH transmission without ACK info
                correlation = {
                    'pdsch_timestamp': pdsch['timestamp'],
                    'harq_timestamp': None,
                    'time_diff_ms': None,
                    'ue_id': pdsch['ue_id'],
                    'rnti': pdsch['rnti'],
                    'harq_id': pdsch['harq_id'],
                    'mcs': pdsch['mcs'],
                    'tbs': pdsch['tbs'],
                    'nrtx': pdsch['nrtx'],
                    'transmission_type': pdsch['transmission_type'],
                    'ack_status': None,
                    'ack_result': 'UNKNOWN'
                }
                self.correlated_data.append(correlation)
        
        self.log(f"Correlation complete: {correlated_count} matched, {uncorrelated_count} unmatched")
    
    def process_log_file(self):
        """Process the entire log file"""
        print(f"📖 Processing log file: {self.log_file}")
        
        try:
            with open(self.log_file, 'r') as f:
                line_count = 0
                current_timestamp = None
                
                for line in f:
                    line_count += 1
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    # Extract timestamp from current line if it has one
                    timestamp_match = self.timestamp_pattern.search(line)
                    if timestamp_match:
                        current_timestamp = self.parse_timestamp(timestamp_match.group(1))
                    
                    # Extract PDSCH transmissions
                    if 'UE PDSCH:' in line:
                        self.extract_pdsch_data(line, current_timestamp)
                    
                    # Extract HARQ-ACK feedback
                    if 'HARQ-ACK:' in line:
                        self.extract_harq_data(line, current_timestamp)
                    
                    # Progress indicator
                    if line_count % 50000 == 0:
                        print(f"  Processed {line_count:,} lines...")
                        
        except FileNotFoundError:
            print(f"❌ Error: Log file '{self.log_file}' not found")
            return False
        except Exception as e:
            print(f"❌ Error processing log file: {e}")
            return False
            
        print(f"✅ Processed {line_count:,} lines total")
        return True
    
    def save_to_csv(self, output_file):
        """Save correlated data to CSV file"""
        if not self.correlated_data:
            print("⚠ No correlated data to save")
            return
        
        # Convert to pandas DataFrame for easier manipulation
        df = pd.DataFrame(self.correlated_data)
        
        # Add derived columns
        df['pdsch_time_relative_s'] = (df['pdsch_timestamp'] - df['pdsch_timestamp'].min()).dt.total_seconds()
        df['bler'] = (df['ack_status'] == 0).astype(int)  # NACK = 1, ACK = 0
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"💾 Saved {len(df)} entries to {output_file}")
        
        # Print statistics
        print(f"\n📊 STATISTICS:")
        print(f"  Total transmissions: {len(df)}")
        print(f"  MCS range: {df['mcs'].min()} - {df['mcs'].max()}")
        print(f"  TBS range: {df['tbs'].min()} - {df['tbs'].max()} bytes")
        
        if 'ack_status' in df.columns and df['ack_status'].notna().any():
            ack_counts = df['ack_result'].value_counts()
            total_with_feedback = df['ack_status'].notna().sum()
            bler = (df['ack_status'] == 0).sum() / total_with_feedback * 100 if total_with_feedback > 0 else 0
            
            print(f"  ACK/NACK feedback: {total_with_feedback} transmissions")
            print(f"  BLER: {bler:.2f}%")
            for result, count in ack_counts.items():
                print(f"    {result}: {count} ({count/len(df)*100:.1f}%)")
        
        # MCS distribution
        mcs_dist = df['mcs'].value_counts().sort_index()
        print(f"  MCS distribution:")
        for mcs, count in mcs_dist.items():
            print(f"    MCS {mcs}: {count} transmissions ({count/len(df)*100:.1f}%)")
    
    def plot_analysis(self, output_prefix="analysis"):
        """Generate analysis plots"""
        if not self.correlated_data:
            print("⚠ No data to plot")
            return
        
        df = pd.DataFrame(self.correlated_data)
        df['pdsch_time_relative_s'] = (df['pdsch_timestamp'] - df['pdsch_timestamp'].min()).dt.total_seconds()
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('5G gNB PDSCH Transmission Analysis', fontsize=16)
        
        # Plot 1: MCS over time
        ax1.scatter(df['pdsch_time_relative_s'], df['mcs'], alpha=0.6, s=10)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('MCS')
        ax1.set_title('MCS over Time')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: TBS over time (colored by MCS)
        scatter = ax2.scatter(df['pdsch_time_relative_s'], df['tbs'], c=df['mcs'], 
                             cmap='viridis', alpha=0.6, s=10)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('TBS (bytes)')
        ax2.set_title('TBS over Time (colored by MCS)')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='MCS')
        
        # Plot 3: MCS distribution
        mcs_counts = df['mcs'].value_counts().sort_index()
        ax3.bar(mcs_counts.index, mcs_counts.values, alpha=0.7)
        ax3.set_xlabel('MCS')
        ax3.set_ylabel('Number of Transmissions')
        ax3.set_title('MCS Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: ACK/NACK analysis (if available)
        if 'ack_status' in df.columns and df['ack_status'].notna().any():
            # BLER by MCS
            bler_by_mcs = df[df['ack_status'].notna()].groupby('mcs').agg({
                'ack_status': ['count', lambda x: (x == 0).mean() * 100]
            }).round(2)
            bler_by_mcs.columns = ['Count', 'BLER_pct']
            bler_by_mcs = bler_by_mcs[bler_by_mcs['Count'] >= 5]  # Only MCS with enough samples
            
            if not bler_by_mcs.empty:
                ax4.bar(bler_by_mcs.index, bler_by_mcs['BLER_pct'], alpha=0.7, color='red')
                ax4.set_xlabel('MCS')
                ax4.set_ylabel('BLER (%)')
                ax4.set_title('BLER by MCS')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Insufficient ACK/NACK data\nfor BLER analysis', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('BLER Analysis')
        else:
            ax4.text(0.5, 0.5, 'No ACK/NACK feedback\ndata available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('ACK/NACK Analysis')
        
        plt.tight_layout()
        plot_file = f"{output_prefix}_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📈 Saved analysis plots to {plot_file}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Process 5G gNB SCHED/MAC logs to extract PDSCH and HARQ-ACK data')
    parser.add_argument('--log-file', required=True, help='Path to the gNB log file')
    parser.add_argument('--output', default='pdsch_harq_analysis.csv', help='Output CSV file name')
    parser.add_argument('--plot', action='store_true', help='Generate analysis plots')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--max-correlation-time-ms', type=int, default=50, 
                       help='Maximum time difference for PDSCH-HARQ correlation (ms)')
    
    args = parser.parse_args()
    
    print("🔍 5G gNB SCHED/MAC Log Processor")
    print("=" * 50)
    
    # Create processor and process log
    processor = SchedulerLogProcessor(args.log_file, args.verbose)
    
    if not processor.process_log_file():
        return 1
    
    # Correlate PDSCH transmissions with HARQ-ACK feedback
    processor.correlate_pdsch_harq(args.max_correlation_time_ms)
    
    # Save results
    processor.save_to_csv(args.output)
    
    # Generate plots if requested
    if args.plot:
        plot_prefix = args.output.replace('.csv', '')
        processor.plot_analysis(plot_prefix)
    
    print("\n🎉 Processing completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
