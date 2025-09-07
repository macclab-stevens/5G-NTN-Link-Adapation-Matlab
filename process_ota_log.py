#!/usr/bin/env python3
"""
OTA Log Processor - Extract and Map MCS to SINR Values
Processes gnb.log files to extract:
- Configuration parameters (mcs_table, olla settings, etc.)
- PDSCH MCS allocations
- SINR values from RRC messages
- Maps MCS values to SINR values
"""

import re
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Process OTA gnb.log file to extract MCS-SINR mapping")
    parser.add_argument('--log-file', type=str, default='./OTA/gnb.log', 
                        help='Path to gnb.log file')
    parser.add_argument('--output-csv', type=str, default='mcs_sinr_mapping.csv',
                        help='Output CSV file for MCS-SINR data')
    parser.add_argument('--plot', action='store_true',
                        help='Generate MCS vs SINR plot')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    return parser.parse_args()

class OTALogProcessor:
    def __init__(self, log_file, verbose=False):
        self.log_file = log_file
        self.verbose = verbose
        self.config = {}
        self.mcs_data = []
        self.sinr_data = []
        self.mcs_sinr_pairs = []
        
        # Regex patterns
        self.config_patterns = {
            'mcs_table': re.compile(r'mcs_table:\s*(\w+)'),
            'olla_cqi_inc_step': re.compile(r'olla_cqi_inc_step:\s*(\d+(?:\.\d+)?)'),
            'olla_target_bler': re.compile(r'olla_target_bler:\s*(\d+(?:\.\d+)?)'),
            'olla_max_cqi_offset': re.compile(r'olla_max_cqi_offset:\s*(\d+(?:\.\d+)?)'),
            'harq_la_cqi_drop_threshold': re.compile(r'harq_la_cqi_drop_threshold:\s*(\d+(?:\.\d+)?)'),
            'harq_la_ri_drop_threshold': re.compile(r'harq_la_ri_drop_threshold:\s*(\d+(?:\.\d+)?)')
        }
        
        # PDSCH MCS pattern
        self.pdsch_pattern = re.compile(
            r'UE PDSCH:.*?ue=(\d+).*?c-rnti=(0x[0-9a-fA-F]+).*?h_id=(\d+).*?rb=\[(\d+)\.\.(\d+)\).*?'
            r'symb=\[(\d+)\.\.(\d+)\).*?tbs=(\d+).*?mcs=(\d+).*?rv=(\d+).*?nrtx=(\d+).*?k1=(\d+)'
        )
        
        # SINR pattern - looking for SINR values in CRC messages
        self.sinr_pattern = re.compile(r'sinr=(\d+(?:\.\d+)?)dB')
        
        # Timestamp pattern to correlate events
        self.timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)')
    
    def log(self, message):
        """Print verbose log messages"""
        if self.verbose:
            print(f"[LOG] {message}")
    
    def parse_timestamp(self, timestamp_str):
        """Parse timestamp string to datetime object"""
        try:
            # Handle format: 2025-09-01T18:04:34.647423
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
    
    def extract_configuration(self, line):
        """Extract configuration parameters from a line"""
        for param, pattern in self.config_patterns.items():
            match = pattern.search(line)
            if match:
                value = match.group(1)
                # Try to convert to appropriate type
                try:
                    if '.' in value:
                        self.config[param] = float(value)
                    else:
                        self.config[param] = int(value) if value.isdigit() else value
                except ValueError:
                    self.config[param] = value
                self.log(f"Found config: {param} = {self.config[param]}")
    
    def extract_pdsch_mcs(self, line, timestamp):
        """Extract PDSCH MCS information from a line"""
        match = self.pdsch_pattern.search(line)
        if match:
            mcs_info = {
                'timestamp': timestamp,
                'ue_id': int(match.group(1)),
                'c_rnti': match.group(2),
                'harq_id': int(match.group(3)),
                'rb_start': int(match.group(4)),
                'rb_end': int(match.group(5)),
                'symb_start': int(match.group(6)),
                'symb_end': int(match.group(7)),
                'tbs': int(match.group(8)),
                'mcs': int(match.group(9)),
                'rv': int(match.group(10)),
                'nrtx': int(match.group(11)),
                'k1': int(match.group(12))
            }
            self.mcs_data.append(mcs_info)
            self.log(f"Found PDSCH MCS: UE={mcs_info['ue_id']}, MCS={mcs_info['mcs']}, TBS={mcs_info['tbs']}")
            return mcs_info
        return None
    
    def extract_sinr(self, line, timestamp):
        """Extract SINR values from CRC messages"""
        matches = self.sinr_pattern.findall(line)
        for match in matches:
            try:
                # SINR value is already in dB format from the log
                actual_sinr = float(match)
                
                sinr_info = {
                    'timestamp': timestamp,
                    'actual_sinr': actual_sinr
                }
                self.sinr_data.append(sinr_info)
                self.log(f"Found SINR: {actual_sinr} dB")
                return sinr_info
            except ValueError:
                continue
        return None
    
    def correlate_mcs_sinr(self, time_window=1.0):
        """Correlate MCS and SINR values within a time window (seconds)"""
        self.log(f"Correlating MCS and SINR data within {time_window}s time window...")
        
        # Debug: Print sample timestamps to understand format
        if self.mcs_data and self.sinr_data:
            self.log(f"Sample MCS timestamps: {[str(m['timestamp']) for m in self.mcs_data[:3]]}")
            self.log(f"Sample SINR timestamps: {[str(s['timestamp']) for s in self.sinr_data[:3]]}")
            
            # Check if timestamps are None
            mcs_none_count = sum(1 for m in self.mcs_data if m['timestamp'] is None)
            sinr_none_count = sum(1 for s in self.sinr_data if s['timestamp'] is None)
            self.log(f"Timestamps with None values: MCS={mcs_none_count}/{len(self.mcs_data)}, SINR={sinr_none_count}/{len(self.sinr_data)}")
        
        for mcs_entry in self.mcs_data:
            if mcs_entry['timestamp'] is None:
                continue
                
            # Find SINR values within time window
            closest_sinr = None
            min_time_diff = float('inf')
            
            for sinr_entry in self.sinr_data:
                if sinr_entry['timestamp'] is None:
                    continue
                
                time_diff = abs((mcs_entry['timestamp'] - sinr_entry['timestamp']).total_seconds())
                
                if time_diff <= time_window and time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_sinr = sinr_entry
            
            if closest_sinr:
                pair = {
                    'timestamp': mcs_entry['timestamp'],
                    'ue_id': mcs_entry['ue_id'],
                    'mcs': mcs_entry['mcs'],
                    'tbs': mcs_entry['tbs'],
                    'rb_count': mcs_entry['rb_end'] - mcs_entry['rb_start'],
                    'sinr_db': closest_sinr['actual_sinr'],
                    'time_diff': min_time_diff
                }
                self.mcs_sinr_pairs.append(pair)
                self.log(f"Correlated: MCS={pair['mcs']}, SINR={pair['sinr_db']:.2f} dB, time_diff={pair['time_diff']:.3f}s")
    
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
                    
                    # Extract configuration parameters
                    self.extract_configuration(line)
                    
                    # Extract PDSCH MCS data (uses current_timestamp since these lines don't have timestamps)
                    if 'UE PDSCH:' in line:
                        self.extract_pdsch_mcs(line, current_timestamp)
                    
                    # Extract SINR data from CRC lines (uses current_timestamp)
                    if 'sinr=' in line and 'CRC:' in line:
                        self.extract_sinr(line, current_timestamp)
                    
                    # Progress indicator
                    if line_count % 10000 == 0:
                        print(f"  Processed {line_count:,} lines...")
                
                print(f"✅ Processed {line_count:,} lines total")
                
        except FileNotFoundError:
            print(f"❌ Error: Log file not found: {self.log_file}")
            return False
        except Exception as e:
            print(f"❌ Error processing log file: {e}")
            return False
        
        return True
    
    def print_summary(self):
        """Print processing summary"""
        print("\n" + "=" * 60)
        print("📊 PROCESSING SUMMARY")
        print("=" * 60)
        
        print("\n🔧 Configuration Parameters:")
        if self.config:
            for param, value in self.config.items():
                print(f"  {param}: {value}")
        else:
            print("  No configuration parameters found")
        
        print(f"\n📈 Data Extracted:")
        print(f"  PDSCH MCS entries: {len(self.mcs_data)}")
        print(f"  SINR entries: {len(self.sinr_data)}")
        print(f"  Correlated MCS-SINR pairs: {len(self.mcs_sinr_pairs)}")
        
        if self.mcs_sinr_pairs:
            mcs_values = [pair['mcs'] for pair in self.mcs_sinr_pairs]
            sinr_values = [pair['sinr_db'] for pair in self.mcs_sinr_pairs]
            
            print(f"\n📊 MCS Statistics:")
            print(f"  MCS range: {min(mcs_values)} - {max(mcs_values)}")
            print(f"  Most common MCS: {max(set(mcs_values), key=mcs_values.count)}")
            
            print(f"\n📊 SINR Statistics:")
            print(f"  SINR range: {min(sinr_values):.2f} - {max(sinr_values):.2f} dB")
            print(f"  Average SINR: {np.mean(sinr_values):.2f} dB")
            print(f"  SINR std dev: {np.std(sinr_values):.2f} dB")
    
    def save_to_csv(self, output_file):
        """Save MCS-SINR pairs to CSV file"""
        if not self.mcs_sinr_pairs:
            print("⚠ No MCS-SINR pairs to save")
            return
        
        df = pd.DataFrame(self.mcs_sinr_pairs)
        df.to_csv(output_file, index=False)
        print(f"💾 Saved {len(self.mcs_sinr_pairs)} MCS-SINR pairs to {output_file}")
    
    def create_plot(self):
        """Create MCS vs SINR scatter plot"""
        if not self.mcs_sinr_pairs:
            print("⚠ No data to plot")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Extract data for plotting
        mcs_values = [pair['mcs'] for pair in self.mcs_sinr_pairs]
        sinr_values = [pair['sinr_db'] for pair in self.mcs_sinr_pairs]
        
        # Create scatter plot
        plt.scatter(sinr_values, mcs_values, alpha=0.6, s=50)
        
        # Calculate and plot trend line
        z = np.polyfit(sinr_values, mcs_values, 1)
        p = np.poly1d(z)
        plt.plot(sorted(sinr_values), p(sorted(sinr_values)), "r--", alpha=0.8, linewidth=2)
        
        plt.xlabel('SINR (dB)')
        plt.ylabel('MCS Index')
        plt.title('MCS vs SINR Mapping from OTA Log')
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Data points: {len(self.mcs_sinr_pairs)}\n'
        stats_text += f'SINR range: {min(sinr_values):.1f} - {max(sinr_values):.1f} dB\n'
        stats_text += f'MCS range: {min(mcs_values)} - {max(mcs_values)}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plot_file = 'mcs_vs_sinr.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📈 Plot saved as {plot_file}")
        plt.show()

def main():
    args = parse_args()
    
    print("🔍 OTA Log Processor - MCS to SINR Mapping")
    print("=" * 50)
    
    # Create processor
    processor = OTALogProcessor(args.log_file, verbose=args.verbose)
    
    # Process the log file
    if not processor.process_log_file():
        return 1
    
    # Correlate MCS and SINR data
    processor.correlate_mcs_sinr(time_window=1.0)
    
    # Print summary
    processor.print_summary()
    
    # Save to CSV
    processor.save_to_csv(args.output_csv)
    
    # Create plot if requested
    if args.plot:
        processor.create_plot()
    
    print("\n🎉 Processing completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
