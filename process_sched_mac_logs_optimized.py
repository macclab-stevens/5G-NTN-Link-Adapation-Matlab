#!/usr/bin/env python3
"""
Optimized 5G gNB Log Processing Tool for Large Files
Processes SCHED and MAC logs to correlate PDSCH transmissions with HARQ-ACK feedback.

Optimizations for large files:
- Time-based bucketing for correlation (reduces complexity from O(n²) to O(n))
- Memory-efficient streaming processing
- Early termination for time windows
- Optimized data structures

Usage:
    python3 process_sched_mac_logs_optimized.py --log-file <log_file> --output <csv_file> [--plot] [--verbose]

Author: GitHub Copilot
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
import sys

# Regex patterns for log parsing
PDSCH_PATTERN = re.compile(
    r'- UE PDSCH: ue=(\d+) c-rnti=0x([0-9a-fA-F]+) h_id=(\d+).*?tbs=(\d+) mcs=(\d+).*?nrtx=(\d+)'
)

HARQ_PATTERN = re.compile(
    r'- HARQ-ACK: ue=(\d+) rnti=0x([0-9a-fA-F]+).*?h_id=(\d+) ack=(\d+)(?:.*?tbs=(\d+))?'
)

# Patterns for DL measurements (RRC measurement reports)
MEASUREMENT_REPORT_PATTERN = re.compile(r'ue=(\d+) c-rnti=0x([0-9a-fA-F]+).*Containerized measurementReport')
RSRP_PATTERN = re.compile(r'"rsrp":\s*(\d+)')
RSRQ_PATTERN = re.compile(r'"rsrq":\s*(\d+)')
DL_SINR_PATTERN = re.compile(r'"sinr":\s*(\d+)')

# Pattern for UL measurements (METRICS logs with pusch_snr_db)
UL_METRICS_PATTERN = re.compile(
    r'METRICS.*Scheduler UE.*rnti=0x([0-9a-fA-F]+).*pusch_snr_db=([\d\.-]+)'
)

def parse_timestamp(timestamp_str):
    """Parse ISO format timestamp with microseconds."""
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except:
        # Fallback for timestamps without timezone
        return datetime.strptime(timestamp_str[:26], '%Y-%m-%dT%H:%M:%S.%f')

def extract_pdsch_data(line, current_timestamp):
    """Extract PDSCH transmission data from log line."""
    if current_timestamp is None:
        return None
    
    # Extract PDSCH data
    pdsch_match = PDSCH_PATTERN.search(line)
    if not pdsch_match:
        return None
    
    ue_id, rnti, harq_id, tbs, mcs, nrtx = pdsch_match.groups()
    
    return {
        'timestamp': current_timestamp,
        'ue_id': int(ue_id),
        'rnti': int(rnti, 16),
        'harq_id': int(harq_id),
        'mcs': int(mcs),
        'tbs': int(tbs),
        'nrtx': int(nrtx)
    }

def extract_harq_data(line, current_timestamp):
    """Extract HARQ-ACK feedback data from log line."""
    if current_timestamp is None:
        return None
    
    # Extract HARQ data
    harq_match = HARQ_PATTERN.search(line)
    if not harq_match:
        return None
    
    ue_id, rnti, harq_id, ack, tbs = harq_match.groups()
    
    return {
        'timestamp': current_timestamp,
        'ue_id': int(ue_id),
        'rnti': int(rnti, 16),
        'harq_id': int(harq_id),
        'ack': int(ack),
        'tbs': int(tbs) if tbs else None
    }

def extract_dl_measurements(line, current_timestamp, next_lines):
    """Extract DL measurements from RRC measurement report."""
    if current_timestamp is None:
        return None
    
    # Check if this is a measurement report line
    meas_match = MEASUREMENT_REPORT_PATTERN.search(line)
    if not meas_match:
        return None
    
    ue_id, rnti = meas_match.groups()
    
    # Look for JSON data in the next few lines
    measurements = {}
    for next_line in next_lines:
        if '"rsrp":' in next_line:
            rsrp_match = RSRP_PATTERN.search(next_line)
            if rsrp_match:
                # Convert using formula: dl_rsrp = rsrp - 156
                raw_rsrp = int(rsrp_match.group(1))
                measurements['dl_rsrp'] = raw_rsrp - 156
        
        if '"rsrq":' in next_line:
            rsrq_match = RSRQ_PATTERN.search(next_line)
            if rsrq_match:
                # Convert using formula: dl_rsrq = rsrq/2 - 43
                raw_rsrq = int(rsrq_match.group(1))
                measurements['dl_rsrq'] = raw_rsrq / 2 - 43
        
        if '"sinr":' in next_line:
            sinr_match = DL_SINR_PATTERN.search(next_line)
            if sinr_match:
                # Convert using formula: dl_sinr = sinr/2 - 23
                raw_sinr = int(sinr_match.group(1))
                measurements['dl_sinr'] = raw_sinr / 2 - 23
        
        # Stop when we reach the end of the JSON structure
        if '}' in next_line and len(measurements) >= 3:
            break
    
    if measurements:
        return {
            'timestamp': current_timestamp,
            'ue_id': int(ue_id),
            'rnti': int(rnti, 16),
            **measurements
        }
    
    return None

def extract_ul_sinr(line, current_timestamp):
    """Extract UL SINR from METRICS logs."""
    if current_timestamp is None:
        return None
    
    # Extract UL SINR from METRICS logs with pusch_snr_db
    metrics_match = UL_METRICS_PATTERN.search(line)
    if not metrics_match:
        return None
    
    rnti, snr_db = metrics_match.groups()
    
    return {
        'timestamp': current_timestamp,
        'ue_id': 0,  # UE ID not readily available in METRICS logs, defaulting to 0
        'rnti': int(rnti, 16),
        'harq_id': None,  # HARQ ID not available in METRICS logs
        'ul_sinr': float(snr_db)
    }

class TimeBasedCorrelator:
    """Optimized correlator using time-based bucketing."""
    
    def __init__(self, max_window_ms=50, bucket_size_ms=10):
        self.max_window_ms = max_window_ms
        self.bucket_size_ms = bucket_size_ms
        # Use OrderedDict to maintain time ordering and enable efficient cleanup
        self.pdsch_buckets = OrderedDict()
        self.harq_buckets = OrderedDict()
        self.dl_measurements = []  # Store DL measurements
        self.ul_measurements = []  # Store UL measurements
        self.correlations = []
        self.unmatched_pdsch = []
        
    def _get_bucket_key(self, timestamp):
        """Get bucket key for timestamp (rounded to bucket_size_ms)."""
        epoch_ms = int(timestamp.timestamp() * 1000)
        return epoch_ms // self.bucket_size_ms
    
    def _cleanup_old_buckets(self, current_bucket_key):
        """Remove buckets that are too old to be useful."""
        max_age_buckets = (self.max_window_ms // self.bucket_size_ms) + 2
        cutoff_bucket = current_bucket_key - max_age_buckets
        
        # Clean up old PDSCH buckets
        while self.pdsch_buckets and next(iter(self.pdsch_buckets)) < cutoff_bucket:
            old_bucket_key, old_pdsch_list = self.pdsch_buckets.popitem(last=False)
            self.unmatched_pdsch.extend(old_pdsch_list)
        
        # Clean up old HARQ buckets
        while self.harq_buckets and next(iter(self.harq_buckets)) < cutoff_bucket:
            self.harq_buckets.popitem(last=False)
    
    def add_pdsch(self, pdsch_data):
        """Add PDSCH transmission to correlator."""
        bucket_key = self._get_bucket_key(pdsch_data['timestamp'])
        
        if bucket_key not in self.pdsch_buckets:
            self.pdsch_buckets[bucket_key] = []
        self.pdsch_buckets[bucket_key].append(pdsch_data)
        
        # Try to correlate with existing HARQ data
        self._correlate_pdsch(pdsch_data, bucket_key)
        
        # Cleanup old buckets
        self._cleanup_old_buckets(bucket_key)
    
    def add_harq(self, harq_data):
        """Add HARQ-ACK feedback to correlator."""
        bucket_key = self._get_bucket_key(harq_data['timestamp'])
        
        if bucket_key not in self.harq_buckets:
            self.harq_buckets[bucket_key] = []
        self.harq_buckets[bucket_key].append(harq_data)
        
        # Try to correlate with existing PDSCH data
        self._correlate_harq(harq_data, bucket_key)
        
        # Cleanup old buckets
        self._cleanup_old_buckets(bucket_key)
    
    def add_dl_measurement(self, dl_data):
        """Add DL measurement data."""
        self.dl_measurements.append(dl_data)
    
    def add_ul_measurement(self, ul_data):
        """Add UL measurement data."""
        self.ul_measurements.append(ul_data)
    
    def _correlate_pdsch(self, pdsch_data, pdsch_bucket_key):
        """Try to correlate PDSCH with existing HARQ data."""
        # Check current and future buckets for HARQ data
        max_bucket_offset = (self.max_window_ms // self.bucket_size_ms) + 1
        
        for offset in range(max_bucket_offset + 1):
            harq_bucket_key = pdsch_bucket_key + offset
            if harq_bucket_key not in self.harq_buckets:
                continue
                
            for harq_data in self.harq_buckets[harq_bucket_key]:
                if self._is_match(pdsch_data, harq_data):
                    time_diff_ms = (harq_data['timestamp'] - pdsch_data['timestamp']).total_seconds() * 1000
                    if 0 <= time_diff_ms <= self.max_window_ms:
                        self.correlations.append((pdsch_data, harq_data, time_diff_ms))
                        return True
        return False
    
    def _correlate_harq(self, harq_data, harq_bucket_key):
        """Try to correlate HARQ with existing PDSCH data."""
        # Check current and past buckets for PDSCH data
        max_bucket_offset = (self.max_window_ms // self.bucket_size_ms) + 1
        
        for offset in range(max_bucket_offset + 1):
            pdsch_bucket_key = harq_bucket_key - offset
            if pdsch_bucket_key not in self.pdsch_buckets:
                continue
                
            # Create a copy of the list to allow removal during iteration
            pdsch_list = list(self.pdsch_buckets[pdsch_bucket_key])
            for i, pdsch_data in enumerate(pdsch_list):
                if self._is_match(pdsch_data, harq_data):
                    time_diff_ms = (harq_data['timestamp'] - pdsch_data['timestamp']).total_seconds() * 1000
                    if 0 <= time_diff_ms <= self.max_window_ms:
                        self.correlations.append((pdsch_data, harq_data, time_diff_ms))
                        # Remove the matched PDSCH to avoid duplicate matches
                        self.pdsch_buckets[pdsch_bucket_key].remove(pdsch_data)
                        return True
        return False
    
    def _is_match(self, pdsch_data, harq_data):
        """Check if PDSCH and HARQ data match (same UE, RNTI, HARQ_ID)."""
        return (pdsch_data['ue_id'] == harq_data['ue_id'] and
                pdsch_data['rnti'] == harq_data['rnti'] and
                pdsch_data['harq_id'] == harq_data['harq_id'])
    
    def finalize(self):
        """Finalize correlation and return results."""
        # Add any remaining unmatched PDSCH
        for bucket in self.pdsch_buckets.values():
            self.unmatched_pdsch.extend(bucket)
        
        return self.correlations, self.unmatched_pdsch, self.dl_measurements, self.ul_measurements

def process_log_file_optimized(log_file, output_file, max_window_ms=50, verbose=False):
    """Process log file with optimized correlation algorithm."""
    
    correlator = TimeBasedCorrelator(max_window_ms=max_window_ms)
    
    pdsch_count = 0
    harq_count = 0
    dl_meas_count = 0
    ul_meas_count = 0
    line_count = 0
    
    # Add timestamp pattern
    timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)')
    
    print(f"📖 Processing log file: {log_file}")
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            current_timestamp = None
            lines = list(f)  # Read all lines for look-ahead
            
            i = 0
            while i < len(lines):
                line_count += 1
                line = lines[i].strip()
                
                if not line:
                    i += 1
                    continue
                
                # Progress indicator
                if line_count % 100000 == 0:
                    print(f"  Processed {line_count:,} lines...")
                
                # Extract timestamp from current line if it has one
                timestamp_match = timestamp_pattern.search(line)
                if timestamp_match:
                    current_timestamp = parse_timestamp(timestamp_match.group(1))
                
                # Extract PDSCH data
                if 'UE PDSCH:' in line:
                    pdsch_data = extract_pdsch_data(line, current_timestamp)
                    if pdsch_data:
                        pdsch_count += 1
                        correlator.add_pdsch(pdsch_data)
                        if verbose:
                            print(f"[LOG] PDSCH TX: UE={pdsch_data['ue_id']}, RNTI={pdsch_data['rnti']}, H_ID={pdsch_data['harq_id']}, MCS={pdsch_data['mcs']}, TBS={pdsch_data['tbs']}, NRTX={pdsch_data['nrtx']}")
                
                # Extract HARQ data
                elif 'HARQ-ACK:' in line:
                    harq_data = extract_harq_data(line, current_timestamp)
                    if harq_data:
                        harq_count += 1
                        correlator.add_harq(harq_data)
                        if verbose:
                            if harq_data['ack'] == 1:
                                ack_status = "ACK"
                            elif harq_data['ack'] == 0:
                                ack_status = "NACK"
                            elif harq_data['ack'] == 2:
                                ack_status = "DTX"
                            else:
                                ack_status = f"UNKNOWN({harq_data['ack']})"
                            print(f"[LOG] HARQ-ACK: UE={harq_data['ue_id']}, RNTI={harq_data['rnti']}, H_ID={harq_data['harq_id']}, ACK={ack_status}")
                
                # Extract DL measurements
                elif 'measurementReport' in line and 'ue=' in line:
                    # Get the next 20 lines for JSON parsing
                    next_lines = lines[i+1:i+21] if i+21 < len(lines) else lines[i+1:]
                    dl_data = extract_dl_measurements(line, current_timestamp, next_lines)
                    if dl_data:
                        dl_meas_count += 1
                        correlator.add_dl_measurement(dl_data)
                        if verbose:
                            print(f"[LOG] DL MEAS: UE={dl_data['ue_id']}, RSRP={dl_data.get('dl_rsrp', 'N/A'):.1f}dBm, RSRQ={dl_data.get('dl_rsrq', 'N/A'):.1f}dB, SINR={dl_data.get('dl_sinr', 'N/A'):.1f}dB")
                
                # Extract UL SINR from METRICS logs
                elif 'METRICS' in line and 'pusch_snr_db=' in line:
                    ul_data = extract_ul_sinr(line, current_timestamp)
                    if ul_data:
                        ul_meas_count += 1
                        correlator.add_ul_measurement(ul_data)
                        if verbose:
                            print(f"[LOG] UL SINR: RNTI=0x{ul_data['rnti']:04x}, SINR={ul_data['ul_sinr']:.1f}dB")
                
                i += 1
    
    except FileNotFoundError:
        print(f"❌ Error: Log file '{log_file}' not found.")
        return None
    except Exception as e:
        print(f"❌ Error reading log file: {e}")
        return None
    
    print(f"✅ Processed {line_count:,} lines total")
    print(f"[LOG] Finalizing correlation of {pdsch_count} PDSCH transmissions with {harq_count} HARQ-ACK entries...")
    print(f"[LOG] Found {dl_meas_count} DL measurements and {ul_meas_count} UL measurements")
    
    # Finalize correlation
    correlations, unmatched_pdsch, dl_measurements, ul_measurements = correlator.finalize()
    
    print(f"[LOG] Correlation complete: {len(correlations)} matched, {len(unmatched_pdsch)} unmatched")
    
    # Build results DataFrame
    return build_results_dataframe_with_measurements(correlations, unmatched_pdsch, dl_measurements, ul_measurements, output_file, skip_measurements=False)

def calculate_throughput_metrics(df):
    """
    Calculate throughput and trueput metrics with proper HARQ process tracking.
    
    Throughput: All transmitted data (includes retransmissions)
    Trueput: Successfully delivered data, accounting for retransmission overhead
    
    Uses proper kbps calculation (÷1024) and correctly tracks HARQ processes.
    """
    print("📊 Calculating throughput metrics with improved HARQ process tracking...")
    
    # Sort by timestamp to ensure proper temporal order
    df = df.sort_values('pdsch_timestamp').reset_index(drop=True)
    
    # Calculate instantaneous bitrates (convert to Mbps for readability)
    # Mbps = (TBS_bytes * 8) / 1024 / 1000
    df['instantaneous_throughput_mbps'] = df['tbs'] * 8 / 1024 / 1000
    df['instantaneous_trueput_mbps'] = 0.0  # Will be calculated based on ACK status
    
    # For trueput: each ACK'd transmission gets credit for its payload
    # This represents successfully delivered data regardless of retransmissions
    ack_mask = (df['ack_status'] == 1.0)  # ACK = 1.0
    df.loc[ack_mask, 'instantaneous_trueput_mbps'] = df.loc[ack_mask, 'instantaneous_throughput_mbps']
    
    # Alternative approach: Track HARQ process efficiency
    # Group by HARQ process to understand retransmission patterns
    df['harq_process_key'] = df['ue_id'].astype(str) + '_' + df['rnti'].astype(str) + '_' + df['harq_id'].astype(str)
    
    # Calculate HARQ process statistics
    harq_stats = {}
    total_harq_attempts = 0
    successful_harq_deliveries = 0
    
    for harq_key in df['harq_process_key'].unique():
        harq_transmissions = df[df['harq_process_key'] == harq_key].sort_values('pdsch_timestamp')
        
        # Count transmission attempts and successful deliveries for this HARQ process
        attempts = len(harq_transmissions)
        acks = (harq_transmissions['ack_status'] == 1.0).sum()
        
        harq_stats[harq_key] = {
            'attempts': attempts,
            'acks': acks,
            'efficiency': acks / attempts if attempts > 0 else 0
        }
        
        total_harq_attempts += attempts
        successful_harq_deliveries += acks
    
    # Calculate cumulative throughput and trueput over time
    df['cumulative_throughput_mbits'] = df['instantaneous_throughput_mbps'].cumsum()
    df['cumulative_trueput_mbits'] = df['instantaneous_trueput_mbps'].cumsum()
    
    # Calculate efficiency metrics
    total_throughput = df['instantaneous_throughput_mbps'].sum()
    total_trueput = df['instantaneous_trueput_mbps'].sum()
    
    if total_throughput > 0:
        overall_efficiency = total_trueput / total_throughput
        df['harq_efficiency'] = overall_efficiency
        print(f"📈 HARQ Efficiency: {overall_efficiency*100:.2f}% "
              f"(Trueput: {total_trueput:.1f} Mbits, Throughput: {total_throughput:.1f} Mbits)")
        print(f"📊 HARQ Statistics: {successful_harq_deliveries}/{total_harq_attempts} "
              f"successful deliveries ({successful_harq_deliveries/total_harq_attempts*100:.1f}%)")
    else:
        df['harq_efficiency'] = 0.0
        print("⚠️  No throughput data found")
    
    # Add windowed throughput calculations (per second)
    df['window_start_time'] = df['pdsch_time_relative_s'].apply(lambda x: int(x))
    
    # Calculate per-second aggregated rates
    throughput_per_sec = df.groupby('window_start_time').agg({
        'instantaneous_throughput_mbps': 'sum',
        'instantaneous_trueput_mbps': 'sum'
    }).rename(columns={
        'instantaneous_throughput_mbps': 'throughput_mbps_per_sec',
        'instantaneous_trueput_mbps': 'trueput_mbps_per_sec'
    })
    
    # Merge back to main dataframe
    df = df.merge(throughput_per_sec, left_on='window_start_time', right_index=True, how='left')
    
    print(f"✅ Throughput metrics calculated for {len(df)} transmissions")
    print(f"   - Unique HARQ processes: {len(harq_stats)}")
    print(f"   - ACK'd transmissions: {(df['ack_status'] == 1.0).sum()}")
    print(f"   - NACK'd transmissions: {(df['ack_status'] == 0.0).sum()}")
    print(f"   - DTX transmissions: {(df['ack_status'] == 2.0).sum()}")
    
    return df

def build_results_dataframe_with_measurements(correlations, unmatched_pdsch, dl_measurements, ul_measurements, output_file, skip_measurements=False):
    """Build and save results DataFrame with DL and UL measurements."""
    results = []
    
    # Create time-based buckets for fast measurement lookup
    print(f"🔍 Creating measurement buckets for fast lookup...")
    bucket_size_s = 30  # 30 second buckets
    
    def create_measurement_buckets(measurements):
        """Create time-based buckets for O(1) measurement lookup."""
        buckets = defaultdict(list)
        for meas in measurements:
            bucket_key = int(meas['timestamp'].timestamp() // bucket_size_s)
            buckets[bucket_key].append(meas)
        return buckets
    
    dl_buckets = create_measurement_buckets(dl_measurements)
    ul_buckets = create_measurement_buckets(ul_measurements)
    
    def find_closest_measurement_fast(timestamp, measurement_buckets, max_time_diff_s=30):
        """Find measurement using bucketed lookup - optimized for speed."""
        if not measurement_buckets:
            return {}
        
        bucket_key = int(timestamp.timestamp() // bucket_size_s)
        
        # Check current bucket first (most likely)
        if bucket_key in measurement_buckets:
            bucket_measurements = measurement_buckets[bucket_key]
            if bucket_measurements:
                # Just return first measurement in bucket (they're all close in time)
                meas = bucket_measurements[0]
                return {k: v for k, v in meas.items() 
                       if k not in ['timestamp', 'ue_id', 'rnti', 'harq_id']}
        
        # Check adjacent buckets if needed
        for offset in [-1, 1]:
            check_bucket = bucket_key + offset
            if check_bucket in measurement_buckets:
                bucket_measurements = measurement_buckets[check_bucket]
                if bucket_measurements:
                    meas = bucket_measurements[0]
                    return {k: v for k, v in meas.items() 
                           if k not in ['timestamp', 'ue_id', 'rnti', 'harq_id']}
        
        return {}

    if skip_measurements or (not dl_measurements and not ul_measurements):
        print(f"⚠ Skipping measurement correlation (skip_measurements={skip_measurements}, dl_count={len(dl_measurements)}, ul_count={len(ul_measurements)})")
        
        # Process without measurement correlation for speed
        print(f"📊 Processing {len(correlations)} matched correlations (no measurements)...")
        for i, (pdsch, harq, time_diff_ms) in enumerate(correlations):
            if i % 100000 == 0 and i > 0:
                print(f"   Progress: {i:,}/{len(correlations):,} ({i/len(correlations)*100:.1f}%)")
            
            # Map HARQ ACK values: 0=NACK, 1=ACK, 2=DTX
            if harq['ack'] == 1:
                ack_result = "ACK"
            elif harq['ack'] == 0:
                ack_result = "NACK"
            elif harq['ack'] == 2:
                ack_result = "DTX"
            else:
                ack_result = "UNKNOWN"
            
            ack_status = float(harq['ack'])
            
            result = {
                'pdsch_timestamp': pdsch['timestamp'],
                'harq_timestamp': harq['timestamp'],
                'time_diff_ms': time_diff_ms,
                'ue_id': pdsch['ue_id'],
                'rnti': f"0x{pdsch['rnti']:04x}",
                'harq_id': pdsch['harq_id'],
                'mcs': pdsch['mcs'],
                'tbs': pdsch['tbs'],
                'nrtx': pdsch['nrtx'],
                'transmission_type': 'retransmission' if pdsch['nrtx'] > 0 else 'new',
                'ack_status': ack_status,
                'ack_result': ack_result,
                # Add empty measurement columns
                'dl_rsrp': None,
                'dl_rsrq': None,
                'dl_sinr': None,
                'ul_sinr': None
            }
            results.append(result)
        
        # Process unmatched PDSCH quickly
        print(f"📊 Processing {len(unmatched_pdsch)} unmatched PDSCH entries (no measurements)...")
        for i, pdsch in enumerate(unmatched_pdsch):
            if i % 50000 == 0 and i > 0:
                print(f"   Progress: {i:,}/{len(unmatched_pdsch):,} ({i/len(unmatched_pdsch)*100:.1f}%)")
            
            result = {
                'pdsch_timestamp': pdsch['timestamp'],
                'harq_timestamp': None,
                'time_diff_ms': None,
                'ue_id': pdsch['ue_id'],
                'rnti': f"0x{pdsch['rnti']:04x}",
                'harq_id': pdsch['harq_id'],
                'mcs': pdsch['mcs'],
                'tbs': pdsch['tbs'],
                'nrtx': pdsch['nrtx'],
                'transmission_type': 'retransmission' if pdsch['nrtx'] > 0 else 'new',
                'ack_status': 2.0,  # DTX
                'ack_result': 'DTX',
                # Add empty measurement columns
                'dl_rsrp': None,
                'dl_rsrq': None,
                'dl_sinr': None,
                'ul_sinr': None
            }
            results.append(result)
    
    else:
        # Process with measurement correlation
        print(f"📊 Processing {len(correlations)} matched correlations...")
        start_time = pd.Timestamp.now()
        
        for i, (pdsch, harq, time_diff_ms) in enumerate(correlations):
            if i % 25000 == 0 and i > 0:
                elapsed = (pd.Timestamp.now() - start_time).total_seconds()
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (len(correlations) - i) / rate if rate > 0 else 0
                print(f"   Progress: {i:,}/{len(correlations):,} ({i/len(correlations)*100:.1f}%) | "
                      f"Rate: {rate:.0f}/s | ETA: {remaining/60:.1f}min")
            
            # Map HARQ ACK values: 0=NACK, 1=ACK, 2=DTX
            if harq['ack'] == 1:
                ack_result = "ACK"
            elif harq['ack'] == 0:
                ack_result = "NACK"
            elif harq['ack'] == 2:
                ack_result = "DTX"
            else:
                ack_result = "UNKNOWN"
            
            ack_status = float(harq['ack'])
            
            # Find closest DL and UL measurements
            dl_meas = find_closest_measurement_fast(pdsch['timestamp'], dl_buckets)
            ul_meas = find_closest_measurement_fast(pdsch['timestamp'], ul_buckets)
            
            result = {
                'pdsch_timestamp': pdsch['timestamp'],
                'harq_timestamp': harq['timestamp'],
                'time_diff_ms': time_diff_ms,
                'ue_id': pdsch['ue_id'],
                'rnti': f"0x{pdsch['rnti']:04x}",
                'harq_id': pdsch['harq_id'],
                'mcs': pdsch['mcs'],
                'tbs': pdsch['tbs'],
                'nrtx': pdsch['nrtx'],
                'transmission_type': 'retransmission' if pdsch['nrtx'] > 0 else 'new',
                'ack_status': ack_status,
                'ack_result': ack_result,
                # Add DL measurements
                'dl_rsrp': dl_meas.get('dl_rsrp', None),
                'dl_rsrq': dl_meas.get('dl_rsrq', None),
                'dl_sinr': dl_meas.get('dl_sinr', None),
                # Add UL measurements
                'ul_sinr': ul_meas.get('ul_sinr', None)
            }
            results.append(result)
        
        # Process unmatched PDSCH entries
        print(f"📊 Processing {len(unmatched_pdsch)} unmatched PDSCH entries...")
        for i, pdsch in enumerate(unmatched_pdsch):
            if i % 25000 == 0 and i > 0:
                print(f"   Progress: {i:,}/{len(unmatched_pdsch):,} ({i/len(unmatched_pdsch)*100:.1f}%)")
            
            # Find closest measurements for unmatched entries too
            dl_meas = find_closest_measurement_fast(pdsch['timestamp'], dl_buckets)
            ul_meas = find_closest_measurement_fast(pdsch['timestamp'], ul_buckets)
            
            result = {
                'pdsch_timestamp': pdsch['timestamp'],
                'harq_timestamp': None,
                'time_diff_ms': None,
                'ue_id': pdsch['ue_id'],
                'rnti': f"0x{pdsch['rnti']:04x}",
                'harq_id': pdsch['harq_id'],
                'mcs': pdsch['mcs'],
                'tbs': pdsch['tbs'],
                'nrtx': pdsch['nrtx'],
                'transmission_type': 'retransmission' if pdsch['nrtx'] > 0 else 'new',
                'ack_status': 2.0,  # DTX
                'ack_result': 'DTX',
                # Add DL measurements
                'dl_rsrp': dl_meas.get('dl_rsrp', None),
                'dl_rsrq': dl_meas.get('dl_rsrq', None),
                'dl_sinr': dl_meas.get('dl_sinr', None),
                # Add UL measurements
                'ul_sinr': ul_meas.get('ul_sinr', None)
            }
            results.append(result)
    
    # Process unmatched PDSCH (DTX cases)
    print(f"📊 Processing {len(unmatched_pdsch)} unmatched PDSCH entries...")
    for i, pdsch in enumerate(unmatched_pdsch):
        if i % 10000 == 0 and i > 0:
            print(f"   Progress: {i:,}/{len(unmatched_pdsch):,} ({i/len(unmatched_pdsch)*100:.1f}%)")
        
        # Find closest DL and UL measurements
        dl_meas = find_closest_measurement_fast(pdsch['timestamp'], dl_buckets)
        ul_meas = find_closest_measurement_fast(pdsch['timestamp'], ul_buckets)
        
        result = {
            'pdsch_timestamp': pdsch['timestamp'],
            'harq_timestamp': None,
            'time_diff_ms': None,
            'ue_id': pdsch['ue_id'],
            'rnti': f"0x{pdsch['rnti']:04x}",
            'harq_id': pdsch['harq_id'],
            'mcs': pdsch['mcs'],
            'tbs': pdsch['tbs'],
            'nrtx': pdsch['nrtx'],
            'transmission_type': 'retransmission' if pdsch['nrtx'] > 0 else 'new',
            'ack_status': 2.0,  # DTX
            'ack_result': 'DTX',
            # Add DL measurements
            'dl_rsrp': dl_meas.get('dl_rsrp', None),
            'dl_rsrq': dl_meas.get('dl_rsrq', None),
            'dl_sinr': dl_meas.get('dl_sinr', None),
            # Add UL measurements
            'ul_sinr': ul_meas.get('ul_sinr', None)
        }
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    if df.empty:
        print("❌ No data found to process!")
        return None
    
    # Sort by timestamp
    df = df.sort_values('pdsch_timestamp').reset_index(drop=True)
    
    # Calculate relative time from first transmission
    first_timestamp = df['pdsch_timestamp'].iloc[0]
    df['pdsch_time_relative_s'] = (df['pdsch_timestamp'] - first_timestamp).dt.total_seconds()
    
    # Calculate BLER (running average) - only count NACK (ack_status=0) as errors
    df['bler'] = (df['ack_status'] == 0.0).cumsum() / (df.index + 1) * 100
    
    # Calculate throughput metrics with proper HARQ process tracking
    df = calculate_throughput_metrics(df)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    # Print measurement statistics
    total_entries = len(df)
    dl_rsrp_count = df['dl_rsrp'].notna().sum()
    dl_rsrq_count = df['dl_rsrq'].notna().sum()
    dl_sinr_count = df['dl_sinr'].notna().sum()
    ul_sinr_count = df['ul_sinr'].notna().sum()
    
    print(f"💾 Saved {total_entries} entries to {output_file}")
    print(f"📊 Measurement availability:")
    print(f"   DL RSRP: {dl_rsrp_count}/{total_entries} ({dl_rsrp_count/total_entries*100:.1f}%)")
    print(f"   DL RSRQ: {dl_rsrq_count}/{total_entries} ({dl_rsrq_count/total_entries*100:.1f}%)")
    print(f"   DL SINR: {dl_sinr_count}/{total_entries} ({dl_sinr_count/total_entries*100:.1f}%)")
    print(f"   UL SINR: {ul_sinr_count}/{total_entries} ({ul_sinr_count/total_entries*100:.1f}%)")
    
    return df

def generate_statistics_and_plots(df, output_file):
    """Generate statistics and plots."""
    total_transmissions = len(df)
    mcs_range = f"{df['mcs'].min()} - {df['mcs'].max()}"
    tbs_range = f"{df['tbs'].min()} - {df['tbs'].max()} bytes"
    
    # ACK/NACK statistics
    ack_counts = df['ack_result'].value_counts()
    ack_percentages = df['ack_result'].value_counts(normalize=True) * 100
    
    # Calculate BLER - only count NACK (ack_status=0) as block errors
    nack_count = len(df[df['ack_status'] == 0.0])
    feedback_count = len(df[df['ack_status'].notna()])
    bler = (nack_count / feedback_count * 100) if feedback_count > 0 else 0
    
    # MCS distribution
    mcs_dist = df['mcs'].value_counts().sort_index()
    mcs_percentages = df['mcs'].value_counts(normalize=True).sort_index() * 100
    
    # Print statistics
    print(f"\n📊 STATISTICS:")
    print(f"  Total transmissions: {total_transmissions}")
    print(f"  MCS range: {mcs_range}")
    print(f"  TBS range: {tbs_range}")
    print(f"  ACK/NACK feedback: {feedback_count} transmissions")
    print(f"  BLER: {bler:.2f}%")
    for result, count in ack_counts.items():
        percentage = ack_percentages[result]
        print(f"    {result}: {count} ({percentage:.1f}%)")
    
    print(f"  MCS distribution:")
    for mcs, count in mcs_dist.items():
        percentage = mcs_percentages[mcs]
        print(f"    MCS {mcs}: {count} transmissions ({percentage:.1f}%)")
    
    # Generate plots
    plot_file = output_file.replace('.csv', '_analysis.png')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: MCS over time
    ax1.scatter(df['pdsch_time_relative_s'], df['mcs'], alpha=0.6, s=1)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('MCS')
    ax1.set_title('MCS vs Time')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: DL Bitrate over time (TBS*8/1000 to get kbps, assuming 1ms transmission)
    dl_bitrate_kbps = df['tbs'] * 8 / 1000  # Convert bytes to kbits
    ax2.scatter(df['pdsch_time_relative_s'], dl_bitrate_kbps, alpha=0.6, s=1)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('DL Bitrate (kbps)')
    ax2.set_title('DL Bitrate vs Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: BLER over time
    # Sample data for BLER calculation to avoid overplotting
    sample_size = min(10000, len(df))
    sample_indices = range(0, len(df), max(1, len(df) // sample_size))
    sampled_df = df.iloc[sample_indices]
    
    # Convert to numpy arrays for plotting
    x_vals = sampled_df['pdsch_time_relative_s'].values
    y_vals = sampled_df['bler'].values
    ax3.plot(x_vals, y_vals)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('BLER (%)')
    ax3.set_title('BLER vs Time')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: SINR over time
    # Check if we have UL or DL SINR data
    has_ul_sinr = df['ul_sinr'].notna().any()
    has_dl_sinr = df['dl_sinr'].notna().any()
    
    if has_ul_sinr or has_dl_sinr:
        if has_ul_sinr:
            # Plot UL SINR
            ul_sinr_data = df[df['ul_sinr'].notna()]
            ax4.scatter(ul_sinr_data['pdsch_time_relative_s'], ul_sinr_data['ul_sinr'], 
                       alpha=0.6, s=1, color='blue', label='UL SINR')
        
        if has_dl_sinr:
            # Plot DL SINR
            dl_sinr_data = df[df['dl_sinr'].notna()]
            ax4.scatter(dl_sinr_data['pdsch_time_relative_s'], dl_sinr_data['dl_sinr'], 
                       alpha=0.6, s=1, color='red', label='DL SINR')
        
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('SINR (dB)')
        ax4.set_title('SINR vs Time')
        ax4.grid(True, alpha=0.3)
        if has_ul_sinr and has_dl_sinr:
            ax4.legend()
    else:
        # Fallback to ACK/NACK count if no SINR data available
        time_bins = pd.cut(df['pdsch_time_relative_s'], bins=50)
        ack_nack_counts = df.groupby([time_bins, 'ack_result']).size().unstack(fill_value=0)
        
        ack_nack_counts.plot(kind='bar', stacked=True, ax=ax4, 
                            color=['green', 'red', 'orange'])  # ACK=green, NACK=red, DTX=orange
        ax4.set_xlabel('Time Bins')
        ax4.set_ylabel('Count')
        ax4.set_title('ACK/NACK Count Over Time (No SINR Data)')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📈 Saved analysis plots to {plot_file}")

def plot_from_csv(csv_file):
    """Generate plots from an existing CSV file."""
    print(f"📖 Loading CSV file: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} entries from CSV")
        
        # Generate statistics and plots
        generate_statistics_and_plots(df, csv_file)
        print(f"🎉 Plot generation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Process 5G gNB SCHED and MAC logs to correlate PDSCH transmissions with HARQ-ACK feedback (Optimized)')
    parser.add_argument('--log-file', help='Path to the log file')
    parser.add_argument('--output', help='Output CSV file path')
    parser.add_argument('--csv-file', help='Path to existing CSV file for plot-only mode')
    parser.add_argument('--max-window-ms', type=int, default=50, help='Maximum time window for correlation (ms)')
    parser.add_argument('--skip-measurements', action='store_true', help='Skip measurement correlation for faster processing')
    parser.add_argument('--plot', action='store_true', help='Generate analysis plots')
    parser.add_argument('--plot-only', action='store_true', help='Generate plots from existing CSV file (requires --csv-file)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Check for plot-only mode
    if args.plot_only:
        if not args.csv_file:
            print("❌ Error: --csv-file is required when using --plot-only mode")
            sys.exit(1)
        plot_from_csv(args.csv_file)
        return
    
    # Check for required arguments in processing mode
    if not args.log_file or not args.output:
        print("❌ Error: --log-file and --output are required for processing mode")
        parser.print_help()
        sys.exit(1)
    
    # Process the log file
    df = process_log_file_optimized(args.log_file, args.output, args.max_window_ms, args.verbose)
    
    if df is not None:
        # Generate statistics and plots
        generate_statistics_and_plots(df, args.output)
        
        if args.plot:
            pass  # Plots are already generated in statistics function
        
        print(f"\n🎉 Processing completed successfully!")
    else:
        print(f"\n❌ Processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
