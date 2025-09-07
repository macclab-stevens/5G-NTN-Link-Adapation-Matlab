#!/usr/bin/python3
"""
Final Optimized Noise Generator - Best Performance
Combines the benefits of all approaches:
- Pre-generates manageable chunks
- Minimizes gaps between transmissions  
- Supports real-time gain updates
- Memory efficient
"""

import uhd
import numpy as np
import argparse
import time
import os
import threading
from scipy import signal

PLATFORM = "b200"
SERIAL = "31577EF"
CONTROL_FILE = "/tmp/usrp_gain.txt"

def parse_args():
    parser = argparse.ArgumentParser(description="Final Optimized USRP Noise Generator")
    parser.add_argument('--center-freq', type=float, default=3410.1e6, help='Center frequency (Hz)')
    parser.add_argument('--sample-rate', type=float, default=20e6, help='Sample rate (Hz)')
    parser.add_argument('--gain', type=float, default=60, help='Initial TX gain (dB)')
    parser.add_argument('--noise-power', type=float, default=1.0, help='Noise power (linear, not dB)')
    parser.add_argument('--duration', type=float, default=600, help='Duration in seconds')
    return parser.parse_args()

def create_control_file(initial_gain):
    """Create initial control file"""
    try:
        with open(CONTROL_FILE, 'w') as f:
            f.write(str(initial_gain))
        print(f"📝 Control file: {CONTROL_FILE}")
        print(f"   Update gain: echo 'NEW_GAIN' > {CONTROL_FILE}")
    except IOError:
        print(f"⚠ Warning: Could not create {CONTROL_FILE}")

def generate_base_patterns(sample_rate, noise_power, pattern_duration=5.0):
    """Generate multiple base noise patterns to cycle through"""
    print(f"🔊 Generating base noise patterns ({pattern_duration}s each)...")
    
    pattern_samples = int(pattern_duration * sample_rate)
    num_patterns = 4  # Generate 4 different patterns to cycle through
    
    # Filter design (once)
    oversample_factor = 2
    oversample_rate = sample_rate * oversample_factor
    nyquist = oversample_rate / 2
    cutoff = 10e6  # 10 MHz for 20 MHz bandwidth
    normalized_cutoff = cutoff / nyquist
    b, a = signal.butter(6, normalized_cutoff, btype='low', analog=False)
    
    patterns = []
    for i in range(num_patterns):
        print(f"   Generating pattern {i+1}/{num_patterns}...")
        
        # Generate oversampled noise
        oversample_samples = pattern_samples * oversample_factor
        noise_real = np.random.normal(0, np.sqrt(noise_power/2), oversample_samples)
        noise_imag = np.random.normal(0, np.sqrt(noise_power/2), oversample_samples)
        noise_oversample = (noise_real + 1j * noise_imag).astype(np.complex64)
        
        # Filter and decimate
        noise_filtered = signal.filtfilt(b, a, noise_oversample)
        pattern = noise_filtered[::oversample_factor][:pattern_samples].astype(np.complex64)
        patterns.append(pattern)
    
    print(f"✅ Generated {num_patterns} patterns, {len(patterns[0]):,} samples each")
    return patterns

class NoiseTransmitter:
    def __init__(self, args):
        self.args = args
        self.current_gain = args.gain
        
        # Initialize USRP
        print("🔧 Initializing USRP...")
        usrp_addr = f"type={PLATFORM},serial={SERIAL}"
        self.usrp = uhd.usrp.MultiUSRP(usrp_addr)
        
        # Configure USRP
        self.usrp.set_tx_rate(args.sample_rate, 0)
        self.usrp.set_tx_freq(args.center_freq, 0)
        self.usrp.set_tx_bandwidth(20e6, 0)
        self.usrp.set_tx_gain(args.gain, 0)
        print("✅ USRP configured")
        
        # Generate base patterns
        self.patterns = generate_base_patterns(args.sample_rate, args.noise_power)
        self.pattern_index = 0
        
    def get_gain_from_file(self):
        """Read gain from control file"""
        try:
            if os.path.exists(CONTROL_FILE):
                with open(CONTROL_FILE, 'r') as f:
                    content = f.read().strip()
                    if content:
                        return float(content)
        except (ValueError, IOError):
            pass
        return self.current_gain
    
    def update_gain_if_needed(self):
        """Check and update gain if changed"""
        new_gain = self.get_gain_from_file()
        if new_gain != self.current_gain:
            print(f"📶 Gain update: {self.current_gain} → {new_gain} dB")
            self.current_gain = new_gain
            try:
                self.usrp.set_tx_gain(new_gain, 0)
                print(f"✅ Gain updated to {new_gain} dB")
                return True
            except Exception as e:
                print(f"❌ Gain update failed: {e}")
                return False
        return False
    
    def get_next_pattern(self):
        """Get next pattern in sequence"""
        pattern = self.patterns[self.pattern_index]
        self.pattern_index = (self.pattern_index + 1) % len(self.patterns)
        return pattern
    
    def create_long_waveform(self, target_duration):
        """Create a long waveform by concatenating patterns"""
        pattern_duration = len(self.patterns[0]) / self.args.sample_rate
        num_patterns_needed = int(np.ceil(target_duration / pattern_duration))
        
        waveform_parts = []
        for i in range(num_patterns_needed):
            pattern = self.get_next_pattern()
            waveform_parts.append(pattern)
        
        # Concatenate all patterns
        full_waveform = np.concatenate(waveform_parts)
        
        # Trim to exact duration
        target_samples = int(target_duration * self.args.sample_rate)
        if len(full_waveform) > target_samples:
            full_waveform = full_waveform[:target_samples]
            
        return full_waveform
    
    def transmit_optimized(self):
        """Optimized transmission approach"""
        print("\n" + "=" * 70)
        print("🚀 STARTING OPTIMIZED TRANSMISSION")
        print("=" * 70)
        
        # For durations up to 1 minute, generate the full waveform
        # For longer durations, use chunks to manage memory
        if self.args.duration <= 60:
            return self.transmit_single_waveform()
        else:
            return self.transmit_chunked()
    
    def transmit_single_waveform(self):
        """Single waveform transmission for shorter durations"""
        print(f"📡 Single waveform mode ({self.args.duration}s)")
        
        # Create long waveform
        print("🔊 Creating waveform from patterns...")
        waveform = self.create_long_waveform(self.args.duration)
        memory_mb = len(waveform) * 8 / 1e6
        print(f"💾 Waveform size: {len(waveform):,} samples ({memory_mb:.1f} MB)")
        
        # Start gain monitoring in background
        gain_monitor = GainMonitorThread(self)
        gain_monitor.start()
        
        try:
            print("📡 Starting transmission...")
            start_time = time.time()
            
            self.usrp.send_waveform(waveform, self.args.duration, self.args.center_freq,
                                  self.args.sample_rate, [0], self.current_gain)
            
            end_time = time.time()
            actual_duration = end_time - start_time
            print(f"✅ Transmission completed in {actual_duration:.2f}s")
            
        finally:
            gain_monitor.stop()
    
    def transmit_chunked(self):
        """Chunked transmission for longer durations"""
        chunk_duration = 30.0  # 30-second chunks
        print(f"📦 Chunked mode ({chunk_duration}s chunks)")
        
        elapsed = 0
        chunk_num = 0
        total_chunks = int(np.ceil(self.args.duration / chunk_duration))
        
        try:
            while elapsed < self.args.duration:
                remaining = self.args.duration - elapsed
                current_chunk_duration = min(chunk_duration, remaining)
                chunk_num += 1
                
                print(f"\n🔊 Chunk {chunk_num}/{total_chunks} ({current_chunk_duration}s)")
                
                # Check for gain updates between chunks
                self.update_gain_if_needed()
                
                # Create chunk waveform
                chunk_waveform = self.create_long_waveform(current_chunk_duration)
                
                # Transmit chunk
                start_time = time.time()
                self.usrp.send_waveform(chunk_waveform, current_chunk_duration,
                                      self.args.center_freq, self.args.sample_rate,
                                      [0], self.current_gain)
                end_time = time.time()
                
                elapsed += (end_time - start_time)
                print(f"✅ Chunk {chunk_num} completed. Total elapsed: {elapsed/60:.2f}min")
                
        except KeyboardInterrupt:
            print(f"\n⚠ Interrupted after {elapsed/60:.2f} minutes")

class GainMonitorThread:
    """Background gain monitoring for single waveform mode"""
    def __init__(self, transmitter):
        self.transmitter = transmitter
        self.stop_flag = threading.Event()
        self.thread = None
        
    def start(self):
        self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.thread.start()
        print("🔄 Background gain monitoring started")
        
    def stop(self):
        if self.thread:
            self.stop_flag.set()
            self.thread.join(timeout=1.0)
        
    def monitor_loop(self):
        while not self.stop_flag.wait(0.2):  # Check every 200ms
            self.transmitter.update_gain_if_needed()

def main():
    args = parse_args()
    
    print("=" * 70)
    print("🔊 FINAL OPTIMIZED NOISE GENERATOR")
    print("=" * 70)
    print(f"📻 Frequency: {args.center_freq/1e6:.1f} MHz")
    print(f"📊 Sample Rate: {args.sample_rate/1e6:.1f} MHz")
    print(f"📐 Bandwidth: 20 MHz (filtered)")
    print(f"⏱ Duration: {args.duration/60:.1f} minutes")
    print(f"📶 Initial Gain: {args.gain} dB")
    print("=" * 70)
    
    # Create control file
    create_control_file(args.gain)
    
    # Create transmitter and run
    transmitter = NoiseTransmitter(args)
    
    try:
        transmitter.transmit_optimized()
        print(f"\n🎉 TRANSMISSION COMPLETED SUCCESSFULLY!")
        
    except KeyboardInterrupt:
        print(f"\n⚠ Transmission interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Cleanup
        try:
            os.remove(CONTROL_FILE)
            print(f"🧹 Cleaned up {CONTROL_FILE}")
        except OSError:
            pass
        
        print("🏁 Done!")

if __name__ == "__main__":
    main()
