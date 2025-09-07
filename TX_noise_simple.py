#!/usr/bin/python3
"""
Simple 10-Minute Noise Generator using send_waveform
Most reliable approach - generates long waveform and transmits it once
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
    parser = argparse.ArgumentParser(description="Simple 10-Minute USRP Noise Generator")
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

def generate_long_waveform(duration, sample_rate, noise_power):
    """Generate filtered noise waveform for the entire duration"""
    print(f"🔊 Generating {duration/60:.1f}-minute noise waveform...")
    
    total_samples = int(duration * sample_rate)
    memory_gb = total_samples * 8 / 1e9  # complex64 = 8 bytes per sample
    print(f"   Samples: {total_samples:,} ({memory_gb:.2f} GB)")
    
    if memory_gb > 4:
        print("⚠ Warning: Large waveform may use significant memory")
    
    # Generate in chunks to manage memory
    chunk_duration = 5.0  # 5-second chunks
    chunk_samples = int(chunk_duration * sample_rate)
    num_chunks = int(np.ceil(total_samples / chunk_samples))
    
    # Design filter for 20 MHz bandwidth
    oversample_factor = 2
    oversample_rate = sample_rate * oversample_factor
    nyquist = oversample_rate / 2
    cutoff = 10e6  # 10 MHz
    normalized_cutoff = cutoff / nyquist
    b, a = signal.butter(6, normalized_cutoff, btype='low', analog=False)
    
    # Pre-allocate waveform
    waveform = np.zeros(total_samples, dtype=np.complex64)
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_samples
        end_idx = min(start_idx + chunk_samples, total_samples)
        actual_samples = end_idx - start_idx
        
        # Generate oversampled noise
        oversample_samples = actual_samples * oversample_factor
        noise_real = np.random.normal(0, np.sqrt(noise_power/2), oversample_samples)
        noise_imag = np.random.normal(0, np.sqrt(noise_power/2), oversample_samples)
        noise_oversample = (noise_real + 1j * noise_imag).astype(np.complex64)
        
        # Filter and decimate
        noise_filtered = signal.filtfilt(b, a, noise_oversample)
        noise_chunk = noise_filtered[::oversample_factor][:actual_samples].astype(np.complex64)
        
        # Store in waveform
        waveform[start_idx:end_idx] = noise_chunk
        
        # Progress
        progress = ((chunk_idx + 1) / num_chunks) * 100
        if chunk_idx % 5 == 0 or chunk_idx == num_chunks - 1:  # Show progress every 5 chunks
            print(f"   Progress: {progress:.1f}%")
    
    print(f"✓ Waveform ready: {len(waveform):,} samples, {len(waveform)/sample_rate:.1f}s")
    return waveform

class GainMonitor:
    """Monitor gain changes during transmission"""
    def __init__(self, usrp, initial_gain):
        self.usrp = usrp
        self.current_gain = initial_gain
        self.stop_flag = threading.Event()
        
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
    
    def monitor_loop(self):
        """Background monitoring loop"""
        while not self.stop_flag.wait(0.1):  # Check every 100ms
            new_gain = self.get_gain_from_file()
            if new_gain != self.current_gain:
                print(f"📶 Gain update: {self.current_gain} → {new_gain} dB")
                old_gain = self.current_gain
                self.current_gain = new_gain
                try:
                    self.usrp.set_tx_gain(new_gain, 0)
                    print(f"✓ USRP gain updated to {new_gain} dB")
                except Exception as e:
                    print(f"✗ Failed to update gain: {e}")
                    self.current_gain = old_gain  # Revert on failure
    
    def start(self):
        """Start monitoring in background thread"""
        self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.thread.start()
        print("🔄 Gain monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        self.stop_flag.set()
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)

def main():
    args = parse_args()
    
    print("=" * 70)
    print("🔊 SIMPLE 10-MINUTE NOISE GENERATOR")
    print("=" * 70)
    print(f"📻 Frequency: {args.center_freq/1e6:.1f} MHz")
    print(f"📊 Sample Rate: {args.sample_rate/1e6:.1f} MHz")
    print(f"📐 Bandwidth: 20 MHz (filtered)")
    print(f"⏱ Duration: {args.duration/60:.1f} minutes")
    print(f"📶 Initial Gain: {args.gain} dB")
    print("=" * 70)
    
    # Initialize USRP
    print("🔧 Initializing USRP...")
    usrp_addr = f"type={PLATFORM},serial={SERIAL}"
    usrp = uhd.usrp.MultiUSRP(usrp_addr)
    
    # Configure USRP
    usrp.set_tx_rate(args.sample_rate, 0)
    usrp.set_tx_freq(args.center_freq, 0) 
    usrp.set_tx_bandwidth(20e6, 0)
    usrp.set_tx_gain(args.gain, 0)
    print("✓ USRP configured")
    
    # Create control file
    create_control_file(args.gain)
    
    # Generate waveform
    waveform = generate_long_waveform(args.duration, args.sample_rate, args.noise_power)
    
    # Setup gain monitoring
    gain_monitor = GainMonitor(usrp, args.gain)
    
    print("\n" + "=" * 70)
    print("🚀 STARTING TRANSMISSION")
    print("=" * 70)
    print(f"📡 Transmitting {args.duration/60:.1f}-minute waveform...")
    print("📶 Gain can be updated during transmission")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 70)
    
    try:
        # Start gain monitoring
        gain_monitor.start()
        
        start_time = time.time()
        
        # Transmit the entire waveform
        print("📡 Transmission started...")
        usrp.send_waveform(waveform, args.duration, args.center_freq, 
                          args.sample_rate, [0], args.gain)
        
        end_time = time.time()
        actual_duration = end_time - start_time
        print(f"\n✅ Transmission completed successfully!")
        print(f"⏱ Actual duration: {actual_duration/60:.2f} minutes")
        
    except KeyboardInterrupt:
        end_time = time.time()
        actual_duration = end_time - start_time
        print(f"\n⚠ Transmission interrupted after {actual_duration/60:.2f} minutes")
    except Exception as e:
        print(f"\n❌ Transmission error: {e}")
    finally:
        # Cleanup
        gain_monitor.stop()
        try:
            os.remove(CONTROL_FILE)
            print(f"🧹 Cleaned up {CONTROL_FILE}")
        except OSError:
            pass
        
        print("🏁 Done!")

if __name__ == "__main__":
    main()
