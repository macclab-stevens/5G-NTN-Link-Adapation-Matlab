#!/usr/bin/python3
"""
Optimized 10-Minute Noise Generator with Memory Management
Uses chunked generation and transmission to avoid memory issues
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
    parser = argparse.ArgumentParser(description="Optimized USRP Noise Generator")
    parser.add_argument('--center-freq', type=float, default=3410.1e6, help='Center frequency (Hz)')
    parser.add_argument('--sample-rate', type=float, default=20e6, help='Sample rate (Hz)')
    parser.add_argument('--gain', type=float, default=60, help='Initial TX gain (dB)')
    parser.add_argument('--noise-power', type=float, default=1.0, help='Noise power (linear, not dB)')
    parser.add_argument('--duration', type=float, default=600, help='Duration in seconds')
    parser.add_argument('--chunk-duration', type=float, default=10.0, help='Chunk duration in seconds')
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

def generate_filtered_chunk(chunk_duration, sample_rate, noise_power):
    """Generate a single filtered noise chunk"""
    chunk_samples = int(chunk_duration * sample_rate)
    
    # Use oversampling and filtering
    oversample_factor = 2
    oversample_rate = sample_rate * oversample_factor
    oversample_samples = chunk_samples * oversample_factor
    
    # Generate oversampled noise
    noise_real = np.random.normal(0, np.sqrt(noise_power/2), oversample_samples)
    noise_imag = np.random.normal(0, np.sqrt(noise_power/2), oversample_samples)
    noise_oversample = (noise_real + 1j * noise_imag).astype(np.complex64)
    
    # Design and apply filter for 20 MHz bandwidth  
    nyquist = oversample_rate / 2
    cutoff = 10e6  # 10 MHz
    normalized_cutoff = cutoff / nyquist
    b, a = signal.butter(6, normalized_cutoff, btype='low', analog=False)
    
    noise_filtered = signal.filtfilt(b, a, noise_oversample)
    
    # Decimate back to original sample rate
    noise_chunk = noise_filtered[::oversample_factor][:chunk_samples].astype(np.complex64)
    
    return noise_chunk

class GainMonitor:
    """Monitor and apply gain changes"""
    def __init__(self, usrp, initial_gain):
        self.usrp = usrp
        self.current_gain = initial_gain
        self.stop_flag = threading.Event()
        self.gain_changed = False
        
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
    
    def check_and_update_gain(self):
        """Check for gain updates and apply if needed"""
        new_gain = self.get_gain_from_file()
        if new_gain != self.current_gain:
            print(f"📶 Gain update: {self.current_gain} → {new_gain} dB")
            old_gain = self.current_gain
            self.current_gain = new_gain
            try:
                self.usrp.set_tx_gain(new_gain, 0)
                print(f"✅ USRP gain updated to {new_gain} dB")
                self.gain_changed = True
                return True
            except Exception as e:
                print(f"❌ Failed to update gain: {e}")
                self.current_gain = old_gain
                return False
        return False

def main():
    args = parse_args()
    
    print("=" * 70)
    print("🔊 OPTIMIZED NOISE GENERATOR")
    print("=" * 70)
    print(f"📻 Frequency: {args.center_freq/1e6:.1f} MHz")
    print(f"📊 Sample Rate: {args.sample_rate/1e6:.1f} MHz")
    print(f"📐 Bandwidth: 20 MHz (filtered)")
    print(f"⏱ Duration: {args.duration/60:.1f} minutes")
    print(f"📦 Chunk Duration: {args.chunk_duration}s")
    print(f"📶 Initial Gain: {args.gain} dB")
    
    chunk_samples = int(args.chunk_duration * args.sample_rate)
    chunk_memory_mb = chunk_samples * 8 / 1e6  # complex64 = 8 bytes
    print(f"💾 Memory per chunk: {chunk_memory_mb:.1f} MB")
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
    print("✅ USRP configured")
    
    # Create control file and gain monitor
    create_control_file(args.gain)
    gain_monitor = GainMonitor(usrp, args.gain)
    
    print("\n" + "=" * 70)
    print("🚀 STARTING CHUNKED TRANSMISSION")
    print("=" * 70)
    print(f"📡 Transmitting {args.duration/60:.1f} minutes in {args.chunk_duration}s chunks")
    print("📶 Gain updates between chunks")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 70)
    
    start_time = time.time()
    chunk_count = 0
    total_chunks = int(np.ceil(args.duration / args.chunk_duration))
    
    try:
        elapsed_time = 0
        
        while elapsed_time < args.duration:
            chunk_start_time = time.time()
            
            # Calculate remaining time and chunk duration
            remaining_time = args.duration - elapsed_time
            current_chunk_duration = min(args.chunk_duration, remaining_time)
            
            # Check for gain updates
            gain_monitor.check_and_update_gain()
            
            # Generate chunk
            print(f"🔊 Generating chunk {chunk_count + 1}/{total_chunks} ({current_chunk_duration}s)...")
            chunk = generate_filtered_chunk(current_chunk_duration, args.sample_rate, args.noise_power)
            
            # Transmit chunk
            print(f"📡 Transmitting chunk {chunk_count + 1}...")
            usrp.send_waveform(chunk, current_chunk_duration, args.center_freq,
                             args.sample_rate, [0], gain_monitor.current_gain)
            
            chunk_count += 1
            elapsed_time = time.time() - start_time
            
            print(f"✅ Chunk {chunk_count} completed. Elapsed: {elapsed_time/60:.2f}min")
            
        end_time = time.time()
        total_duration = end_time - start_time
        print(f"\n🎉 TRANSMISSION COMPLETED!")
        print(f"⏱ Total time: {total_duration/60:.2f} minutes")
        print(f"📦 Chunks transmitted: {chunk_count}")
        
    except KeyboardInterrupt:
        end_time = time.time()
        total_duration = end_time - start_time
        print(f"\n⚠ Transmission interrupted after {total_duration/60:.2f} minutes")
        print(f"📦 Chunks completed: {chunk_count}")
    except Exception as e:
        print(f"\n❌ Error during transmission: {e}")
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
