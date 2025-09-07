#!/usr/bin/python3
"""
Memory-Efficient 10-Minute Noise Generator
Uses small repeating patterns to minimize memory while maintaining consistency
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
    parser = argparse.ArgumentParser(description="Memory-Efficient USRP Noise Generator")
    parser.add_argument('--center-freq', type=float, default=3410.1e6, help='Center frequency (Hz)')
    parser.add_argument('--sample-rate', type=float, default=20e6, help='Sample rate (Hz)')
    parser.add_argument('--gain', type=float, default=60, help='Initial TX gain (dB)')
    parser.add_argument('--noise-power', type=float, default=1.0, help='Noise power (linear, not dB)')
    parser.add_argument('--duration', type=float, default=600, help='Duration in seconds')
    parser.add_argument('--pattern-duration', type=float, default=2.0, help='Base pattern duration (seconds)')
    return parser.parse_args()

def create_control_file(initial_gain):
    """Create initial control file with proper permissions"""
    try:
        with open(CONTROL_FILE, 'w') as f:
            f.write(str(initial_gain))
        # Set permissions so both user and root can write
        os.chmod(CONTROL_FILE, 0o666)
        print(f"📝 Control file: {CONTROL_FILE}")
        print(f"   Update gain: echo 'NEW_GAIN' > {CONTROL_FILE}")
        print(f"   Root access: sudo bash -c \"echo 'GAIN' > {CONTROL_FILE}\"")
    except IOError:
        print(f"⚠ Warning: Could not create {CONTROL_FILE}")

def generate_filtered_pattern(pattern_duration, sample_rate, noise_power):
    """Generate a single filtered noise pattern"""
    pattern_samples = int(pattern_duration * sample_rate)
    
    # Use oversampling for better filtering
    oversample_factor = 2
    oversample_rate = sample_rate * oversample_factor
    oversample_samples = pattern_samples * oversample_factor
    
    # Generate oversampled white noise
    noise_real = np.random.normal(0, np.sqrt(noise_power/2), oversample_samples)
    noise_imag = np.random.normal(0, np.sqrt(noise_power/2), oversample_samples)
    noise_oversample = (noise_real + 1j * noise_imag).astype(np.complex64)
    
    # Design and apply 20 MHz bandwidth filter
    nyquist = oversample_rate / 2
    cutoff = 10e6  # 10 MHz
    normalized_cutoff = cutoff / nyquist
    b, a = signal.butter(6, normalized_cutoff, btype='low', analog=False)
    
    noise_filtered = signal.filtfilt(b, a, noise_oversample)
    
    # Decimate back to original sample rate
    pattern = noise_filtered[::oversample_factor][:pattern_samples].astype(np.complex64)
    
    return pattern

class EfficientNoiseTransmitter:
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
        
        # Generate base pattern
        print(f"🔊 Generating {args.pattern_duration}s base pattern...")
        self.base_pattern = generate_filtered_pattern(args.pattern_duration, args.sample_rate, args.noise_power)
        
        pattern_memory_mb = len(self.base_pattern) * 8 / 1e6
        print(f"💾 Pattern size: {len(self.base_pattern):,} samples ({pattern_memory_mb:.1f} MB)")
        
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
            old_gain = self.current_gain
            self.current_gain = new_gain
            try:
                self.usrp.set_tx_gain(new_gain, 0)
                print(f"✅ Gain updated to {new_gain} dB")
                return True
            except Exception as e:
                print(f"❌ Gain update failed: {e}")
                self.current_gain = old_gain
                return False
        return False
    
    def transmit_with_repetition(self):
        """Transmit by repeating the base pattern"""
        print("\n" + "=" * 70)
        print("🚀 STARTING MEMORY-EFFICIENT TRANSMISSION")
        print("=" * 70)
        
        pattern_duration = len(self.base_pattern) / self.args.sample_rate
        total_repetitions = int(np.ceil(self.args.duration / pattern_duration))
        
        print(f"📡 Pattern duration: {pattern_duration}s")
        print(f"🔄 Total repetitions needed: {total_repetitions}")
        print(f"📶 Gain updates every {pattern_duration}s")
        print("🛑 Press Ctrl+C to stop")
        print("=" * 70)
        
        start_time = time.time()
        
        try:
            for rep in range(total_repetitions):
                # Check for gain updates before each repetition
                self.update_gain_if_needed()
                
                # Calculate remaining time
                elapsed = time.time() - start_time
                remaining = self.args.duration - elapsed
                
                if remaining <= 0:
                    break
                
                # Use shorter duration for the last repetition if needed
                current_duration = min(pattern_duration, remaining)
                
                # Show progress occasionally
                if rep % 10 == 0 or rep == total_repetitions - 1:
                    progress = (elapsed / self.args.duration) * 100
                    print(f"📊 Progress: {progress:.1f}% ({elapsed/60:.1f}min elapsed)")
                
                # Transmit the pattern
                self.usrp.send_waveform(self.base_pattern, current_duration, 
                                      self.args.center_freq, self.args.sample_rate, 
                                      [0], self.current_gain)
                
            end_time = time.time()
            actual_duration = end_time - start_time
            print(f"\n✅ Transmission completed!")
            print(f"⏱ Actual duration: {actual_duration/60:.2f} minutes")
            print(f"🔄 Pattern repetitions: {rep + 1}")
            
        except KeyboardInterrupt:
            end_time = time.time()
            actual_duration = end_time - start_time
            print(f"\n⚠ Transmission interrupted after {actual_duration/60:.2f} minutes")
            print(f"🔄 Completed repetitions: {rep}")

def main():
    args = parse_args()
    
    # Calculate memory usage
    pattern_samples = int(args.pattern_duration * args.sample_rate)
    pattern_memory_mb = pattern_samples * 8 / 1e6
    total_data_gb = (args.duration * args.sample_rate * 8) / 1e9
    
    print("=" * 70)
    print("🔊 MEMORY-EFFICIENT NOISE GENERATOR")
    print("=" * 70)
    print(f"📻 Frequency: {args.center_freq/1e6:.1f} MHz")
    print(f"📊 Sample Rate: {args.sample_rate/1e6:.1f} MHz")
    print(f"📐 Bandwidth: 20 MHz (filtered)")
    print(f"⏱ Total Duration: {args.duration/60:.1f} minutes")
    print(f"🔄 Pattern Duration: {args.pattern_duration}s")
    print(f"📶 Initial Gain: {args.gain} dB")
    print("─" * 70)
    print(f"💾 Pattern Memory: {pattern_memory_mb:.1f} MB")
    print(f"📊 Total Data (if stored): {total_data_gb:.1f} GB")
    print(f"✅ Actual Memory Used: {pattern_memory_mb:.1f} MB (99.9% savings!)")
    print("=" * 70)
    
    # Create control file
    create_control_file(args.gain)
    
    # Create and run transmitter
    transmitter = EfficientNoiseTransmitter(args)
    
    try:
        transmitter.transmit_with_repetition()
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
