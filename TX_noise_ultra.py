#!/usr/bin/python3
"""
Ultra-Efficient 10-Minute Noise Generator
Uses multiple small patterns (100-500ms) to minimize memory while avoiding repetition artifacts
Perfect for long transmissions with minimal memory usage
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
    parser = argparse.ArgumentParser(description="Ultra-Efficient USRP Noise Generator")
    parser.add_argument('--center-freq', type=float, default=3410.1e6, help='Center frequency (Hz)')
    parser.add_argument('--sample-rate', type=float, default=20e6, help='Sample rate (Hz)')
    parser.add_argument('--gain', type=float, default=60, help='Initial TX gain (dB)')
    parser.add_argument('--noise-power', type=float, default=1.0, help='Noise power (linear, not dB)')
    parser.add_argument('--duration', type=float, default=600, help='Duration in seconds')
    parser.add_argument('--pattern-size', type=float, default=0.5, help='Pattern size in seconds (0.1-2.0)')
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

def generate_pattern_bank(pattern_size, sample_rate, noise_power, num_patterns=8):
    """Generate a bank of different filtered noise patterns"""
    print(f"🔊 Generating {num_patterns} patterns ({pattern_size}s each)...")
    
    pattern_samples = int(pattern_size * sample_rate)
    
    # Filter design (once for all patterns)
    oversample_factor = 2
    oversample_rate = sample_rate * oversample_factor
    nyquist = oversample_rate / 2
    cutoff = 10e6  # 10 MHz for 20 MHz bandwidth
    normalized_cutoff = cutoff / nyquist
    b, a = signal.butter(6, normalized_cutoff, btype='low', analog=False)
    
    patterns = []
    for i in range(num_patterns):
        # Generate oversampled noise
        oversample_samples = pattern_samples * oversample_factor
        noise_real = np.random.normal(0, np.sqrt(noise_power/2), oversample_samples)
        noise_imag = np.random.normal(0, np.sqrt(noise_power/2), oversample_samples)
        noise_oversample = (noise_real + 1j * noise_imag).astype(np.complex64)
        
        # Filter and decimate
        noise_filtered = signal.filtfilt(b, a, noise_oversample)
        pattern = noise_filtered[::oversample_factor][:pattern_samples].astype(np.complex64)
        patterns.append(pattern)
    
    total_memory_mb = len(patterns) * len(patterns[0]) * 8 / 1e6
    print(f"✅ Generated {num_patterns} patterns")
    print(f"💾 Total pattern memory: {total_memory_mb:.1f} MB")
    
    return patterns

class UltraEfficientTransmitter:
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
        
        # Generate pattern bank
        self.patterns = generate_pattern_bank(args.pattern_size, args.sample_rate, args.noise_power)
        self.pattern_index = 0
        
        # Calculate transmission parameters
        self.pattern_duration = args.pattern_size
        self.total_patterns_needed = int(np.ceil(args.duration / self.pattern_duration))
        
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
            print(f"📶 Gain: {self.current_gain} → {new_gain} dB")
            old_gain = self.current_gain
            self.current_gain = new_gain
            try:
                self.usrp.set_tx_gain(new_gain, 0)
                return True
            except Exception as e:
                print(f"❌ Gain update failed: {e}")
                self.current_gain = old_gain
                return False
        return False
    
    def get_next_pattern(self):
        """Get next pattern from the bank (cycling through them)"""
        pattern = self.patterns[self.pattern_index]
        self.pattern_index = (self.pattern_index + 1) % len(self.patterns)
        return pattern
    
    def transmit_patterns(self):
        """Transmit using pattern cycling"""
        print("\n" + "=" * 70)
        print("🚀 STARTING ULTRA-EFFICIENT TRANSMISSION")
        print("=" * 70)
        print(f"📡 Pattern size: {self.pattern_duration}s")
        print(f"🔄 Total patterns: {self.total_patterns_needed}")
        print(f"🎯 Pattern variations: {len(self.patterns)}")
        print(f"📶 Gain checks every {self.pattern_duration}s")
        print("=" * 70)
        
        start_time = time.time()
        gain_check_interval = max(1.0, self.pattern_duration * 10)  # Check gain every 10 patterns or 1s minimum
        last_gain_check = start_time
        
        try:
            for pattern_num in range(self.total_patterns_needed):
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Check if we've exceeded the duration
                if elapsed >= self.args.duration:
                    break
                
                # Check for gain updates periodically
                if current_time - last_gain_check >= gain_check_interval:
                    self.update_gain_if_needed()
                    last_gain_check = current_time
                
                # Calculate remaining time and adjust pattern duration if needed
                remaining = self.args.duration - elapsed
                current_pattern_duration = min(self.pattern_duration, remaining)
                
                # Get next pattern and transmit
                pattern = self.get_next_pattern()
                self.usrp.send_waveform(pattern, current_pattern_duration,
                                      self.args.center_freq, self.args.sample_rate,
                                      [0], self.current_gain)
                
                # Show progress every 30 seconds
                if pattern_num % max(1, int(30 / self.pattern_duration)) == 0:
                    progress = (elapsed / self.args.duration) * 100
                    print(f"📊 {progress:.1f}% ({elapsed/60:.1f}min) - Pattern {pattern_num + 1}")
                
            end_time = time.time()
            actual_duration = end_time - start_time
            print(f"\n✅ Transmission completed!")
            print(f"⏱ Duration: {actual_duration/60:.2f} minutes")
            print(f"🔄 Patterns sent: {pattern_num + 1}")
            
        except KeyboardInterrupt:
            end_time = time.time()
            actual_duration = end_time - start_time
            print(f"\n⚠ Interrupted after {actual_duration/60:.2f} minutes")
            print(f"🔄 Patterns sent: {pattern_num}")

def main():
    args = parse_args()
    
    # Validate pattern size
    if args.pattern_size < 0.1 or args.pattern_size > 2.0:
        print("⚠ Warning: Pattern size should be between 0.1 and 2.0 seconds")
        args.pattern_size = max(0.1, min(2.0, args.pattern_size))
    
    # Calculate memory efficiency
    pattern_samples = int(args.pattern_size * args.sample_rate)
    num_patterns = 8
    total_pattern_memory_mb = num_patterns * pattern_samples * 8 / 1e6
    full_duration_gb = (args.duration * args.sample_rate * 8) / 1e9
    memory_savings = (1 - (total_pattern_memory_mb / 1000) / full_duration_gb) * 100
    
    print("=" * 70)
    print("🚀 ULTRA-EFFICIENT NOISE GENERATOR")
    print("=" * 70)
    print(f"📻 Frequency: {args.center_freq/1e6:.1f} MHz")
    print(f"📊 Sample Rate: {args.sample_rate/1e6:.1f} MHz")
    print(f"📐 Bandwidth: 20 MHz (filtered)")
    print(f"⏱ Duration: {args.duration/60:.1f} minutes")
    print(f"🔄 Pattern Size: {args.pattern_size}s")
    print(f"📶 Initial Gain: {args.gain} dB")
    print("─" * 70)
    print(f"💾 Pattern Memory: {total_pattern_memory_mb:.1f} MB")
    print(f"📊 If stored fully: {full_duration_gb:.1f} GB")
    print(f"✅ Memory Savings: {memory_savings:.1f}%")
    print("=" * 70)
    
    # Create control file
    create_control_file(args.gain)
    
    # Create and run transmitter
    transmitter = UltraEfficientTransmitter(args)
    
    try:
        transmitter.transmit_patterns()
        print(f"\n🎉 SUCCESS!")
        
    except KeyboardInterrupt:
        print(f"\n⚠ Interrupted by user")
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
