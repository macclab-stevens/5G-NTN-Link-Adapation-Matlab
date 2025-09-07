#!/usr/bin/python3
"""
Improved USRP Noise Generator with Consistent Transmission
Addresses SNR measurement variations by:
1. Using continuous streaming instead of repeated short blocks
2. Pre-generating long waveforms to minimize repetition artifacts
3. Using threaded gain control to avoid transmission interruptions
4. Implementing proper buffering to prevent gaps
"""

import uhd
import numpy as np
import argparse
import time
import os
import threading
import queue
from scipy import signal

PLATFORM = "b200"
SERIAL = "31577EF"
CONTROL_FILE = "/tmp/usrp_gain.txt"

class NoiseTransmitter:
    def __init__(self, args):
        self.center_freq = args.center_freq
        self.sample_rate = args.sample_rate
        self.initial_gain = args.gain
        self.noise_power = args.noise_power
        self.duration = args.duration
        self.current_gain = self.initial_gain
        
        # Threading control
        self.gain_update_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # Initialize USRP
        usrp_addr = f"type={PLATFORM},serial={SERIAL}"
        self.usrp = uhd.usrp.MultiUSRP(usrp_addr)
        
        # Configure USRP
        print(f"Configuring USRP: center_freq={self.center_freq}, sample_rate={self.sample_rate}")
        self.usrp.set_tx_rate(self.sample_rate, 0)
        self.usrp.set_tx_freq(self.center_freq, 0)
        self.usrp.set_tx_bandwidth(20e6, 0)
        self.usrp.set_tx_gain(self.current_gain, 0)
        
        # Generate long noise waveform to minimize repetition artifacts
        self.generate_long_waveform()
        
    def generate_long_waveform(self):
        """Generate a long filtered noise waveform to minimize repetition artifacts"""
        print("Generating long filtered noise waveform...")
        
        # Generate enough samples for the full duration or at least 10 seconds
        waveform_duration = max(self.duration, 10.0) if self.duration > 0 else 10.0
        total_samples = int(waveform_duration * self.sample_rate)
        
        # Generate in chunks to manage memory efficiently
        chunk_duration = 1.0  # 1 second chunks
        chunk_samples = int(chunk_duration * self.sample_rate)
        
        print(f"Generating {waveform_duration:.1f}s of noise ({total_samples:,} samples)")
        
        # For consistent filtering, generate a longer sequence and filter the whole thing
        oversample_factor = 2
        oversample_rate = self.sample_rate * oversample_factor
        total_oversample_samples = total_samples * oversample_factor
        
        # Design filter once
        nyquist = oversample_rate / 2
        cutoff = 10e6  # 10 MHz (half of 20 MHz BW)
        normalized_cutoff = cutoff / nyquist
        b, a = signal.butter(6, normalized_cutoff, btype='low', analog=False)
        
        # Generate and filter in chunks to manage memory
        self.noise_waveform = np.zeros(total_samples, dtype=np.complex64)
        
        for chunk_start in range(0, total_samples, chunk_samples):
            chunk_end = min(chunk_start + chunk_samples, total_samples)
            chunk_size = chunk_end - chunk_start
            oversample_chunk_size = chunk_size * oversample_factor
            
            # Generate oversampled white noise for this chunk
            noise_real = np.random.normal(0, np.sqrt(self.noise_power/2), oversample_chunk_size)
            noise_imag = np.random.normal(0, np.sqrt(self.noise_power/2), oversample_chunk_size)
            noise_oversample = (noise_real + 1j * noise_imag).astype(np.complex64)
            
            # Filter and decimate
            noise_filtered = signal.filtfilt(b, a, noise_oversample)
            noise_chunk = noise_filtered[::oversample_factor].astype(np.complex64)
            
            # Store in the main waveform
            self.noise_waveform[chunk_start:chunk_end] = noise_chunk[:chunk_size]
            
            if chunk_start % (10 * chunk_samples) == 0:  # Progress every 10 chunks
                progress = (chunk_start / total_samples) * 100
                print(f"  Progress: {progress:.1f}%")
        
        print(f"Generated waveform: {len(self.noise_waveform):,} samples, {len(self.noise_waveform)/self.sample_rate:.1f}s duration")
        
    def gain_monitor_thread(self):
        """Background thread to monitor gain changes"""
        last_check_time = time.time()
        check_interval = 0.1  # Check every 100ms
        
        while not self.stop_event.is_set():
            current_time = time.time()
            
            if current_time - last_check_time >= check_interval:
                new_gain = self.get_gain_from_file()
                if new_gain != self.current_gain:
                    print(f"Gain change detected: {self.current_gain} -> {new_gain} dB")
                    self.gain_update_queue.put(new_gain)
                last_check_time = current_time
            
            time.sleep(0.01)  # Small sleep to prevent busy waiting
    
    def get_gain_from_file(self):
        """Read gain from control file, return current gain if file doesn't exist or is invalid"""
        try:
            if os.path.exists(CONTROL_FILE):
                with open(CONTROL_FILE, 'r') as f:
                    content = f.read().strip()
                    if content:
                        gain = float(content)
                        return gain
        except (ValueError, IOError):
            pass
        return self.current_gain
    
    def write_initial_control_file(self):
        """Create initial control file with default gain"""
        try:
            with open(CONTROL_FILE, 'w') as f:
                f.write(str(self.current_gain))
            print(f"Control file created: {CONTROL_FILE}")
            print(f"Change gain by writing new value to this file (e.g., echo '50' > {CONTROL_FILE})")
        except IOError:
            print(f"Warning: Could not create control file {CONTROL_FILE}")
    
    def transmit_continuous(self):
        """Transmit using continuous streaming for best consistency"""
        print("Starting continuous streaming transmission...")
        
        # Create TX streamer
        stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
        tx_stream = self.usrp.get_tx_stream(stream_args)
        
        # Start streaming
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_continuous)
        stream_cmd.stream_now = True
        tx_stream.issue_stream_cmd(stream_cmd)
        
        # Transmission parameters
        samples_per_buffer = int(0.1 * self.sample_rate)  # 100ms buffers
        waveform_samples = len(self.noise_waveform)
        waveform_position = 0
        
        print(f"Transmission started with {samples_per_buffer:,} samples per buffer")
        print("Monitoring for gain changes...")
        
        try:
            while True:
                # Check for gain updates (non-blocking)
                try:
                    new_gain = self.gain_update_queue.get_nowait()
                    self.current_gain = new_gain
                    self.usrp.set_tx_gain(self.current_gain, 0)
                    print(f"USRP TX gain updated to: {self.current_gain} dB")
                except queue.Empty:
                    pass
                
                # Prepare next buffer from the long waveform
                if waveform_position + samples_per_buffer <= waveform_samples:
                    # Normal case: take samples from current position
                    buffer = self.noise_waveform[waveform_position:waveform_position + samples_per_buffer]
                    waveform_position += samples_per_buffer
                else:
                    # Wrap around case: take remaining samples + samples from beginning
                    remaining = waveform_samples - waveform_position
                    needed_from_start = samples_per_buffer - remaining
                    
                    buffer = np.concatenate([
                        self.noise_waveform[waveform_position:],
                        self.noise_waveform[:needed_from_start]
                    ])
                    waveform_position = needed_from_start
                
                # Send buffer
                metadata = uhd.types.TXMetadata()
                tx_stream.send(buffer, metadata)
                
                # Check if we should stop (for finite duration)
                if self.duration > 0:
                    current_time = time.time()
                    if current_time - self.start_time >= self.duration:
                        break
                        
        except KeyboardInterrupt:
            print("\nStopping transmission...")
        finally:
            # Stop streaming
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_continuous)
            tx_stream.issue_stream_cmd(stream_cmd)
            time.sleep(0.1)  # Allow stream to stop
    
    def transmit_waveform_method(self):
        """Alternative: Use send_waveform with a very long waveform"""
        print("Starting transmission using send_waveform method...")
        
        # For very long transmissions, we'll use the pre-generated waveform
        # and let UHD handle the repetition
        try:
            if self.duration > 0:
                print(f"Transmitting for {self.duration} seconds...")
                self.usrp.send_waveform(self.noise_waveform, self.duration, 
                                      self.center_freq, self.sample_rate, [0], self.current_gain)
            else:
                print("Transmitting indefinitely...")
                # For infinite transmission, repeatedly send the long waveform
                while True:
                    # Check for gain updates
                    new_gain = self.get_gain_from_file()
                    if new_gain != self.current_gain:
                        print(f"Gain change detected: {self.current_gain} -> {new_gain} dB")
                        self.current_gain = new_gain
                        self.usrp.set_tx_gain(self.current_gain, 0)
                        print(f"USRP TX gain updated to: {self.current_gain} dB")
                    
                    # Send the long waveform (will take 10+ seconds)
                    waveform_duration = len(self.noise_waveform) / self.sample_rate
                    self.usrp.send_waveform(self.noise_waveform, waveform_duration,
                                          self.center_freq, self.sample_rate, [0], self.current_gain)
                    
        except KeyboardInterrupt:
            print("\nStopping transmission...")
    
    def run(self, method='continuous'):
        """Run the noise transmitter"""
        self.write_initial_control_file()
        self.start_time = time.time()
        
        if method == 'continuous':
            # Start gain monitoring thread
            gain_thread = threading.Thread(target=self.gain_monitor_thread, daemon=True)
            gain_thread.start()
            
            try:
                self.transmit_continuous()
            finally:
                self.stop_event.set()
                gain_thread.join(timeout=1.0)
        else:
            self.transmit_waveform_method()
        
        # Cleanup
        try:
            os.remove(CONTROL_FILE)
            print(f"Removed control file: {CONTROL_FILE}")
        except OSError:
            pass

def parse_args():
    parser = argparse.ArgumentParser(description="Improved USRP Noise Generator with Consistent Transmission")
    parser.add_argument('--center-freq', type=float, default=3410.1e6, help='Center frequency (Hz)')
    parser.add_argument('--sample-rate', type=float, default=20e6, help='Sample rate (Hz)')
    parser.add_argument('--gain', type=float, default=60, help='Initial TX gain (dB)')
    parser.add_argument('--duration', type=float, default=600, help='Transmission duration (seconds), 0 for infinite')
    parser.add_argument('--noise-power', type=float, default=1.0, help='Noise power (linear, not dB)')
    parser.add_argument('--method', choices=['continuous', 'waveform'], default='continuous',
                        help='Transmission method: continuous streaming or send_waveform')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 60)
    print("Improved USRP Noise Generator")
    print("=" * 60)
    print(f"Center frequency: {args.center_freq/1e6:.1f} MHz")
    print(f"Sample rate: {args.sample_rate/1e6:.1f} MHz")
    print(f"Initial gain: {args.gain} dB")
    print(f"Duration: {'Infinite' if args.duration == 0 else f'{args.duration} seconds'}")
    print(f"Method: {args.method}")
    print("=" * 60)
    
    transmitter = NoiseTransmitter(args)
    transmitter.run(method=args.method)

if __name__ == "__main__":
    main()
