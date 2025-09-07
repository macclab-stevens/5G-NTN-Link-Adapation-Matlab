#!/usr/bin/python3
"""
Streaming Noise Generator - Zero-Gap Transmission
Uses UHD streaming API for truly continuous transmission with real-time gain updates
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

def parse_args():
    parser = argparse.ArgumentParser(description="Streaming USRP Noise Generator - Zero Gaps")
    parser.add_argument('--center-freq', type=float, default=3410.1e6, help='Center frequency (Hz)')
    parser.add_argument('--sample-rate', type=float, default=20e6, help='Sample rate (Hz)')
    parser.add_argument('--gain', type=float, default=60, help='Initial TX gain (dB)')
    parser.add_argument('--noise-power', type=float, default=1.0, help='Noise power (linear, not dB)')
    parser.add_argument('--duration', type=float, default=600, help='Duration in seconds (0 = infinite)')
    parser.add_argument('--buffer-size', type=float, default=0.1, help='Buffer size in seconds')
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

def generate_noise_buffer(buffer_samples, noise_power):
    """Generate a single buffer of filtered noise"""
    # For efficiency, we'll generate white noise and apply a simple scaling
    # rather than filtering every buffer (which would be computationally expensive)
    noise_real = np.random.normal(0, np.sqrt(noise_power/2), buffer_samples)
    noise_imag = np.random.normal(0, np.sqrt(noise_power/2), buffer_samples)
    return (noise_real + 1j * noise_imag).astype(np.complex64)

def generate_base_noise_waveform(duration, sample_rate, noise_power):
    """Generate a base filtered noise waveform that will be cycled through"""
    print(f"🔊 Generating base noise waveform ({duration:.1f}s)...")
    
    total_samples = int(duration * sample_rate)
    
    # Use oversampling and filtering for the base waveform
    oversample_factor = 2
    oversample_rate = sample_rate * oversample_factor
    oversample_samples = total_samples * oversample_factor
    
    # Generate oversampled white noise
    noise_real = np.random.normal(0, np.sqrt(noise_power/2), oversample_samples)
    noise_imag = np.random.normal(0, np.sqrt(noise_power/2), oversample_samples)
    noise_oversample = (noise_real + 1j * noise_imag).astype(np.complex64)
    
    # Design and apply filter for 20 MHz bandwidth
    nyquist = oversample_rate / 2
    cutoff = 10e6  # 10 MHz
    normalized_cutoff = cutoff / nyquist
    b, a = signal.butter(6, normalized_cutoff, btype='low', analog=False)
    
    print("   Applying 20 MHz bandwidth filter...")
    noise_filtered = signal.filtfilt(b, a, noise_oversample)
    
    # Decimate back to original sample rate
    noise_waveform = noise_filtered[::oversample_factor].astype(np.complex64)
    
    print(f"✓ Base waveform ready: {len(noise_waveform):,} samples")
    return noise_waveform

class StreamingNoiseTransmitter:
    def __init__(self, args):
        self.args = args
        self.current_gain = args.gain
        self.stop_event = threading.Event()
        self.gain_queue = queue.Queue()
        
        # Initialize USRP
        print("🔧 Initializing USRP...")
        usrp_addr = f"type={PLATFORM},serial={SERIAL}"
        self.usrp = uhd.usrp.MultiUSRP(usrp_addr)
        
        # Configure USRP
        self.usrp.set_tx_rate(args.sample_rate, 0)
        self.usrp.set_tx_freq(args.center_freq, 0)
        self.usrp.set_tx_bandwidth(20e6, 0)
        self.usrp.set_tx_gain(args.gain, 0)
        print("✓ USRP configured")
        
        # Generate base noise waveform (5 seconds worth)
        base_duration = 5.0
        self.base_waveform = generate_base_noise_waveform(base_duration, args.sample_rate, args.noise_power)
        self.waveform_position = 0
        
        # Buffer parameters
        self.buffer_samples = int(args.buffer_size * args.sample_rate)
        print(f"📊 Buffer size: {self.buffer_samples:,} samples ({args.buffer_size*1000:.1f}ms)")
        
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
    
    def gain_monitor_thread(self):
        """Background thread to monitor gain changes"""
        while not self.stop_event.wait(0.05):  # Check every 50ms
            new_gain = self.get_gain_from_file()
            if new_gain != self.current_gain:
                self.gain_queue.put(new_gain)
    
    def get_next_buffer(self):
        """Get next buffer from the base waveform (cycling through it)"""
        waveform_samples = len(self.base_waveform)
        
        if self.waveform_position + self.buffer_samples <= waveform_samples:
            # Normal case: get samples from current position
            buffer = self.base_waveform[self.waveform_position:self.waveform_position + self.buffer_samples]
            self.waveform_position += self.buffer_samples
        else:
            # Wrap-around case: get remaining + start from beginning
            remaining = waveform_samples - self.waveform_position
            needed_from_start = self.buffer_samples - remaining
            
            buffer = np.concatenate([
                self.base_waveform[self.waveform_position:],
                self.base_waveform[:needed_from_start]
            ])
            self.waveform_position = needed_from_start
        
        return buffer
    
    def transmit(self):
        """Main transmission loop using streaming"""
        print("\n" + "=" * 70)
        print("🚀 STARTING STREAMING TRANSMISSION")
        print("=" * 70)
        
        # Create TX streamer
        stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
        tx_stream = self.usrp.get_tx_stream(stream_args)
        
        # Start gain monitoring
        gain_thread = threading.Thread(target=self.gain_monitor_thread, daemon=True)
        gain_thread.start()
        
        # Start streaming
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = True
        tx_stream.issue_stream_cmd(stream_cmd)
        
        print("📡 Stream started - continuous transmission active")
        print("🔄 Cycling through base waveform to minimize repetition artifacts")
        print("⚡ Real-time gain updates enabled")
        
        start_time = time.time()
        buffers_sent = 0
        last_status_time = start_time
        
        try:
            while True:
                # Check for gain updates (non-blocking)
                try:
                    new_gain = self.gain_queue.get_nowait()
                    print(f"📶 Gain update: {self.current_gain} → {new_gain} dB")
                    self.current_gain = new_gain
                    self.usrp.set_tx_gain(new_gain, 0)
                    print("✓ Gain applied seamlessly during transmission")
                except queue.Empty:
                    pass
                
                # Get next buffer and transmit
                buffer = self.get_next_buffer()
                metadata = uhd.types.TXMetadata()
                tx_stream.send(buffer, metadata)
                
                buffers_sent += 1
                
                # Status update every 10 seconds
                current_time = time.time()
                if current_time - last_status_time >= 10.0:
                    elapsed = current_time - start_time
                    print(f"📊 Status: {elapsed/60:.1f}m elapsed, {buffers_sent:,} buffers sent")
                    last_status_time = current_time
                
                # Check duration limit
                if self.args.duration > 0 and current_time - start_time >= self.args.duration:
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹ Transmission interrupted by user")
        finally:
            # Stop streaming
            print("🛑 Stopping stream...")
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
            tx_stream.issue_stream_cmd(stream_cmd)
            time.sleep(0.1)  # Allow stream to stop cleanly
            
            self.stop_event.set()
            gain_thread.join(timeout=1.0)
            
            end_time = time.time()
            total_duration = end_time - start_time
            print(f"✓ Total transmission time: {total_duration/60:.2f} minutes")
            print(f"✓ Buffers transmitted: {buffers_sent:,}")

def main():
    args = parse_args()
    
    print("=" * 70)
    print("🔊 STREAMING NOISE GENERATOR - ZERO GAPS")
    print("=" * 70)
    print(f"📻 Frequency: {args.center_freq/1e6:.1f} MHz")
    print(f"📊 Sample Rate: {args.sample_rate/1e6:.1f} MHz")
    print(f"📐 Bandwidth: 20 MHz (filtered)")
    print(f"⏱ Duration: {'Infinite' if args.duration == 0 else f'{args.duration/60:.1f} minutes'}")
    print(f"📶 Initial Gain: {args.gain} dB")
    print(f"🔧 Buffer Size: {args.buffer_size*1000:.1f}ms")
    print("=" * 70)
    
    # Create control file
    create_control_file(args.gain)
    
    # Create and run transmitter
    transmitter = StreamingNoiseTransmitter(args)
    
    try:
        transmitter.transmit()
    finally:
        # Cleanup
        try:
            os.remove(CONTROL_FILE)
            print(f"🧹 Cleaned up {CONTROL_FILE}")
        except OSError:
            pass
        
        print("🏁 Transmission ended")

if __name__ == "__main__":
    main()
