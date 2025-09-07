#!/usr/bin/python3
"""
Test UHD API attributes to find correct streaming commands
"""

import uhd

print("UHD Version:", uhd.get_version_string())
print("\nChecking StreamMode attributes:")
try:
    print("Available StreamMode attributes:")
    for attr in dir(uhd.types.StreamMode):
        if not attr.startswith('_'):
            print(f"  - {attr}")
except Exception as e:
    print(f"Error accessing StreamMode: {e}")

print("\nChecking StreamCMD:")
try:
    # Try creating a StreamCMD to see what works
    print("Testing StreamCMD creation...")
    
    # Try different variations
    test_modes = ['start_cont', 'start_continuous', 'START_CONTINUOUS', 'START_CONT']
    for mode in test_modes:
        try:
            if hasattr(uhd.types.StreamMode, mode):
                stream_mode = getattr(uhd.types.StreamMode, mode)
                cmd = uhd.types.StreamCMD(stream_mode)
                print(f"✓ {mode} works!")
                break
        except Exception as e:
            print(f"✗ {mode} failed: {e}")
            
except Exception as e:
    print(f"Error with StreamCMD: {e}")

# Alternative approach - check if we can use send_waveform instead
print("\nChecking if send_waveform is available:")
try:
    usrp_addr = "type=b200"  # Don't need actual device for this test
    usrp = uhd.usrp.MultiUSRP(usrp_addr)
    if hasattr(usrp, 'send_waveform'):
        print("✓ send_waveform is available")
    else:
        print("✗ send_waveform not available")
except Exception as e:
    print(f"Cannot test USRP (expected without hardware): {e}")
