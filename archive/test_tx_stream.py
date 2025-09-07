#!/usr/bin/python3
"""
Test tx_streamer methods
"""

import uhd

try:
    usrp_addr = "type=b200,serial=31577EF"
    usrp = uhd.usrp.MultiUSRP(usrp_addr)
    
    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    tx_stream = usrp.get_tx_stream(stream_args)
    
    print("tx_streamer methods:")
    for attr in dir(tx_stream):
        if not attr.startswith('_'):
            print(f"  - {attr}")
            
except Exception as e:
    print(f"Error: {e}")
