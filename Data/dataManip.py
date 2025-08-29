import os
import pandas as pd
import re

# Directory containing your CSV files
directory = "/Users/ericforbes/Documents/GitHub/5G-NTN-Link-Adapation-Matlab/Data/Archive"

for filename in os.listdir(directory):
    if filename.endswith(".csv") and "ThroughputCalulation_BLERw" in filename:
        # Extract window and TarGbler using regex
        match = re.search(r'BLERw(\d+)Tbler([0-9.]+)', filename)
        if match:
            window = int(match.group(1))
            Targetbler = float(match.group(2))
        else:
            window = None
            Targetbler = None

        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)
        df['window'] = window
        df['Targetbler'] = Targetbler

        # Save one directory up from Archive
        parent_dir = os.path.dirname(directory)
        outpath = os.path.join(parent_dir, filename)
        df.to_csv(outpath, index=False)
        print(f"Saved: {filename} to {outpath} with window={window}, Targetbler={Targetbler}")