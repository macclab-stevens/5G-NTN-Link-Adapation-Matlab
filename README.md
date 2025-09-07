# 5G-NTN-Link-Adapation-Matlab
A Matlab script to develop link adaptation algorithms for Non-Terrestrial Networks and Environments

<img width="1063" alt="image" src="https://github.com/macclab-stevens/5G-NTN-Link-Adapation-Matlab/assets/163568786/1c75d8b4-69f9-4443-84b2-68feda5b0a05">

# DataSet
The data set can be found hosted on Harvard Dataverse. And can be saved in the /Data/ folder for consistency.

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/BXBOTB


# scripts for reference:

tail -f /tmp/gnb.log | awk '{if ($0~/sinr\"/)print$2/2-23; }

# configuraiton notes:
```yml
ru_sdr:
  device_driver: uhd
  device_args: type=b200,serial=315AF4C,num_recv_frames=64,num_send_frames=64
  srate: 23.04
  otw_format: sc12
  tx_gain: 62 #80
  rx_gain: 3 #64
```
produces
```
DL SINR : 5-5.5
PUSCH   : 5.5
```