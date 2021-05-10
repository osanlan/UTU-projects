import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import itertools




## READING FILES ##

data_ecg = pd.read_csv("C:\\Users\\Oskari\\Google Drive (osanlan@utu.fi)\\School\\APfHW\\data\\res_ecg.csv")
data_ppg = pd.read_csv("C:\\Users\\Oskari\\Google Drive (osanlan@utu.fi)\\School\\APfHW\\data\\res_ppg.csv")

ecg = data_ecg["sig"]
time_ecg = np.linspace(0,2000/200, 2000)

ppg = data_ppg["sig"]
time_ppg = np.linspace(0,2000/200, 2000)

# Create segments and plot
segments_ecg=[]
for i in range(0,24000,2000):
    segments_ecg.append(ecg[i:(i+2000)])
    plt.plot(time_ecg, ecg[i:(i+2000)],label="ECG",c="b")
      
segments_ppg=[]
for i in range(0,24000,2000):
    segments_ppg.append(ppg[i:(i+2000)])
    plt.plot(time_ppg, ppg[i:(i+2000)],label="PPG",c="y")
    
plt.title("10 second segments")
plt.ylabel("Signal")
plt.xlabel("Time (s)")
plt.legend(["ECG","PPG"]) # Legend colors are a little wonky
plt.show()




## AVERAGING WAVEFORM ##
w=200 # plot window

# transpose dataframe for ecg
segments_ecg_ind = list(map(list, itertools.zip_longest(*segments_ecg, fillvalue=None)))

segments_ecg_ave = []
for i in segments_ecg_ind:
    a = i
    b = [k for k in a if k is not None] 
    segments_ecg_ave.append(np.mean(b))

#plt.plot(segments_ecg_ind, c="y") #individual signals
plt.plot(time_ecg[:w], segments_ecg_ave[:w], c="g")


# transpose dataframe for ppg
segments_ppg_ind = list(map(list,itertools.zip_longest(*segments_ppg, fillvalue=None)))

segments_ppg_ave = []
for i in segments_ppg_ind:
    a = i
    b = [k for k in a if k is not None] 
    segments_ppg_ave.append(np.mean(b))
    
#plt.plot(segments_ppg_ind, c="y") #individual signals
plt.plot(time_ppg[:w], segments_ppg_ave[:w], c="r")


plt.title("Average waveform of signal")
plt.ylabel("Signal")
plt.xlabel("Time (s)")
plt.legend(["ECG","PPG"])
plt.show()




## TIME INTERVALS ##
# find peaks and compute time between them, function already done in previous
# excercises





