import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import argrelextrema
import statistics as stat
import pyhrv.nonlinear as nl




## READING FILES ##

data_ecg =  pd.read_csv("C:\\Users\\Oskari\\Google Drive (osanlan@utu.fi)\\School\\APfHW\\data\\filt_ecg.csv")
data_ppg = pd.read_csv("C:\\Users\\Oskari\\Google Drive (osanlan@utu.fi)\\School\\APfHW\\data\\filt_ppg.csv")

ecg = data_ecg["sig"]
time_ecg = data_ecg["time"]

ppg = data_ppg["sig"]
time_ppg = data_ppg["time"]




## FINDING PEAKS ##


# peak_finder detecs peaks in the signal and calculates average distances
# between the peaks. Heart rate is calculated as well
def peak_finder(data,fs):
        peaks = argrelextrema(data.to_numpy(), np.greater)
        RR_intervals = np.diff(peaks[0])

        avg_peak_intervals = stat.mean(RR_intervals)
        print("Peaks on average", avg_peak_intervals,
              "measurement points apart")

        # converting RR_intervals to seconds
        RR_intervals_s = [item/fs for item in RR_intervals]

        avg_peak_intervals = stat.mean(RR_intervals_s)
        print("translates", round(avg_peak_intervals, 2), "seconds apart")


        heartRate = round(60.0/avg_peak_intervals,1)
        print("and", heartRate, "bpm\n")
    
        return peaks, RR_intervals_s
    

fs = 200
     
peaks_ecg, ecg_peak_diff = peak_finder(ecg, fs)
peaks_ppg, ppg_peak_diff = peak_finder(ppg, fs)

plt.plot(ecg_peak_diff[1:]) # First one is visually an outlier
plt.plot(ppg_peak_diff)

plt.title("Signal peak intervals")
plt.xlabel("Time (s)")
plt.ylabel("Peak interval (s)")
plt.legend(["ECG","PPG"])

"""
With the one outlier removed the rest of the peaks (or heart beats) are 
0.78-0.90 seconds apart which is plausible
"""

plt.figure()




## POINCARÈ ##

# Using the pyhrv package for heart rate variability
results = nl.poincare(ecg_peak_diff)
print(results["sd1"])

results = nl.poincare(ppg_peak_diff)

print(results["sd1"])

"""
I couldn't find a way to do Lorentz plots and the definition for them was 
the same as for Poincarè plots
"""




## TIME DELAY BETWEEN ECG R PEAK AND PPG FOOT ##


# Convert to arrays
peaks_ecg = np.concatenate(list(peaks_ecg))
peaks_ppg = np.concatenate(list(peaks_ppg))

# Convert to seconds
peaks_ecg_s = [item/fs for item in peaks_ecg]
peaks_ppg_s = [item/fs for item in peaks_ppg]


# Calculate mean difference between peaks
delay = []
for e, p in zip(peaks_ecg_s,peaks_ppg_s):
    delay.append(p-e)
    
print("Average time delay between ecg R-peak and PPG foot is",
      round(np.mean(delay),2), "s")  





















