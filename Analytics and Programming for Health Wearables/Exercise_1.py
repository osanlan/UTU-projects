import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt




## READING FILES ##

data =  pd.read_csv("C:\\Users\\Oskari\\Google Drive (osanlan@utu.fi)\\School\\APfHW\\data\\ecg_ppg_clean.csv")

ecg = data["ecg"]
time_ecg = data["timestamps_ecg"]

ppg_1_ir = data["ppg_1_ir"]
ppg_1_ir = ppg_1_ir.dropna()
time_ppg_1_ir = data["timestamps_ppg_1_ir"]
time_ppg_1_ir = time_ppg_1_ir[:len(ppg_1_ir)] # removing timestamps for na rows




## PLOTTING ##
# standardizations
ecg = ecg - np.mean(ecg)
ecg = ecg / np.max(ecg)

ppg_1_ir = ppg_1_ir * -1
ppg_1_ir = ppg_1_ir - np.mean(ppg_1_ir)
ppg_1_ir = ppg_1_ir / np.max(ppg_1_ir)


time_ecg = np.linspace(0,len(ecg)/128, len(ecg))
time_ppg_1_ir = np.linspace(0,len(ppg_1_ir)/100, len(ppg_1_ir))

plt.figure()
plt.plot(time_ecg, ecg, linewidth=1, label="ecg")
plt.plot(time_ppg_1_ir, ppg_1_ir, "orange", label="ppg_1_ir",linewidth=1)
plt.legend()
plt.ylabel("Signal")
plt.xlabel("Time (s)")

plt.title("ECG and PPG_1_IR vs time")

# Visual inspection of signals:
# Look like normal ecg and ppg


## RESAMPLING ##

fs = 200

resampled_ecg = signal.resample(ecg, 24000)#round(len(ecg)/128*200))
resampled_ppg_1_ir = signal.resample(ppg_1_ir, 24000)#round(len(ppg_1_ir)/100*200))

time_new_ecg = np.linspace(0, len(resampled_ecg)/fs, len(resampled_ecg))
time_new_ppg = np.linspace(0, len(resampled_ecg)/fs, len(resampled_ppg_1_ir))

plt.figure()
plt.plot(time_new_ecg, resampled_ecg)
plt.plot(time_new_ppg, resampled_ppg_1_ir)
plt.legend(["ECG","PPG"])
plt.ylabel("Signal")
plt.xlabel("Time (s)")

plt.title("Resampled signals")


to_save = pd.DataFrame(resampled_ecg)
to_save.columns = ["sig"]
to_save["time"] = time_new_ecg
to_save.to_csv("data\\res_ecg.csv", columns=["time","sig"], index=False)

to_save = pd.DataFrame(resampled_ppg_1_ir)
to_save.columns = ["sig"]
to_save["time"] = time_new_ppg
to_save.to_csv("data\\res_ppg.csv", columns=["time","sig"], index=False)

## FREQUENCY DOMAIN ##

PSD_freqs_ecg, PSD_ecg = signal.welch(resampled_ecg, fs)
PSD_freqs_ppg, PSD_ppg = signal.welch(resampled_ppg_1_ir, fs)

plt.figure(figsize=(15,6))
plt.semilogy(PSD_freqs_ecg, PSD_ecg)
plt.semilogx(PSD_freqs_ppg, PSD_ppg)
plt.xlabel("frequency [Hz]")
plt.ylabel("PSD [V**2/Hz]")
plt.legend(["ECG","PPG"])

plt.title("PSD vs frequency")


freqs_df_ecg = pd.DataFrame({'PSD': PSD_ecg})
freqs_df_ecg.index = PSD_freqs_ecg

dominant_freq_ecg = freqs_df_ecg[0.7:3].idxmax().values[0]
print("The dominant frequenzy for ecg is", dominant_freq_ecg, "Hz")

freqs_df_ppg = pd.DataFrame({'PSD': PSD_ppg})
freqs_df_ppg.index = PSD_freqs_ppg

dominant_freq_ppg = freqs_df_ppg[0.7:3].idxmax().values[0]
print("The dominant frequenzy for ppg is", dominant_freq_ppg, "Hz")

# Most infomation is between 0.7 and 3 Hz as expected since human produced 
# signal are between there


## FILTER ##

def bandPassFilter(sig, order, dom_freq):

  fs = 132
  lowcut = dom_freq - 0.3
  highcut = dom_freq + 0.3
  
  # maximum frequency is Nyquist frequency which his half of the sampling freq
  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq

  b, a = signal.butter(order, [low, high], btype='band', analog=False)
  filt_signal = signal.filtfilt(b, a, sig, axis=0)

  return filt_signal


filt_ecg = bandPassFilter(resampled_ecg, 2, dominant_freq_ecg)
fig = plt.figure(figsize=(30,7))
plt.plot(time_new_ecg, filt_ecg, label="filtered ecg signal", linewidth=0.5)

fig.legend(bbox_to_anchor=(0.20,0.8));

filt_ppg = bandPassFilter(resampled_ppg_1_ir, 2, dominant_freq_ppg)
fig = plt.figure(figsize=(30,7))
plt.plot(time_new_ppg, filt_ppg,
         label="filtered ppg_1_ir signal", linewidth=0.5)

fig.legend(bbox_to_anchor=(0.20,0.8));



filt_ecg = pd.DataFrame(filt_ecg)
filt_ecg.columns = ["sig"]
filt_ecg["time"] = time_new_ecg
filt_ecg.to_csv("data\\filt_ecg.csv", columns=["time","sig"], index=False)

filt_ppg = pd.DataFrame(filt_ppg)
filt_ppg.columns = ["sig"]
filt_ppg["time"] = time_new_ppg
filt_ppg.to_csv("data\\filt_ppg.csv", columns=["time","sig"], index=False)

