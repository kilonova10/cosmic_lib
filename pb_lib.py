import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt  

from scipy.signal import butter,filtfilt
from scipy import signal
import matplotlib.pyplot as plt


''' given a signal, apply a bandpass filter and return the transformed signal''' 
def apply_bandpass(low, high, sig):
    # Filter requirements.
    fs = 30.0       # sample rate, Hz
    T = len(sig)/fs       # Sample Period
    
    # provide your filter frequencies in Hz
    cutoff = [low, high]

    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2  # sin wave can be approx represented as quadratic
    n = int(T * fs) # total number of samples
    def butter_bandpass_filter(data, cutoff, fs, order):
        # get the filter coefficients 
        b, a = butter(order, [cutoff[0]/nyq, cutoff[1]/nyq], btype='bandpass', analog=False)
        y = filtfilt(b, a, data)
        return y
    y = butter_bandpass_filter(sig, cutoff, fs, order)
    return y

''' remove systematic noise from a signal '''
def get_detrend(sig):
    detrend = signal.detrend(sig)
    return detrend

''' find peaks in a given signal and plot (optional) ''' 
def peak_finder(cleaned, plot = False):
    peaks = signal.find_peaks(cleaned)
    peaks = signal.find_peaks(cleaned, height = 0.2*max(cleaned))
    
    if(plot == True):
        plt.figure(figsize = (20,10))
        plt.scatter(peaks[0], cleaned[peaks[0]], color = 'r', label = 'peaks')
        plt.plot(cleaned, label = 'filtered, detrended Y signal')
        plt.legend()
        plt.xlabel('Frame')
        plt.ylabel('Processed Y signal')
        plt.grid()
        plt.show()
    return (60*30*len(peaks[0])/cleaned.shape[0])