"""this is the module version"""

import librosa
import numpy as np
import math

# ---------------------------------------------------------------------------- #
# utilities
# ---------------------------------------------------------------------------- #

def ratio_to_cents(r): return 1200.0 * np.log2(r)

def cents_to_ratio(c): return np.power(2, c/1200.0)

def freq_to_midi(f): return 69.0 + 12.0 * np.log2(f/440.0)

def midi_to_freq(m): return np.power(2, (m-69.0)/12.0) * 440.0 if m!= 0.0 else 0.0

# def bin_to_freq(b): return b * float(sr) / float(n_fft)
# def freq_to_bin(f): return int(round(f/(float(sr)/float(n_fft)))
# set params internall to pyfid and call from there?

# ---------------------------------------------------------------------------- #
# ppitch
# ---------------------------------------------------------------------------- #

def ppitch(y, sr=44100, n_fft=4096, win_length=1024, hop_length=2048, 
           num_peaks=20, num_pitches=3):
 
  """Polyphonic pitch tracking.

  :parameters:
    - y
    - sr
    - n_fft
    - hop_length
    - num_peaks

  :returns:
    - pitches
    - STFT
    - peaks

   """

  # other params FUCKING: (for now)

  # -------------------------------------------------------------------------- #
  # go to time-freq
  # -------------------------------------------------------------------------- #

  if_gram, D = librosa.core.ifgram(y, 
                                   sr=sr, 
                                   n_fft=n_fft, 
                                   win_length=win_length,
                                   hop_length=hop_length)

  # -------------------------------------------------------------------------- #
  # find peaks
  # -------------------------------------------------------------------------- #

  peak_thresh = 1e-3

  # bins (freq range) to search [make sure we have room either side]
  min_bin = 2 # make sure there's room below (For peak searching)
  max_bin = 743 # make sure there's room above (for peak searching)

  # things we know
  num_bins, num_frames = if_gram.shape

  # store pitches here
  pitches = np.zeros([num_frames, num_pitches])
  peaks = np.zeros([num_frames,num_peaks])

  # loop through frames
  for i in range(num_frames):

    # grab frequency, magnitudes, total power
    frqs = if_gram[:,i]
    mags = np.abs(D[:,i])
    total_power = mags.sum()

    # neighbor bins
    lower  = mags[(min_bin-1):(max_bin)]
    middle = mags[(min_bin)  :(max_bin+1)]
    upper  = mags[(min_bin+1):(max_bin+2)]
    
    # all local peaks
    peaks_mask_all = (middle > lower) & (middle > upper)

    # resize mask to dimensions of im_gram
    zeros_left = np.zeros(min_bin)
    zeros_right = np.zeros(num_bins - min_bin - max_bin + 1)
    peaks_mask_all = np.concatenate((zeros_left, peaks_mask_all, zeros_right)) 

    # find the first <num_peaks> (at most)
    peaks_mags_all = peaks_mask_all * mags
    top20 = np.argsort(peaks_mags_all)[::-1][0:num_peaks]

    # extract just the peaks (freqs and normed mags)
    peaks_frqs = frqs[top20]
    peaks_mags = mags[top20]

    # note number of peaks found
    num_peaks_found = top20.shape[0]

    # ---------------------------------------------------------------------------#
    # next...
    # ---------------------------------------------------------------------------#

    # params
    # FUCKING: redefined or same as before???
    # these are the range you'll consider for fundamentals
    min_freq = 55.0
    max_freq = 1000.0

    # from min_bin to max_bin in 48ths of an octave

    # [1] bin index to frequency from min_freq to max_freq in 48ths of an octave
    def b2f(index): return min_freq * np.power(np.power(2, 1.0/48.0), index)

    # [2] max_bin is the bin at max_freq (FUCKING: rounds down?)
    max_bin = int(math.log(max_freq / min_freq, math.pow(2, 1.0/48.0))) 

    # now, generate them
    histo = np.fromfunction(lambda x,y: b2f(y), (num_peaks_found, max_bin))
  
    frqs_tile = np.tile(peaks_frqs, (max_bin,1)).transpose()
    mags_tile = np.tile(peaks_mags, (max_bin,1)).transpose()

    # likelihood function for each bin frequency
    def ml_a(amp): 
      """a factor depending on the amplitude of the ith peak"""
      return np.sqrt(np.sqrt(amp))

    def ml_t(freq_ratio):
      """how closely the ith peak is tuned to a multiple of f"""
      max_dist = 25 # cents
      cents = np.abs(ratio_to_cents(freq_ratio))
      dist = np.clip(1.0 - (cents / 25), 0, 1)
      return dist

    def ml_i(nearest_multiple): 
      """whether the peak is closest to a high or low multiple of f"""
      return 1/np.power(nearest_multiple+1, 1) # FUCKING: control param here
      # return np.ones_like(nearest_multiple)

    ml = (ml_a(mags_tile) * \
          ml_t((frqs_tile/histo) / (frqs_tile/histo).round()) * \
          ml_i((frqs_tile/histo).round())).sum(axis=0)

    ml_hat = (ml_a(mags_tile) * \
              ml_t((frqs_tile/histo) / (frqs_tile/histo).round()) * \
              ml_i((frqs_tile/histo).round()))

    pitches[i] = b2f(ml.argsort()[::-1][0:num_pitches])
    peaks[i] = peaks_frqs

  return pitches, D, peaks 

