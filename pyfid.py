"""this is the module version"""

import librosa
import numpy as np
import math

# --------------------------------------------------------------------------- #
# utilities
# --------------------------------------------------------------------------- #

def ratio_to_cents(r): 
  return 1200.0 * np.log2(r)

def cents_to_ratio(c): 
  return np.power(2, c/1200.0)

def freq_to_midi(f): 
  return 69.0 + 12.0 * np.log2(f/440.0)

def midi_to_freq(m): 
  return np.power(2, (m-69.0)/12.0) * 440.0 if m!= 0.0 else 0.0

# def bin_to_freq(b): return b * float(sr) / float(n_fft)
# def freq_to_bin(f): return int(round(f/(float(sr)/float(n_fft)))
# set params internall to pyfid and call from there?

# --------------------------------------------------------------------------- #
# ppitch
# --------------------------------------------------------------------------- #

def ppitch(y, sr=44100, n_fft=4096, win_length=1024, hop_length=2048, 
  num_peaks=20, num_pitches=3, min_bin=2, max_bin=743, min_freq=55.0, 
  max_freq=1000.0):
 
  """Polyphonic pitch estimation.

  :parameters:
    - y (time series)
    - sr
    - n_fft
    - hop_length
    - num_peaks
    - num_pitches
    - min_bin: min_bin for peak picking
    - max_bin: max_bin for peak picking
    - min_freq: min fundamental freq
    - max_freq: max fundamental freq

  :returns:
    - pitches
    - STFT
    - peaks

  :todo:
    -> min/max_bin should be in freqs so invariant to fft size

   """

  # other params FUCKING: (for now)

  # ------------------------------------------------------------------------- #
  # go to time-freq
  # ------------------------------------------------------------------------- #

  if_gram, D = librosa.core.ifgram(y, sr=sr, n_fft=n_fft, 
    win_length=win_length,hop_length=hop_length)

  # ------------------------------------------------------------------------- #
  # find peaks
  # ------------------------------------------------------------------------- #

  peak_thresh = 1e-3

  # bins (freq range) to search [make sure we have room either side]
  #min_bin = min_bin # \\--> make sure there's room below (for peak searching)
  #max_bin = max_bin # \\--> make sure there's room above (for peak searching)

  # things we know
  num_bins, num_frames = if_gram.shape

  # store pitches here
  pitches = np.zeros([num_frames, num_pitches])       # pitch bins
  peaks = np.zeros([num_frames, num_peaks])           # top20 peaks
  fundamentals = np.zeros([num_frames, num_pitches])  # funds (least squares)
  confidences = np.zeros([num_frames, num_pitches])   # confidence scores

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

    # ----------------------------------------------------------------------- #
    # maximum liklihood 
    # ----------------------------------------------------------------------- #

    # range we'll consider for fundamentals
    #min_freq = min_freq
    #max_freq = max_freq

    # from min_bin to max_bin in 48ths of an octave (1/4 tones)

    # [1] bin index to frequency min_freq to max_freq in 48ths of an octave
    def b2f(index): return min_freq * np.power(np.power(2, 1.0/48.0), index)

    # [2] max_histo_bin is the bin at max_freq (FUCKING: rounds down?)
    max_histo_bin = int(math.log(max_freq / min_freq, math.pow(2, 1.0/48.0))) 

    # now, generate them
    histo = np.fromfunction(lambda x,y: b2f(y), (num_peaks_found, max_histo_bin))
  
    frqs_tile = np.tile(peaks_frqs, (max_histo_bin,1)).transpose()
    mags_tile = np.tile(peaks_mags, (max_histo_bin,1)).transpose()

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

    indices = ml.argsort()[::-1][0:num_pitches]
    pitches[i] = b2f(indices)
    confidences[i] = ml[indices]
    peaks[i] = peaks_frqs

    # ----------------------------------------------------------------------- #
    # estimate fundamental (least squares)
    # ----------------------------------------------------------------------- #

    """estimate fundamental (least squares)

    here we solve a least squares approximation to give a more precise
    estimate of the fundamental within a histogram bin.
    
    least squares solution :: WAx ~ Wb
   
    A = matrix of harmonic integers (num_peaks x 1)
    b = matrix of actual frequencies (num_peaks x 1)
    W = matrix of weights (on digaonal, num_peaks x num_peaks)
    x = fundamental (singletone matrix)
   
    """

    ml_peaks = pitches[i] # regions strong ml
    width = 25 # count peaks w/in 25 cents

    frame_fundamentals = []
    for bin_frq in ml_peaks:

      # vector of nearest harmonics
      nearest_harmonic = (peaks_frqs/bin_frq).round()

      # mask in range? vector of bools
      mask = np.abs(ratio_to_cents( 
        (peaks_frqs/bin_frq) / (peaks_frqs/bin_frq).round())) <= width

      # weight (same harmonic weight as above)
      weights = ml_i( (peaks_frqs/bin_frq).round() )

      # build matrices
      A = np.matrix(nearest_harmonic).T
      b = np.matrix(peaks_frqs).T
      W = np.matrix(np.diag(mask * weights))

      # do least squares
      fund = np.linalg.lstsq(W*A, W*b)[0][0].item()

      # append
      frame_fundamentals += [fund]

    fundamentals[i] = np.array(frame_fundamentals)

  # ------------------------------------------------------------------------- #
  # pitch tracks
  # ------------------------------------------------------------------------- #

  """pitch tracks

  here we parse the raw frequencies into pitch tracks. currently, we do this 
  by minimizing the abs of the frame to frame difference, but we could use
  any feature, such as distance from the running mean, variance, etc.

    todo: --> implement more pitch track options
    
  """

  # move through the frames
  # have pitch tracks ready to go
  # for each num_pitches put in track with nearest frame to frame difference
  # note: what if the pitch confidence is low, might want to put track to sleep...
  # or if it's too far from anything just for a sec
  # or more sophisticated for like minimizing total error...
  # here we do it dumb way from strong to weak closest match (pref to first)
  tracks = np.zeros([num_frames, num_pitches])   # confidence scores
  
  tracks[0] = fundamentals[0]

  tracks[0] = fundamentals[0]
  for i in range(1, num_frames):
    prev_tracks = list(tracks[i-1])
    new_funds = list(fundamentals[i])
    for j in range(num_pitches):
      wewant = min(new_funds, key=lambda x: abs(x-prev_tracks[j]))
      index = new_funds.index(wewant)
      new_funds.pop(index)
      tracks[i,j]= wewant

  return fundamentals, pitches, D, peaks, confidences, tracks

