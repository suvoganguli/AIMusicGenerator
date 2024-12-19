import os
import time
from typing import Optional, Any, List

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from skimage.transform import resize


def timestretch(encodings: np.ndarray,
                factor: float) -> np.ndarray:
  """
  Returns the given encodings timestretch by the given factor.

  :param encodings: the encodings
  :param factor: the timestretch factor, bigger is faster, 1 is idempotent
  :return:
  """
  min_encodings, max_encodings = encodings.min(), encodings.max()
  encodings_norms = (encodings - min_encodings) / (
      max_encodings - min_encodings)
  timestretches = []
  for encodings_norm in encodings_norms:
    timestretch = resize(encodings_norm,
                         (int(encodings_norm.shape[0] * factor),
                          encodings_norm.shape[1]),
                         mode='reflect')
    timestretch = ((timestretch * (max_encodings - min_encodings))
                   + min_encodings)
    timestretches.append(timestretch)
  return np.array(timestretches)


def save_encoding(encodings: List[np.ndarray],
                  filenames: List[str],
                  output_dir: str = "encodings") -> None:
  """
  Saves the given encodings in the given output_dir with
  their corresponding filenames.

  :param encodings: the list of encodings to save
  :param filenames: the list of filename to save the encodings with,
  will add ".npy" if not present
  :param output_dir: the output dir
  """
  os.makedirs(output_dir, exist_ok=True)
  for encoding, filename in zip(encodings, filenames):
    filename = filename if filename.endswith(".npy") else filename + ".npy"
    np.save(os.path.join(output_dir, filename), encoding)


def load_encodings(filenames: List[str],
                   input_dir: str = "encodings") -> List[np.ndarray]:
  """
  Loads the encodings from the given filenames and the given input_dir.

  :param filenames: the list of filename to load the encodings from
  :param input_dir: the input dir
  """
  encodings = []
  for filename in filenames:
    encoding = np.load(os.path.join(input_dir, filename))
    encodings.append(encoding)
  return encodings


def save_encoding_plot(encoding: np.ndarray,
                       filename: Optional[str] = None,
                       output_dir: str = "output") -> None:
  """
  Save the encoding plot of the given encoding to the given filename in the
  given output_dir.

  :param encoding: the encoding to save
  :param filename: the optional filename, set to "%Y-%m-%d_%H%M%S".png if None
  :param output_dir: the output dir
  """
  plt.figure()
  plt.plot(encoding[0])
  if not filename:
    date_and_time = time.strftime("%Y-%m-%d_%H%M%S")
    filename = f"{date_and_time}.png"
  path = os.path.join(output_dir, filename)
  plt.savefig(fname=path, dpi=300)
  plt.close()


def save_spectrogram_plot(audio: Any,
                          sample_rate: int = 16000,
                          filename: Optional[str] = None,
                          output_dir: str = "output") -> None:
  """
  Saves the spectrogram plot of the given audio to the given filename in
  the given output_dir. The resulting plot is a Constant-Q transform (CQT)
  spectrogram with the vertical axis being the amplitude converted to
  dB-scale.

  :param audio: the audio content, as a floating point time series
  :param sample_rate: the sampling rate of the file
  :param filename: the optional filename, set to "%Y-%m-%d_%H%M%S".png if None
  :param output_dir: the output dir
  """
  os.makedirs(output_dir, exist_ok=True)

  # Pitch min and max corresponds to the pitch min and max
  # of the wavenet training checkpoint
  pitch_min = np.min(36)
  pitch_max = np.max(84)
  frequency_min = librosa.midi_to_hz(pitch_min)
  frequency_max = 2 * librosa.midi_to_hz(pitch_max)
  octaves = int(np.ceil(np.log2(frequency_max) - np.log2(frequency_min)))
  bins_per_octave = 32
  num_bins = int(bins_per_octave * octaves)
  hop_length = 2048
  constant_q_transform = librosa.cqt(
    audio,
    sr=sample_rate,
    hop_length=hop_length,
    fmin=frequency_min,
    n_bins=num_bins,
    bins_per_octave=bins_per_octave)
  plt.figure()
  plt.axis("off")
  librosa.display.specshow(
    librosa.amplitude_to_db(constant_q_transform, ref=np.max),
    sr=sample_rate)

  if not filename:
    date_and_time = time.strftime("%Y-%m-%d_%H%M%S")
    filename = f"{date_and_time}.png"
  path = os.path.join(output_dir, filename)
  plt.savefig(fname=path, dpi=600)
  plt.close()


def save_rainbowgram_plot(audio,
                          sample_rate: int = 16000,
                          filename: str = None,
                          output_dir: str = "output") -> None:
  """
  Saves the spectrogram plot of the given audio to the given filename in
  the given output_dir. The resulting plot is a Constant-Q transform (CQT)
  spectrogram with the vertical axis being the amplitude converted to
  dB-scale, and the intensity of lines proportional to the log magnitude of
  the power spectrum and the color given by the derivative of the phase,
  making the phase visible as "rainbow colors", hence the affective name
  "rainbowgrams" (given by the Magenta team).

  :param audio: the audio content, as a floating point time series
  :param sample_rate: the sampling rate of the file
  :param filename: the optional filename, set to "%Y-%m-%d_%H%M%S".png if None
  :param output_dir: the output dir
  """
  os.makedirs(output_dir, exist_ok=True)

  # Configuration from https://arxiv.org/abs/1704.01279
  # and https://gist.github.com/jesseengel/e223622e255bd5b8c9130407397a0494
  peak = 70
  hop_length = 256
  over_sample = 4
  res_factor = 0.8
  octaves = 6
  notes_per_octave = 10
  color_dict = {
    "red": ((0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0)),
    "green": ((0.0, 0.0, 0.0),
              (1.0, 0.0, 0.0)),
    "blue": ((0.0, 0.0, 0.0),
             (1.0, 0.0, 0.0)),
    "alpha": ((0.0, 1.0, 1.0),
              (1.0, 0.0, 0.0))
  }
  color_mask = LinearSegmentedColormap("ColorMask", color_dict)
  plt.register_cmap(cmap=color_mask)

  # Init subplots, there is only one plot but we have to use 2 cmap,
  # which means 2 call to ax.matshow that wouldn"t work with a single plot.
  fig, ax = plt.subplots()
  plt.axis("off")

  bins_per_octave = int(notes_per_octave * over_sample)
  num_bins = int(octaves * notes_per_octave * over_sample)
  constant_q_transform = librosa.cqt(audio,
                                     sr=sample_rate,
                                     hop_length=hop_length,
                                     bins_per_octave=bins_per_octave,
                                     n_bins=num_bins,
                                     filter_scale=res_factor,
                                     fmin=librosa.note_to_hz("C2"))
  mag, phase = librosa.core.magphase(constant_q_transform)
  phase_angle = np.angle(phase)
  phase_unwrapped = np.unwrap(phase_angle)
  dphase = phase_unwrapped[:, 1:] - phase_unwrapped[:, :-1]
  dphase = np.concatenate([phase_unwrapped[:, 0:1], dphase], axis=1) / np.pi
  mag = (librosa.amplitude_to_db(mag,
                                 amin=1e-13,
                                 top_db=peak,
                                 ref=np.max) / peak) + 1
  ax.matshow(dphase[::-1, :], cmap=plt.cm.rainbow)
  ax.matshow(mag[::-1, :], cmap=color_mask)

  if not filename:
    date_and_time = time.strftime("%Y-%m-%d_%H%M%S")
    filename = f"{date_and_time}.png"
  path = os.path.join(output_dir, filename)
  plt.savefig(fname=path, dpi=600)
  plt.close(fig)
