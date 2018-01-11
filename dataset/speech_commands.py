"""Mnist dataset preprocessing and specifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import numpy as np
import os
import random
from six.moves import urllib
import struct
import tensorflow as tf
from scipy.io import wavfile
from python_speech_features import logfbank

TRAIN_DIR = "data/train/"
TRAIN_AUDIO_DIR = TRAIN_DIR + "audio/"
VAL_LIST_FILE = "validation_list.txt"
TEST_LIST_FILE = "testing_list.txt"
TEST_DIR = "data/test/"
TEST_AUDIO_DIR = TEST_DIR + "audio/"

AUDIO_SIZE = 16000
NUM_CLASSES = 12
NUM_CHANNELS = 40
WORDS = ["yes", "no", "up", "down", "left", "right",
         "on", "off", "stop", "go", "silence", "unknown"]
DICT = {
    "_background_noise_": set(),
    "bed": {"B", "EH", "D"},
    "bird": {"B", "ER", "D"},
    "cat": {"K", "AE", "T"},
    "dog": {"D", "AO", "G"},
    "down": {"D", "AW", "N"},
    "eight": {"EY", "T"},
    "five": {"F", "AY", "V"},
    "four": {"F", "AO", "R"},
    "go": {"G", "OW"},
    "happy": {"HH", "AE", "P", "IY"},
    "house": {"HH", "AW", "S"},
    "left": {"L", "EH", "F", "T"},
    "marvin": {"M", "AA", "R", "V", "IH", "N"},
    "nine": {"N", "AY"},
    "no": {"N", "OW"},
    "off": {"AO", "F"},
    "on": {"AA", "N"},
    "one": {"W", "AH", "N"},
    "right": {"R", "AY", "T"},
    "seven": {"S", "EH", "V", "AH", "N"},
    "sheila": {"SH", "IY", "L", "AH"},
    "six": {"S", "IH", "K"},
    "stop": {"S", "T", "AA", "P"},
    "three": {"TH", "R", "IY"},
    "tree": {"T", "R", "IY"},
    "two": {"T", "UW"},
    "up": {"AH", "P"},
    "wow": {"W", "AW"},
    "yes": {"Y", "EH", "S"},
    "zero": {"Z", "IY", "R", "OW"}
}
PHONES = sorted(set.union(*list(DICT.values())))
TRAIN_WORDS = sorted(DICT.keys())
WORD_LABELS = np.zeros([len(DICT.keys()), 1], np.int32)
PHONE_LABELS = np.zeros([len(DICT.keys()), len(PHONES)], np.int32)
TRAIN_LIST = []
VAL_LIST = []
TEST_LIST = []


def get_params():
  """Return dataset parameters."""
  return {
      "num_words": len(WORDS),
      "num_phones": len(PHONES),
  }


def prepare():
  """This function will be called once to prepare the dataset."""
  PHONE_LABELS.fill(0)
  for i, word in enumerate(TRAIN_WORDS):
    if word == "_background_noise_":
      word_label = WORDS.index("silence")
    elif word not in WORDS:
      word_label = WORDS.index("unknown")
    else:
      word_label = WORDS.index(word)
    WORD_LABELS[i, 0] = word_label
    for j, phone in enumerate(PHONES):
      if phone in DICT[word]:
        PHONE_LABELS[i, j] = 1

  TRAIN_LIST[:] = []
  VAL_LIST[:] = []
  TEST_LIST[:] = []
  with open(TRAIN_DIR + VAL_LIST_FILE) as f:
    VAL_LIST.extend(f.read().splitlines())
  for word in TRAIN_WORDS:
    for file in os.listdir(TRAIN_AUDIO_DIR + word):
      if file.endswith(".wav"):
        filepath = word + "/" + file
        if filepath not in VAL_LIST:
          TRAIN_LIST.append(filepath)
        if word == "_background_noise_":
          for i in range(100):
            TRAIN_LIST.append(filepath)
  random.shuffle(TRAIN_LIST)
  for file in os.listdir(TEST_AUDIO_DIR):
    if file.endswith(".wav"):
      TEST_LIST.append(file)
  TEST_LIST.sort()


def read(mode):
  """Create an instance of the dataset object."""
  file_list = {
      tf.estimator.ModeKeys.TRAIN: TRAIN_LIST,
      tf.estimator.ModeKeys.EVAL: VAL_LIST,
      tf.estimator.ModeKeys.PREDICT: TEST_LIST
  }[mode]

  def gen():
    for file in file_list:
      if mode == tf.estimator.ModeKeys.PREDICT:
        idx = 0
        filepath = TEST_AUDIO_DIR + file
      else:
        word = file.split("/", 1)[0]
        idx = TRAIN_WORDS.index(word)
        filepath = TRAIN_AUDIO_DIR + file
      rate, sig = wavfile.read(filepath)
      if len(sig) < AUDIO_SIZE:
        # print(len(sig))
        sig_pad = np.zeros(AUDIO_SIZE)
        offset = (AUDIO_SIZE - len(sig)) // 2
        sig_pad[offset:offset + len(sig)] = sig
        sig = sig_pad
      elif len(sig) > AUDIO_SIZE:
        offset = random.randrange(len(sig) - AUDIO_SIZE)
        sig = sig[offset:offset + AUDIO_SIZE]
      fbank_feat = logfbank(sig, rate, nfilt=NUM_CHANNELS)
      yield (fbank_feat, WORD_LABELS[idx], PHONE_LABELS[idx])

  return tf.contrib.data.Dataset.from_generator(
      gen, (tf.float32, tf.int32, tf.int32), ([99, NUM_CHANNELS], [1, ], [len(PHONES), ])
  )


def parse(mode, audio, word_label, phone_label):
  """Parse input record to features and labels."""
  audio = tf.to_float(audio)
  word_label = tf.to_int32(word_label)
  phone_label = tf.to_int32(phone_label)
  return {"audio": audio}, {"word_label": word_label, "phone_label": phone_label}
