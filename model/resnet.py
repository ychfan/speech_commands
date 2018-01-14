"""Simple convolutional neural network classififer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from common import metrics
from common import ops

FLAGS = tf.flags.FLAGS


def get_params():
  return {
      "weight_decay": 0.0002,
      "input_drop_rate": 0.2,
      "drop_rate": 0.5
  }


def model(features, labels, mode, params):
  """CNN classifier model."""
  audios = features["audio"]
  sample_rate = 16000.0
  stfts = tf.contrib.signal.stft(audios, frame_length=400, frame_step=160,
                                 fft_length=1024)
  spectrograms = tf.abs(stfts)
  num_spectrogram_bins = stfts.shape[-1].value
  lower_edge_hertz, upper_edge_hertz, num_mel_bins = 40.0, 7600.0, 40
  linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
      upper_edge_hertz)
  mel_spectrograms = tf.tensordot(
      spectrograms, linear_to_mel_weight_matrix, 1)
  mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))
  log_mel_spectrograms = tf.log(mel_spectrograms + 1e-6)
  audios = log_mel_spectrograms
  
  if mode != tf.estimator.ModeKeys.PREDICT:
    word_labels = labels["word_label"]
    phone_labels = labels["phone_label"]

  training = mode == tf.estimator.ModeKeys.TRAIN
  drop_rate = params.drop_rate if training else 0.0

  #x = tf.layers.dropout(audios, params.input_drop_rate)
  x = audios

  x = tf.layers.conv1d(x, 64, 3, padding="same",
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_decay))

  def res_block(x, training, name):
    with tf.variable_scope(name):
      x = tf.layers.batch_normalization(x, training=training)
      x = tf.nn.relu(x)
      x = tf.layers.conv1d(x, 32, 1, padding="same",
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_decay))

      x = tf.layers.batch_normalization(x, training=training)
      x = tf.nn.relu(x)
      x = tf.layers.conv1d(x, 32, 3, padding="same",
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_decay))

      x = tf.layers.batch_normalization(x, training=training)
      x = tf.nn.relu(x)
      x = tf.layers.conv1d(x, 64, 1, padding="same",
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_decay))
      return x
  for pool in range(3):
    for block in range(3):
      x += res_block(x, training, "block" + str(pool) + str(block))
    x = tf.layers.max_pooling1d(x, 5, 5, padding="same")

  x = tf.layers.flatten(x)
  phone_logits = tf.layers.dense(x, params.num_phones,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_decay))
  word_logits = tf.layers.dense(x, params.num_words,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_decay))

  predictions = tf.argmax(word_logits, axis=-1)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return {"predictions": predictions}, None, None

  loss = tf.losses.hinge_loss(labels=phone_labels, logits=phone_logits) * 0.1 + \
      tf.losses.sparse_softmax_cross_entropy(
          labels=word_labels, logits=word_logits) * 1.0

  eval_metrics = {
      "phone_accuracy": tf.metrics.precision(phone_labels, phone_logits),
      "word_accuracy": tf.metrics.accuracy(word_labels, predictions),
  }

  return {"predictions": predictions}, loss, eval_metrics
