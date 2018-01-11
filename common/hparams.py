"""Simple convolutional neural network classififer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

DEFAULT_PARAMS = {
  "optimizer": "Adam",
  "learning_rate": 0.001,
  "momentum": 0.9,
  "warmup_steps": 1000,
  "constant_steps": 200000,
  "decay_steps": 200000,
  "decay_rate": 0.6,
  "batch_size": 1024,
}


def get_params(model, dataset, params=""):
  params_dict = DEFAULT_PARAMS
  params_dict.update(model.get_params())
  params_dict.update(dataset.get_params())

  hp = tf.contrib.training.HParams(**params_dict)
  hp.parse(params)

  return hp
