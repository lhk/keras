import pytest
import numpy as np
import copy
from numpy.testing import assert_allclose

from keras.utils import CustomObjectScope
from keras.layers import wrappers, Input, Layer
from keras.layers import RNN
from keras import layers
from keras.models import Sequential, Model, model_from_json
from keras import backend as K
from keras.utils.generic_utils import object_list_uid, to_list
import tensorflow as tf

def test_WeightDropout():
    rate = 0.5
    seed = 0
    # first, test with Dense layer
    model = Sequential()
    model.add(wrappers.WeightDropout(layers.Dense(2), rate=rate, seed=seed, input_shape=(40,)))
    model.add(layers.Activation('relu'))
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(np.random.random((10, 40)), np.random.random((10, 2)),
              epochs=1,
              batch_size=10)

    file_writer = tf.summary.FileWriter('/home/lhk/logs', K.get_session().graph)
    # test config
    model.get_config()

    # test when specifying a batch_input_shape
    test_input = np.random.random((100, 40))
    test_output = model.predict(test_input)
    weights = model.layers[0].get_weights()

    reference = Sequential()
    reference.add(wrappers.WeightDropout(layers.Dense(2), rate=rate, seed=seed,
                                         batch_input_shape=(1, 40)))
    reference.add(layers.Activation('relu'))
    reference.compile(optimizer='rmsprop', loss='mse')
    reference.layers[0].set_weights(weights)

    reference_output = reference.predict(test_input)
    assert_allclose(test_output, reference_output, atol=1e-05)


    # now set learning phase and check if weights are dropped
    K.set_learning_phase(True)
    test_output = model.predict(test_input)
    weights = model.layers[0].get_weights()


if __name__ == '__main__':
    pytest.main([__file__])
