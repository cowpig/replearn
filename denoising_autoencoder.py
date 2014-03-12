import numpy as np
import theano
import theano.tensor as T
import loader

data = loader.generate_play_set()

# shuffle

inputs = theano.shared(data[0])
labels = theano.shared(data[1])