import numpy as np
import theano
import theano.tensor as T
import loader
from sklearn.preprocessing import scale

class HiddenLayer(object):
    def __init__(self, input_tensor, n_in, n_out, activation=T.tanh):
        self.input = input_tensor
        W_values = numpy.asarray(np.random.uniform(
                                    low=-numpy.sqrt(6. / (n_in + n_out)),
                                    high=numpy.sqrt(6. / (n_in + n_out)),
                                    size=(n_in, n_out)), dtype=theano.config.floatX)
        if activation == theano.tensor.nnet.sigmoid:
            W_values *= 4

        self.W = theano.shared(value=W_values, name='W')

        b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b')
        self.output = activation(T.dot(input, self.W) + self.b)

        self.params = [self.W, self.b]

data = loader.generate_play_set()
np.random.shuffle(data)

n_train = 10000

one_hots = np.array([line[0][0] for line in data[:n_train]])
infos = np.array([line[0][1] for line in data[:n_train]])
infos[:, 0] /= np.max(infos[:, 0])
infos[:, 15] /= np.max(infos[:, 15])
raws = np.array([line[0][2] for line in data[:n_train]], dtype=theano.config.floatX)
raws = (raws - np.min(raws)) / (np.max(raws) - np.min(raws))

input_matrix = np.concatenate((one_hots, infos, raws), axis=1)
inputs = theano.shared(input_matrix)

