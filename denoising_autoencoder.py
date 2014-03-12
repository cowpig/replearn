import numpy as np
import theano
import theano.tensor as T
import loader
from sklearn.preprocessing import scale

data = loader.generate_play_set()
np.random.shuffle(data)

n_train = 500

one_hots = np.array([line[0][0] for line in data[:n_train]])
infos = np.array([line[0][1] for line in data[:n_train]])
# scale infos columns 0 and 15?
raws = np.array([line[0][2] for line in data[:n_train]])
# scale(raws, axis=1, copy=False)

inputs = theano.shared(np.concatenate((one_hots, infos, raws), axis=1))

labels = np.array(line[1] for line in data[n_train])

##############
# autoencoder

HIDDEN_UNITS = 500

n_in = np.shape(inputs)[0]
n_hidden = HIDDEN_UNITS

# initialization of weights as suggested in theano tutorials
initialized_W = np.asarray(np.random.uniform(
                                    low=-4 * np.sqrt(6. / (n_hidden + n_in)),
                                    high=4 * np.sqrt(6. / (n_hidden + n_in)),
                                    size=(n_in, n_hidden)), 
                            dtype=np.float64)

W = theano.shared(initialized_W, 'W')
W_out = W.T

b_in = theano.shared(np.zeros(n_in), 'b_in')
b_hidden = theano.shared(np.zeros(n_hidden), 'b_hidden')

