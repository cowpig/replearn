import numpy as np
import theano
import theano.tensor as T
import loader
from sklearn.preprocessing import scale

data = loader.generate_play_set()
np.random.shuffle(data)

n_train = 10000

one_hots = np.array([line[0][0] for line in data[:n_train]])
infos = np.array([line[0][1] for line in data[:n_train]])
infos[0] /= np.max(infos[:, 0])
infos[15] /= np.max(infos[:, 15])
raws = np.array([line[0][2] for line in data[:n_train]], dtype=np.float64)
raws = (raws - np.min(raws)) / (np.max(raws) - np.min(raws))

input_matrix = np.concatenate((one_hots, infos, raws), axis=1)
inputs = theano.shared(input_matrix)

##############
# autoencoder

HIDDEN_UNITS = 500
LEARNING_RATE = 0.1
CORRUPTION = 0.3

n_in = np.shape(input_matrix)[1]
n_hidden = HIDDEN_UNITS

# initialization of weights as suggested in theano tutorials
initialized_W = np.asarray(np.random.uniform(
                                    low=-4 * np.sqrt(6. / (n_hidden + n_in)),
                                    high=4 * np.sqrt(6. / (n_hidden + n_in)),
                                    size=(n_in, n_hidden)), 
                            dtype=np.float64)

W = theano.shared(initialized_W, 'W')

b_in = theano.shared(np.zeros(n_hidden), 'b_in')
b_out = theano.shared(np.zeros(n_in), 'b_out')

x = T.dmatrix('x')

active_hidden = T.nnet.sigmoid(T.dot(x, W) + b_in)
output = T.nnet.sigmoid(T.dot(active_hidden, W.T) + b_out)

entropy = -T.sum(x * T.log(output) + (1 - x) * T.log(1 - output), axis=1)

train_step = theano.function([i, batch_size], output, 
	givens={x:inputs[i:i+batch_size]}, mode="DebugMode")

cost = T.mean(entropy)

parameters = [W, b_in, b_out]
gradients = T.grad(cost, parameters)

updates = []
for param, grad in zip(parameters, gradients):
    updates.append((param, param - LEARNING_RATE * grad))

i, batch_size = T.lscalars('i', 'batch_size')
train_step = theano.function([i, batch_size], cost, updates=updates, 
								givens={x:inputs[i:i+batch_size]})#, mode="DebugMode")

def epoch(batch_size_to_use):
    i=0
    costs = []
    while i < n_train + batch_size_to_use:
    	print "i {}, batch_size {}, n_train {}".format(i, batch_size_to_use, n_train)
        costs.append(train_step(i, batch_size_to_use))

    return costs

while True:
    cost = epoch(1000)
    print "costs: {}".format(cost)
    print "avg: {}".format(np.mean(cost))