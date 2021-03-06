import numpy as np
import theano
import theano.tensor as T
import loader
from sklearn.preprocessing import scale
import time

class Autoencoder(object):
    def __init__(self, input_tensor, n_in, n_hidden, learning_rate, pct_blackout=0.2):
        # initialization of weights as suggested in theano tutorials
        initialized_W = np.asarray(np.random.uniform(
                                            low=-4 * np.sqrt(6. / (n_hidden + n_in)),
                                            high=4 * np.sqrt(6. / (n_hidden + n_in)),
                                            size=(n_in, n_hidden)), 
                                    dtype=theano.config.floatX)
        matrixType = T.TensorType(theano.config.floatX, (False,)*2)

        self.W = theano.shared(initialized_W, 'W')

        self.b_in = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX), 'b_in')
        self.b_out = theano.shared(np.zeros(n_in, dtype=theano.config.floatX), 'b_out')

        self.inputs = input_tensor

        self.x = matrixType('x')
        self.noise = T.shared_randomstreams.RandomStreams(1234).binomial(
                            (self.x.shape), n=1, p=1-(pct_blackout), 
                            dtype=theano.config.floatX)
        self.noisy = self.noise * self.x
        self.active_hidden = T.nnet.sigmoid(T.dot(self.noisy, self.W) + self.b_in)
        self.output = T.nnet.sigmoid(T.dot(self.active_hidden, self.W.T) + self.b_out)

        self.entropy = -T.sum(self.x * T.log(self.output) + 
                                (1 - self.x) * T.log(1 - self.output), axis=1)

        self.cost = T.mean(self.entropy)

        self.parameters = [self.W, self.b_in, self.b_out]
        self.gradients = T.grad(self.cost, self.parameters)

        self.updates = []
        for param, grad in zip(self.parameters, self.gradients):
            self.updates.append((param, param - learning_rate * grad))

        i, batch_size = T.iscalars('i', 'batch_size')
        self.train_step = theano.function([i, batch_size], self.cost, 
                                            updates=self.updates, 
                                            givens={self.x:self.inputs[i:i+batch_size]})
                                            #, mode="DebugMode")

def epoch(batch_size_to_use, n_train, theano_function):
    i=0
    costs = []
    while i + batch_size_to_use < n_train:
        # print "i {}, batch_size {}, n_train {}".format(i, batch_size_to_use, n_train)
        costs.append(theano_function(i, batch_size_to_use))
        i += batch_size_to_use

    return costs

if __name__ == "__main__":
    data = loader.generate_play_set()
    np.random.shuffle(data)

    n_train = 100000

    one_hots = np.array([line[0][0] for line in data[:n_train]], dtype=theano.config.floatX)
    infos = np.array([line[0][1] for line in data[:n_train]], dtype=theano.config.floatX)
    infos[:, 0] /= np.max(infos[:, 0])
    infos[:, 15] /= np.max(infos[:, 15])
    raws = np.array([line[0][2] for line in data[:n_train]], dtype=theano.config.floatX)
    raws = (raws - np.min(raws)) / (np.max(raws) - np.min(raws))

    input_matrix = np.concatenate((one_hots, infos, raws), axis=1)
    inputs = theano.shared(input_matrix, borrow=True)

    HIDDEN_UNITS = 500
    LEARNING_RATE = 0.1

    aa = Autoencoder(inputs, np.shape(input_matrix)[1], HIDDEN_UNITS, LEARNING_RATE)
    train_step = aa.train_step

    cost = [99999]
    n = 0
    start = time.time()
    print time
    for x in xrange(50):
        n += 1
        cost = epoch(1000, n_train, train_step)
        print "=== epoch {} ===".format(n)
        print "costs: {}".format([line[()] for line in cost])
        print "avg: {}".format(np.mean(cost))

    elapsed = (time.time() - start)
    print "ELAPSED TIME: {}".format(elapsed)