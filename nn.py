import numpy as np
import theano
import theano.tensor as T
import loader
from sklearn.preprocessing import scale
from denoising_autoencoder import epoch

class NNLayer(object):
    def __init__(self, input_tensor, n_in, n_out, activation=T.tanh):
        self.input = input_tensor
        W_values = np.asarray(np.random.uniform(
                                    low=-np.sqrt(6. / (n_in + n_out)),
                                    high=np.sqrt(6. / (n_in + n_out)),
                                    size=(n_in, n_out)), dtype=theano.config.floatX)

        if activation == theano.tensor.nnet.sigmoid:
            W_values *= 4

        self.W = theano.shared(value=W_values, name='W')

        b_values = np.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b')
        self.output = activation(T.dot(input_tensor, self.W) + self.b)

        self.params = [self.W, self.b]



if __name__ == "__main__":
    data = loader.generate_play_set()
    np.random.shuffle(data)

    n_train = 10000

    one_hots = np.array([line[0][0] for line in data[:n_train]])
    infos = np.array([line[0][1] for line in data[:n_train]])
    infos[:, 0] /= np.max(infos[:, 0])
    infos[:, 15] /= np.max(infos[:, 15])
    # TODO: figure this out
    # infos_boolean = np.concatenate(infos[:, 1:15], infos[:,16:])
    raws = np.array([line[0][2] for line in data[:n_train]], dtype=theano.config.floatX)
    raws = (raws - np.min(raws)) / (np.max(raws) - np.min(raws))

    input_matrix = np.concatenate((one_hots, infos, raws), axis=1)
    inputs = theano.shared(input_matrix)

    label_mat = np.mat([line[1] for line in data[:n_train]], dtype=theano.config.floatX).transpose()
    label_mat = (label_mat - np.min(label_mat)) / (np.max(label_mat) - np.min(label_mat))
    labels = theano.shared(label_mat)


    # network
    LEARNING_RATE = 0.01

    x = T.dmatrix('x')
    y = T.dmatrix('y')

    hidden1 = NNLayer(x, np.shape(input_matrix)[1], 500)
    hidden2 = NNLayer(hidden1.output, 500, 500)
    out_layer = NNLayer(hidden2.output, 500, 1)

    # i, batch_size = T.lscalars('i', 'batch_size')
    
    # train_step = theano.function([i, batch_size], out_layer.output, 
    #                             givens={x : inputs[i:i+batch_size]})
                                # mode="DebugMode")

    params = hidden1.params + hidden2.params + out_layer.params
    cost = T.sqrt(T.mean(T.sqr(out_layer.output - y)))

    gradients = T.grad(cost, params)
    updates = []
    for param, grad in zip(params, gradients):
        updates.append((param, param - LEARNING_RATE * grad))

    i, batch_size = T.lscalars('i', 'batch_size')
    
    train_step = theano.function([i, batch_size], cost, 
                                updates=updates, 
                                givens={x : inputs[i:i+batch_size],
                                        y : labels[i:i+batch_size]})
                                # mode="DebugMode")

    n = 0
    while True:
        n += 1
        cost = epoch(1000, n_train, train_step)
        print "=== epoch {} ===".format(n)
        print "costs: {}".format([line[()] for line in cost])
        print "avg: {}".format(np.mean(cost))