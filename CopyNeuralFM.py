import os
import sys
import math
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from time import time
import argparse
import LoadData as DATA
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run Neural FM.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='frappe',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=50,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='Pre-train flag. 0: train from scratch; 1: load from pretrain file')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--layers', nargs='?', default='[64,32]',
                        help="Size of each layer.")
    parser.add_argument('--keep_prob', nargs='?', default='[1.0,1.0]',
                        help='Keep probability (i.e., 1-dropout_ratio) for each deep layer and the Bi-Interaction layer. 1: no dropout. Note that the last index is for the Bi-Interaction layer.')
    parser.add_argument('--lamda', type=float, default=0.0001,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='log_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--optimizer', nargs='?', default='AdamOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=0,
                        help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--activation', nargs='?', default='relu',
                        help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')
    parser.add_argument('--early_stop', type=int, default=1,
                        help='Whether to perform early stop (0 or 1)')
    parser.add_argument('--gpu', type=str, default='0', help='gpu index')
    parser.add_argument('--use_mix', type=bool, default=False, help='gpu index')
    parser.add_argument('--learning_decay', type=float, default=0.99, help='learning rate decay ratio')
    return parser.parse_args()


class NeuralFM(BaseEstimator, TransformerMixin):
    def __init__(self, features_M, hidden_factor, layers, loss_type, pretrain_flag, epoch, batch_size, learning_rate,
                 lamda_bilinear,
                 keep_prob, optimizer_type, batch_norm, activation_function, verbose, early_stop, data,
                 random_seed=2016):
        # bind params to class
        self.batch_size = batch_size
        self.hidden_factor = hidden_factor
        self.layers = layers
        self.loss_type = loss_type
        self.pretrain_flag = pretrain_flag
        self.features_M = features_M
        self.lamda_bilinear = lamda_bilinear
        self.epoch = epoch
        self.random_seed = random_seed
        self.keep_prob = np.array(keep_prob)
        self.no_dropout = np.array([1 for i in range(len(keep_prob))])
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.verbose = verbose
        self.activation_function = activation_function
        self.early_stop = early_stop
        self.data = data
        # performance of each epoch
        self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []

        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.train_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            self.feature_weights = tf.placeholder(tf.float32, shape=[None, None])  # None * features_M
            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1
            self.dropout_keep = tf.placeholder(tf.float32, shape=[None])
            self.train_phase = tf.placeholder(tf.bool)

            # Variables.
            self.weights = self._initialize_weights()

            # Model.
            # _________ sum_square part _____________
            # get the summed up embeddings of features.
            nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'],
                                                        self.train_features) * tf.expand_dims(self.feature_weights, 2)
            self.summed_features_emb = tf.reduce_sum(nonzero_embeddings, 1)  # None * K
            # get the element-multiplication
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

            # _________ square_sum part _____________
            self.squared_features_emb = tf.square(nonzero_embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            # ________ FM __________
            self.FM = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K
            if self.batch_norm:
                self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase, scope_bn='bn_fm')
            self.FM = tf.nn.dropout(self.FM, self.dropout_keep[-1])  # dropout at the bilinear interactin layer

            # ________ Deep Layers __________
            for i in range(0, len(self.layers)):
                self.FM = tf.add(tf.matmul(self.FM, self.weights['layer_%d' % i]),
                                 self.weights['bias_%d' % i])  # None * layer[i] * 1
                if self.batch_norm:
                    self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase,
                                                    scope_bn='bn_%d' % i)  # None * layer[i] * 1
                self.FM = self.activation_function(self.FM)
                self.FM = tf.nn.dropout(self.FM, self.dropout_keep[i])  # dropout at each Deep layer
            self.FM = tf.matmul(self.FM, self.weights['prediction'])  # None * 1

            # _________out _________
            Bilinear = tf.reduce_sum(self.FM, 1, keep_dims=True)  # None * 1
            self.Feature_bias = tf.reduce_sum(
                tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features) * tf.expand_dims(
                    self.feature_weights, 2),
                1)  # None * 1
            Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1
            self.out1 = tf.add_n([Bilinear, self.Feature_bias, Bias])  # None * 1

            # Compute the loss.
            if self.loss_type == 'square_loss':
                if self.lamda_bilinear > 0:
                    self.loss = tf.nn.l2_loss(
                        tf.subtract(self.train_labels, self.out1)) + tf.contrib.layers.l2_regularizer(
                        self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
                else:
                    self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out1))
            elif self.loss_type == 'log_loss':
                self.out = tf.sigmoid(self.out1)
                if self.lamda_bilinear > 0:
                    self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, epsilon=1e-07,
                                                           scope=None) + self.lamda_bilinear * (
                                        tf.nn.l2_loss(self.weights['layer_0']) + tf.nn.l2_loss(
                                    self.weights['bias_0']) + tf.nn.l2_loss(self.weights['prediction']))
                else:
                    self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, epsilon=1e-07, scope=None)

            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9).minimize(
                    self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()  # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)

    def _initialize_weights(self):
        all_weights = dict()
        if self.pretrain_flag > 0:  # with pretrain
            pretrain_file = '../pretrain/%s_%d/%s_%d' % (
                args.dataset, args.hidden_factor, args.dataset, args.hidden_factor)
            weight_saver = tf.train.import_meta_graph(pretrain_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            feature_embeddings = pretrain_graph.get_tensor_by_name('feature_embeddings:0')
            feature_bias = pretrain_graph.get_tensor_by_name('feature_bias:0')
            bias = pretrain_graph.get_tensor_by_name('bias:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, pretrain_file)
                fe, fb, b = sess.run([feature_embeddings, feature_bias, bias])
            all_weights['feature_embeddings'] = tf.Variable(fe, dtype=tf.float32)
            all_weights['feature_bias'] = tf.Variable(fb, dtype=tf.float32)
            all_weights['bias'] = tf.Variable(b, dtype=tf.float32)
        else:  # without pretrain
            all_weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.features_M, self.hidden_factor], 0.0, 0.01),
                name='feature_embeddings')  # features_M * K
            all_weights['feature_bias'] = tf.Variable(tf.random_uniform([self.features_M, 1], 0.0, 0.0),
                                                      name='feature_bias')  # features_M * 1
            all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1
        # deep layers
        num_layer = len(self.layers)
        scale = 10.
        if num_layer > 0:
            glorot = np.sqrt(scale / (self.hidden_factor + self.layers[0]))
            all_weights['layer_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.hidden_factor, self.layers[0])), dtype=np.float32)
            all_weights['bias_0'] = tf.Variable(np.zeros((1, self.layers[0]), np.float32))  # 1 * layers[0]
            for i in range(1, num_layer):
                glorot = np.sqrt(scale / (self.layers[i - 1] + self.layers[i]))
                all_weights['layer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.layers[i - 1], self.layers[i])),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                all_weights['bias_%d' % i] = tf.Variable(np.zeros((1, self.layers[i]), np.float32))  # 1 * layer[i]
                # prediction layer
            glorot = np.sqrt(scale / (self.layers[-1] + 1))
            all_weights['prediction'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.layers[-1], 1)),
                                                    dtype=np.float32)  # layers[-1] * 1
        else:
            all_weights['prediction'] = tf.Variable(
                np.ones((self.hidden_factor, 1), dtype=np.float32))  # hidden_factor * 1
        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.train_features: data['X'], self.train_labels: data['Y'], self.feature_weights: data['weight'],
                     self.dropout_keep: self.keep_prob,
                     self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, data, batch_size):  # generate a random block of training data
        start_index = np.random.randint(0, len(data['Y']) - batch_size)
        X, Y = [], []
        # forward get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i + 1
            else:
                break
        # backward get sample
        i = start_index
        while len(X) < batch_size and i >= 0:
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i - 1
            else:
                break
        return {'X': X, 'Y': Y}

    def shuffle_in_unison_scary(self, a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def get_batch(self, X, Y, weights, batch_size):
        index = np.random.permutation(len(Y))
        X = X[index]
        Y = Y[index]
        weights = weights[index]
        batch_list = []
        for i in range(len(Y) // batch_size):
            batch_list.append({'X': X[i * batch_size:i * batch_size + batch_size],
                               'Y': Y[i * batch_size:i * batch_size + batch_size][:, None],
                               'weight': weights[i * batch_size:i * batch_size + batch_size]})
        return batch_list

    def mix_all(self, data, error):
        X = np.array(data['X'])
        Y = np.array(data['Y'])
        W = data['weights']

        ratio = 1
        index = np.random.choice(np.arange(len(X)), int(len(X) * ratio))

        X = np.concatenate([X, X], axis=0)
        Y = np.concatenate([Y, Y], axis=0)
        weights = np.concatenate([W, W], axis=0)
        return X, Y, weights

    def train(self, Train_data, Validation_data, Test_data):  # fit a dataset
        # Check Init performance

        self.train_log = []
        self.valid_log = []
        self.test_log = []
        self.best_prediction = None

        if self.verbose > 0:
            t2 = time()
            self.train_log.append(self.evaluate(Train_data))
            self.valid_log.append(self.evaluate(Validation_data))
            self.test_log.append(self.evaluate(Test_data))
            print("Init: \t train=%.4f %.4f %.4f, validation=%.4f %.4f %.4f, test=%.4f %.4f %.4f [%.1f s]" % (
                self.train_log[-1][0], self.train_log[-1][1], self.train_log[-1][2], self.valid_log[-1][0],
                self.valid_log[-1][1], self.valid_log[-1][2], self.test_log[-1][0], self.test_log[-1][1],
                self.test_log[-1][2],
                time() - t2))

        Y = np.array(Train_data['Y'])
        for epoch in range(self.epoch):
            t1 = time()
            X, Y, W = self.mix_all(Train_data, None)
            total_batch = int(len(Y) / self.batch_size)
            batch_list = self.get_batch(X, Y, W, self.batch_size)
            for i in range(total_batch):
                # generate a batch
                batch_xs = batch_list[i]
                # Fit training
                self.partial_fit(batch_xs)
            t2 = time()

            # output validation
            self.train_log.append(self.evaluate(Train_data))
            self.valid_log.append(self.evaluate(Validation_data))
            self.test_log.append(self.evaluate(Test_data))

            if (self.test_log[-1][1] < np.max(np.array(self.test_log[:-1])[:, 1])):
                self.best_prediction = self.prediction(Test_data)

            if self.verbose > 0 and epoch % self.verbose == 0:
                print("Epoch %d [%.1f s]\ttrain=%.4f %.4f %.4f, validation=%.4f %.4f %.4f, test=%.4f %.4f %.4f [%.1f s]"
                      % (epoch + 1, t2 - t1, self.train_log[-1][0], self.train_log[-1][1], self.train_log[-1][2],
                         self.valid_log[-1][0],
                         self.valid_log[-1][1], self.valid_log[-1][2], self.test_log[-1][0], self.test_log[-1][1],
                         self.test_log[-1][2], time() - t2))
            if (epoch > 5 and self.valid_log[-1][1] < self.valid_log[-2][1] and self.valid_log[-2][1] <
                    self.valid_log[-3][1] and self.valid_log[-3][1] < self.valid_log[-4][1] and self.valid_log[-4][1] <
                    self.valid_log[-5][1] and args.dataset != 'enzymes'):
                break

    def eva_termination(self, valid):
        if self.loss_type == 'square_loss':
            if len(valid) > 5:
                if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
                    return True
        else:
            if len(valid) > 5:
                if valid[-1] < valid[-2] and valid[-2] < valid[-3] and valid[-3] < valid[-4] and valid[-4] < valid[-5]:
                    return True
        return False

    def evaluate(self, data):  # evaluate the results for an input set
        num_example = len(data['Y'])
        X = np.array(data['X'])
        Y = np.array(data['Y'])[:, None]
        weights = data['weights']

        batch_size = 500000
        y_pred = np.zeros(len(X))
        for i in range((len(X) - 1) // batch_size + 1):
            cur_x = X[i * batch_size:np.minimum(i * batch_size + batch_size, len(X))]
            cur_y = Y[i * batch_size:np.minimum(i * batch_size + batch_size, len(X))]
            cur_w = weights[i * batch_size:np.minimum(i * batch_size + batch_size, len(X))]
            feed_dict = {self.train_features: cur_x, self.train_labels: cur_y, self.dropout_keep: self.no_dropout,
                         self.feature_weights: cur_w,
                         self.train_phase: False}
            predictions = self.sess.run((self.out), feed_dict=feed_dict)
            y_pred[i * batch_size:np.minimum(i * batch_size + batch_size, len(X))] = np.reshape(predictions, (-1))

        y_true = np.reshape(data['Y'], (num_example,))

        predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
        predictions_bounded = np.minimum(predictions_bounded,
                                         np.ones(num_example) * max(y_true))  # bound the higher values
        RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
        # RMSE = np.mean(np.sign(predictions_bounded) == y_true)
        from sklearn import metrics
        AUC = metrics.roc_auc_score(y_true, predictions_bounded)
        logloss = log_loss(y_true, predictions_bounded)
        return [RMSE, AUC, logloss]

    def prediction(self, data):  # evaluate the results for an input set
        num_example = len(data['Y'])
        X = np.array(data['X'])
        Y = np.array(data['Y'])[:, None]
        weights = data['weights']

        batch_size = 500000
        y_pred = np.zeros(len(X))
        for i in range((len(X) - 1) // batch_size + 1):
            cur_x = X[i * batch_size:np.minimum(i * batch_size + batch_size, len(X))]
            cur_y = Y[i * batch_size:np.minimum(i * batch_size + batch_size, len(X))]
            cur_w = weights[i * batch_size:np.minimum(i * batch_size + batch_size, len(X))]
            feed_dict = {self.train_features: cur_x, self.train_labels: cur_y, self.dropout_keep: self.no_dropout,
                         self.feature_weights: cur_w,
                         self.train_phase: False}
            predictions = self.sess.run((self.out1), feed_dict=feed_dict)
            y_pred[i * batch_size:np.minimum(i * batch_size + batch_size, len(X))] = np.reshape(predictions, (-1))
        return y_pred


if __name__ == '__main__':
    # Data loading
    import utils

    args = utils.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    data = DATA.LoadData(args.path, args.dataset, args.loss_type)
    if args.verbose > 0:
        print(
            "Neural FM: dataset=%s, hidden_factor=%d, dropout_keep=%s, layers=%s, loss_type=%s, pretrain=%d, #epoch=%d, batch=%d, lr=%.4f, lambda=%.4f, optimizer=%s, batch_norm=%d, activation=%s, early_stop=%d"
            % (args.dataset, args.hidden_factor, args.keep_prob, args.layers, args.loss_type, args.pretrain, args.epoch,
               args.batch_size, args.lr, args.lamda, args.optimizer, args.batch_norm, args.activation, args.early_stop))

    save_file = '../pretrain/%s_%d/%s_%d' % (args.dataset, args.hidden_factor, args.dataset, args.hidden_factor)

    activation_function = tf.nn.relu
    if args.activation == 'sigmoid':
        activation_function = tf.sigmoid
    elif args.activation == 'tanh':
        activation_function == tf.tanh
    elif args.activation == 'identity':
        activation_function = tf.identity

    # Training
    t1 = time()
    model = NeuralFM(data.features_M, args.hidden_factor, eval(args.layers), args.loss_type, args.pretrain, args.epoch,
                     args.batch_size, args.lr, args.lamda, eval(args.keep_prob), args.optimizer, args.batch_norm,
                     activation_function, args.verbose, args.early_stop, data)
    model.train(data.Train_data, data.Validation_data, data.Test_data)

    # Find the best validation result across iterations
    train = np.array(model.train_log)
    valid = np.array(model.valid_log)
    test = np.array(model.test_log)
    best_mse = test[np.argmin(valid[:, 0]), 0]
    best_auc = test[np.argmax(valid[:, 1]), 1]
    best_logloss = test[np.argmin(valid[:, 2]), 2]
    print("Best train = %.4f, valid = %.4f, test = %.4f [%.1f s]"
          % (best_mse, best_auc, best_logloss,
             time() - t1))
    np.save("results/train_train_%s_CopyNeuralFM.npy" % args.dataset, train)
    np.save("results/train_valid_%s_CopyNeuralFM.npy" % args.dataset, valid)
    np.save("results/train_test_%s_CopyNeuralFM.npy" % args.dataset, test)
    np.save("results/best_%s_CopyNeuralFM.npy" % args.dataset, model.best_prediction)

    with open('results/%s_CopyNeuralFM.txt' % args.dataset, 'a') as f:
        f.write("%f %f %f\n" % (best_mse, best_auc, best_logloss))
