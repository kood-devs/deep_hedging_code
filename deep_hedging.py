"""
    Deep hedging strategy implementation (simple dense newral network based...)
    see also: https://nbviewer.jupyter.org/urls/people.math.ethz.ch/~jteichma/lecture_ml_web/deep_hedging_keras_bsanalysis.ipynb
"""
from datetime import datetime

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

import keras.backend as K
from keras import initializers
from keras.layers import Input, Dense, Subtract, Multiply, Lambda, Add, BatchNormalization, Activation, concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model  # to visualize model structure

# CONST
# Stock process params
N = 30  # time disrectization
S0 = 100
strike = 100
T = 1/12
sigma = 0.20

# Neural network structure
m = 1  # dimension of price
d = 3  # number of layers in strategy
n = 32  # nodes in the layers (common in all intermediate layers)

# Learning params
NUM_TRAIN = 10**3
NUM_TEST = 10**3
EPOCH = 5000
# BATCH_SIZE = 256  # this param is not used...


def calculate_BS_price(S0, strike, T, sigma, r=0.00):
    d1 = (np.log(S0/strike) + (r + 0.5 * sigma**2) * T) / (np.sqrt(T) * sigma)
    d2 = (np.log(S0/strike) + (r - 0.5 * sigma**2) * T) / (np.sqrt(T) * sigma)
    fst = S0 * norm.cdf(d1)
    snd = strike * norm.cdf(d2)
    return fst - snd


def generate_stock_process(num_path):
    x_val = ([S0*np.ones((num_path, m))] + [np.zeros((num_path, m))] +
             [np.random.normal(-(sigma)**2*T/(2*N), sigma*np.sqrt(T)/np.sqrt(N), (num_path, m)) for i in range(N)])
    y_val = np.zeros((num_path, 1))
    return x_val, y_val


class GenerateStockPrice:
    def __init__(self, num_path):
        self.num_path = num_path

    def generate_stock_process(self):
        def generate_input():
            while True:
                x_val = ([S0*np.ones((self.num_path, m))] + [np.zeros((self.num_path, m))] +
                         [np.random.normal(-(sigma)**2*T/(2*N), sigma*np.sqrt(T)/np.sqrt(N), (self.num_path, m)) for i in range(N)])
                y_val = np.zeros((self.num_path, 1))
                yield x_val, y_val
        return generate_input()


def main():
    """
    This code is written following "keras functional API" guide.
    If model has multi inputs, we have to use functional API, not Sequence.

    see also: https://keras.io/ja/getting-started/functional-api-guide/
    """

    # ---------------------------------------
    #  model construction
    # ---------------------------------------
    # Architecture
    # architecture is the same for all networks(from t=0 to T)
    layers = []
    # batch normalized layers
    for j in range(N):
        for i in range(d):
            if i < d - 1:
                # Use he_normal for weight initialization
                # see also: https://qiita.com/mgmk2/items/dc562303e178aa6306ca
                nodes = n
                layer = Dense(nodes,
                              kernel_initializer=initializers.he_normal(),
                              bias_initializer=initializers.he_uniform(),
                              name=str(i)+str(j))
            else:
                nodes = m
                layer = Dense(nodes,
                              kernel_initializer=initializers.he_uniform(),
                              bias_initializer=initializers.he_uniform(),
                              name=str(i)+str(j))
            layers.append(layer)

    # Implementing the loss function
    price = Input(shape=(m,))
    hedge = Input(shape=(m,))
    inputs = [price]+[hedge]

    for j in range(N):
        if j == 0:
            # case t = 0
            strategy = price
            inital_layer = Dense(1, use_bias=False, activation='linear',
                                 kernel_initializer=initializers.he_normal(),
                                 name='S0')
            strategy = inital_layer(strategy)
        else:
            # strategy = concatenate([logprice, strategy])  # doesn't work
            # strategy = logprice
            strategy = hedge
            for k in range(d):
                # batch normalization
                # see also: https://qiita.com/t-tkd3a/items/14950dbf55f7a3095600
                if k < d - 1:
                    strategy = layers[k+j*d](strategy)
                    strategy = BatchNormalization()(strategy)
                    strategy = Activation('relu')(strategy)
                else:
                    strategy = layers[k+j*d](strategy)
                    # strategy = BatchNormalization()(strategy)
                    strategy = Activation('linear')(strategy)
        incr = Input(shape=(m,))  # log(s_t+1)
        logprice = Lambda(lambda x: K.log(x))(price)  # s_t => log(s_t)
        logprice = Add()([logprice, incr])  # log(s_t) => log(s_t+1)
        pricenew = Lambda(lambda x: K.exp(x))(logprice)  # log(s_t+1) => s_t+1
        priceincr = Subtract()([pricenew, price])  # s_t+1 - s_t
        hedgenew = Multiply()([strategy, priceincr])  # 今期の損益
        hedge = Add()([hedge, hedgenew])  # ヘッジ損益を更新
        inputs.append(incr)
        price = pricenew
    # output layer
    priceBS = calculate_BS_price(S0, strike, T, sigma)
    # payoff = Lambda(lambda x: 0.5*(K.abs(x-strike)+x-strike))(price)
    payoff = Lambda(lambda x: 0.5*(K.abs(x-strike)+x-strike) - priceBS)(price)
    outputs = Subtract()([payoff, hedge])  # payoff minus priceBS minus hedge

    # inputs are initial price, hedging strategy(init 0) and increments of the log price process
    model_hedge = Model(inputs=inputs, outputs=outputs)
    # model_hedge.summary()
    plot_model(model_hedge, to_file='result/model_hedge_{}.png'.format(
        datetime.today().strftime("%Y%m%d%H%M")))

    # ---------------------------------------
    #  train
    # ---------------------------------------
    # epochごとにデータを変えるように
    obj = GenerateStockPrice(NUM_TRAIN)
    obj_generator = obj.generate_stock_process()
    # adjust learning rate
    # see https://keras.io/ja/optimizers/
    # adam = Adam(lr=0.005)  # original
    adam = Adam(lr=0.001)  # original
    model_hedge.compile(optimizer=adam, loss='mean_squared_error')
    # model_hedge.compile(optimizer=adam, loss='mean_absolute_error')
    model_hedge.fit_generator(obj_generator, steps_per_epoch=1, epochs=EPOCH)

    # ---------------------------------------
    #  test
    # ---------------------------------------
    xtest, _ = generate_stock_process(NUM_TEST)

    # show hedge-payoff graph
    stock_price = np.exp(np.sum(xtest[2:], axis=0) + np.log(S0))
    payoff_ = np.maximum(stock_price - strike, 0) - priceBS
    output_ = model_hedge.predict(xtest)
    hedge_ = payoff_ - output_
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(stock_price, hedge_, s=1)
    ax.set_title(
        'deep hedge test (sample_num:{}, epoch:{})'.format(NUM_TEST, EPOCH))
    ax.set_xlabel('stock price')
    ax.set_ylabel('deep hedge pl')
    fig.savefig('result/test_plot_{}.png'.format(
        datetime.today().strftime("%Y%m%d%H%M")))
    fig.show()


if __name__ == '__main__':
    t = datetime.now()
    main()
    t = datetime.now() - t
    print(t)
