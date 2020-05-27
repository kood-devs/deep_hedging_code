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
from keras.layers import Input, Dense, Subtract, Multiply, Lambda, Add, BatchNormalization, Activation
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
d = 2  # number of layers in strategy
n = 32  # nodes in the layers (common in all intermediate layers)

# Learning params
NUM_TRAIN = 10**5
NUM_TEST = 10**5
EPOCH = 500
BATCH_SIZE = 256


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
    for j in range(N):
        for i in range(d):
            if i < d-1:
                nodes = n
                layer = Dense(nodes, activation='relu', trainable=True,
                              kernel_initializer=initializers.RandomNormal(
                                  0, 1),
                              bias_initializer='random_normal',
                              name=str(i)+str(j))
            else:
                nodes = m
                layer = Dense(nodes, activation='linear', trainable=True,
                              kernel_initializer=initializers.RandomNormal(
                                  0, 1),
                              bias_initializer='random_normal',
                              name=str(i)+str(j))
            layers.append(layer)

    # Implementing the loss function
    # Initial input is spot and pnl
    price = Input(shape=(m,))  # initial spot
    hedge = Input(shape=(m,))  # total profit and loss (init 0)
    inputs = [price]+[hedge]

    for j in range(N):
        strategy = price
        for k in range(d):
            strategy = layers[k+j*d](strategy)
            # batch normalization
            # this doesn't work...
            # if k != d - 1:
            #     strategy = layers[k+j*d](strategy)
            #     strategy = BatchNormalization()(strategy)
            #     strategy = Activation('relu')(strategy)
            # else:
            #     strategy = layers[k+j*d](strategy)
            #     strategy = BatchNormalization()(strategy)
            #     strategy = Activation('linear')(strategy)
        incr = Input(shape=(m,))
        logprice = Lambda(lambda x: K.log(x))(price)  # s_t -> log(s_t)
        logprice = Add()([logprice, incr])  # log(s_t) -> log(s_t+1)
        pricenew = Lambda(lambda x: K.exp(x))(logprice)  # log(s_t+1) -> s_t+1
        priceincr = Subtract()([pricenew, price])
        hedgenew = Multiply()([strategy, priceincr])
        hedge = Add()([hedge, hedgenew])
        inputs = inputs + [incr]
        price = pricenew

    priceBS = calculate_BS_price(S0, strike, T, sigma)
    payoff = Lambda(lambda x: 0.5*(K.abs(x-strike)+x-strike) - priceBS)(price)
    outputs = Subtract()([payoff, hedge])  # payoff minus priceBS minus hedge

    # inputs are 1.initial price, 2.hedging strategy(init 0), 3.increments of the log price process
    model_hedge = Model(inputs=inputs, outputs=outputs)
    # model_hedge.summary()
    plot_model(model_hedge, to_file='model_hedge_{}.png'.format(
        datetime.today().strftime("%Y%m%d%H%M")))

    # ---------------------------------------
    #  train
    # ---------------------------------------
    # xtrain consists of the price S0,
    # the initial hedging being 0, and the increments of the log price process
    xtrain, ytrain = generate_stock_process(NUM_TRAIN)

    # adjust learning rate
    # see https://keras.io/ja/optimizers/
    adam = Adam(lr=0.005)
    model_hedge.compile(optimizer=adam, loss='mean_squared_error')
    model_hedge.fit(x=xtrain, y=ytrain, batch_size=BATCH_SIZE,
                    epochs=EPOCH, verbose=True)
    model_hedge.save('model_{}.h5'.format(
        datetime.today().strftime("%Y%m%d%H%M")))
    model_hedge = load_model('model_202005270052.h5', compile=True)

    # show error graph
    fig = plt.figure()
    train_result = model_hedge.predict(xtrain)
    plt.hist(train_result, bins=50)
    train_mean = np.mean(train_result)
    train_std = np.std(train_result)
    plt.title('err mean:{}, err std:{}'.format(train_mean, train_std))
    fig.savefig('train_{}.png'.format(datetime.today().strftime("%Y%m%d%H%M")))
    # print('train mean\t: {}'.format(train_mean))
    # print('train std\t: {}'.format(train_std))

    # show hedge-payoff graph
    stock_price = np.exp(np.sum(xtrain[2:], axis=0) + np.log(S0))
    payoff_ = np.maximum(stock_price - strike, 0) - priceBS
    output_ = model_hedge.predict(xtrain)
    hedge_ = payoff_ - output_
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(stock_price, hedge_, s=1)
    ax.set_title(
        'deep hedge train (sample_num:{}, epoch:{})'.format(NUM_TRAIN, EPOCH))
    ax.set_xlabel('stock price')
    ax.set_ylabel('deep hedge pl')
    fig.savefig('train_plot_{}.png'.format(
        datetime.today().strftime("%Y%m%d%H%M")))
    fig.show()

    # ---------------------------------------
    #  test
    # ---------------------------------------
    xtest, _ = generate_stock_process(NUM_TEST)

    # show error graph
    fig = plt.figure()
    test_result = model_hedge.predict(xtest)
    plt.hist(test_result, bins=50)
    test_mean = np.mean(test_result)
    test_std = np.std(test_result)
    plt.title('err mean:{}, err std:{}'.format(test_mean, train_std))
    fig.savefig('test_{}.png'.format(datetime.today().strftime("%Y%m%d%H%M")))

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
    fig.savefig('test_plot_{}.png'.format(
        datetime.today().strftime("%Y%m%d%H%M")))
    fig.show()


if __name__ == '__main__':
    main()
