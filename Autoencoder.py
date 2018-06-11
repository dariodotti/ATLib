import numpy as np
import cv2
import matplotlib as plt
import theanets as the
import utils


def load_AE_weights():
    AE_weights_level_1 = utils.load_matrix_pickle(
            'C:/Users/dario.dotti/Documents/data_for_vocabulary/ny_station/layer1_fromL3/18x18/144_l1_weight.txt')
    return AE_weights_level_1


def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


def tanh_function(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu_function(x):

    for i,row in enumerate(x):
        for j,value in enumerate(row):
            if value < 0:
                x[i,j] = 0
    return x


def encode_features_using_AE_layer1_cluster_activation(feature_matrix):


def training(my_data_tr, my_data_val):
    __model = the.Autoencoder(
        layers=(14400, (900, 'sigmoid'), (225, 'sigmoid'), ('tied', 900, 'sigmoid'), ('tied', 14400, 'sigmoid')),
        loss='my_cross_entropy')


    __model.train(np.array(my_data_tr), np.array(my_data_val), algo='adadelta',
                  hidden_l1=dict(weight=0.08, pattern='in:out'))

    print 'training done..'

    return True


def display_weights(weights):
    tied_weights = True
    channels =1
    k = min(len(weights), 9)
    imgs = np.eye(weights[0].shape[0])
    for i, weight in enumerate(weights[:-1]):
        imgs = np.dot(weight.T, imgs)
        plot_images(imgs,
                    100 + 10 * k + i + 1,
                    channels=channels,
                    title='Layer {}'.format(i+1))
    weight = weights[-1]
    n = weight.shape[1] / channels
    if int(np.sqrt(n)) ** 2 != n:
        return
    if tied_weights:
        imgs = np.dot(weight.T, imgs)
        plot_images(imgs,
                    100 + 10 * k + k,
                    channels=channels,
                    title='Layer {}'.format(k))
    else:
        plot_images(weight,
                    100 + 10 * k + k,
                    channels=channels,
                    title='Decoding weights')


def plot_images(imgs, loc, title=None, channels=1):

    n = int(np.sqrt(len(imgs)))
    assert n * n == len(imgs), 'images array must contain a square number of rows!'
    s = int(np.sqrt(len(imgs[0]) / channels))
    assert s * s == len(imgs[0]) / channels, 'images must be square!'

    img = np.zeros(((s+1) * n - 1, (s+1) * n - 1, channels), dtype=imgs[0].dtype)
    for i, pix in enumerate(imgs):
        r, c = divmod(i, n)
        img[r * (s+1):(r+1) * (s+1) - 1,
            c * (s+1):(c+1) * (s+1) - 1] = pix.reshape((s, s, channels))

    img -= img.min()
    img /= img.max()

    cv2.imshow('ciao',img)
    cv2.waitKey(0)

    ax = plt.gcf().add_subplot(loc)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    ax.imshow(img.squeeze(), cmap=plt.cm.gray)
    if title:
        ax.set_title(title)