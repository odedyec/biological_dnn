import numpy as np
from sklearn.metrics import average_precision_score
from model import *
import os


def predict_on_pbm(model, pbm_dat):
    pbm_dat = pbm_dat.reshape((len(pbm_dat), 60, 4, 1))
    print pbm_dat.shape
    how_much = len(pbm_dat)
    # res = np.zeros((how_much, 1))
    res = predict(model, pbm_dat[:, 0:36, :, :])
    # for i in xrange(16):
    #     p = predict(model, pbm_dat[0:how_much, (i+0):(i+20), :, :])
    #     res = res + (p[:, 1]).reshape(how_much, 1)
    np.savetxt('pbm.csv', res, fmt='%.3f', newline=os.linesep)
    idx = np.arange(how_much).reshape(how_much, 1)
    res2 = np.concatenate((idx, res), axis=1)
    res2 = np.sort(res2, axis=-1)
    np.savetxt('pbm2.csv', res2, fmt='%.3f', newline=os.linesep)
    print res2




def predict_and_calculate_aupr(model, x_test, y_test):
    p1 = predict(model, x_test[0:10000, :, :, :])
    p2 = predict(model, x_test[-10000:-1, :, :, :])
    y_score = np.concatenate((p1[:, 1] - p1[:, 0], p2[:, 1] - p2[:, 0]))
    y_test = np.concatenate((np.argmax(y_test[0:10000, :], axis=1), np.argmax(y_test[-10000:-1, :], axis=1)))

    # y_score = np.zeros(y_test.shape)
    # for i in xrange(len(idx)):
    #     y_score[i, idx[i]] = 1

    average_precision = average_precision_score(y_test, y_score)

    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_test, y_score)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))
    plt.show()