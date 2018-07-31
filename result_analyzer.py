import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from model import *
import os
import time


def my_pbm_aupr(result):
    cnt = (result < 100).astype(np.int)
    prec = np.zeros((100, 1), dtype=np.float)
    recall = np.zeros((100, 1), dtype=np.float)
    ap = 0
    for i in range(100):
        prec[i, 0] = np.sum(cnt[0:i+1]) / (i+1)
        recall[i, 0] = np.sum(cnt[0:i+1]) / 100
        if i == 0: continue
        ap += (recall[i, 0] - recall[i-1, 0]) * prec[i, 0]
    print('PBM average: '+str(ap))
    fig1 = plt.figure(101)
    plt.cla()
    plt.plot(recall, prec)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall graph\nAUPR = '+str(ap)+'    '+str(np.sum(cnt[0:100]))+'/100')
    # plt.show()
    fig1.savefig('aupr_graphs/'+str(time.time())+'.png')
    plt.cla()
    return cnt, ap

def predict_on_pbm(model, pbm_dat):
    pbm_dat = pbm_dat.reshape((len(pbm_dat), 60, 4, 1))
    print (pbm_dat.shape)
    how_much = len(pbm_dat)
    # res = np.zeros((how_much, 1))
    res = predict(model, pbm_dat[:, 0:36, :, :])
    # for i in xrange(16):
    #     p = predict(model, pbm_dat[0:how_much, (i+0):(i+20), :, :])
    #     res = res + (p[:, 1]).reshape(how_much, 1)
    np.savetxt('pbm.csv', res, fmt='%.3f', newline=os.linesep)
    idx = np.arange(how_much).reshape(how_much, 1)
    # res2 = np.concatenate((idx, res), axis=1)
    res2 = np.argsort(res[:, 0])
    cnt, ap = my_pbm_aupr(res2)
    np.savetxt('pbm2.csv', res2, fmt='%.3f', newline=os.linesep)
    num_correct = np.sum(res2[0:100] < 100)
    return cnt, ap


def predict_and_calculate_aupr(model, x_test, y_test):
    p1 = predict(model, x_test)

    np.savetxt('y_score.csv', p1, fmt='%.3f', newline=os.linesep)
    np.savetxt('y_testsss.csv', y_test, fmt='%.3f', newline=os.linesep)
    # y_score = np.zeros(y_test.shape)
    # for i in xrange(len(idx)):
    #     y_score[i, idx[i]] = 1

    # average_precision = average_precision_score(y_test, y_score)

    # print('Average precision-recall score: {0:0.2f}'.format(
    #     average_precision))


    # precision, recall, _ = precision_recall_curve(y_test, y_score)
    # fig = plt.figure(3)
    # plt.step(recall, precision, color='b', alpha=0.2,
    #          where='post')
    # plt.fill_between(recall, precision, step='post', alpha=0.2,
    #                  color='b')
    #
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
    #     average_precision))
    # fig.savefig('aupr_selex.png')
    # plt.show()


def plot_acc_loss(history):
    # summarize history for accuracy
    fig1 = plt.figure(1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    fig1.savefig('accuracy.png')
    # plt.show()
    # summarize history for loss
    fig2 = plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    fig2.savefig('loss.png')
    # plt.show()