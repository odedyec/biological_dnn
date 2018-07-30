import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from model import *
import os


def my_pbm_aupr():
    pass

def predict_on_pbm(model, pbm_dat):
    print ("pbm_dat shape",pbm_dat.shape)
    pbm_dat = pbm_dat.reshape((len(pbm_dat), 60, 4, 1))
    print ("pbm_dat re-shaped",pbm_dat.shape)
    how_much = len(pbm_dat)
    # res = np.zeros((how_much, 1))
    res = predict(model, pbm_dat[:, 0:35, :, :])
    # res = predict(model, pbm_dat[:, 0:36, :, :])
    # for i in xrange(16):
    #     p = predict(model, pbm_dat[0:how_much, (i+0):(i+20), :, :])
    #     res = res + (p[:, 1]).reshape(how_much, 1)
    np.savetxt('pbm.csv', res, fmt='%.3f', newline=os.linesep)
    idx = np.arange(how_much).reshape(how_much, 1)
    # res2 = np.concatenate((idx, res), axis=1)
    res2 = np.argsort(res[:, 0])
    np.savetxt('pbm2.csv', res2, fmt='%.3f', newline=os.linesep)
    num_correct = np.sum(res2[0:100] < 100)
    f = open('result.txt', 'a')
    f.write(str(num_correct)+'\n')
    f.close()
    print (num_correct)




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