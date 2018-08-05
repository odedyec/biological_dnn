#!/usr/bin/env python3

from dataset_generator import *
from model import *
import sys
from result_analyzer import *

SELEX_SIZE = 36
TRAIN = True
GENERATE_DATASET = True
LOAD_ENTIRE_MODEL = False

model = None
model = build_model(SELEX_SIZE)
model.summary()


def main(PBM_FILE, SELEX_FILES):
    global model
    x_train, x_test, y_train, y_test, pbm_data = generate_data(PBM_FILE, SELEX_FILES, GENERATE_DATASET, train_size=TRAIN_SIZE, SELEX_SIZE=SELEX_SIZE, test_size=TEST_SIZE)
    x_train, y_train = suffle_data_label(x_train, y_train)
    x_test, y_test = suffle_data_label(x_test, y_test)
    print("Train size", x_train.shape, y_train.shape)
    print("Test size", x_test.shape, y_test.shape)
    np.savetxt('x_test.csv', x_test[0, :, :, 0], fmt='%.3f', newline=os.linesep)
    np.savetxt('y_test.csv', y_test, fmt='%.3f', newline=os.linesep)
    np.savetxt('x_train.csv', x_train[0, :, :, 0], fmt='%.3f', newline=os.linesep)
    np.savetxt('y_train.csv', y_train, fmt='%.3f', newline=os.linesep)
    """ Setup model """

    if LOAD_ENTIRE_MODEL:
        model = load_entire_model()
        model.summary()
    else:

        if TRAIN:  # Train network

            model, history = train(model, x_train, y_train)#, debug=False)
            save_network(model)
            # plot_acc_loss(history)
        else:      # Load network from file
            model = load_model(model)
        print("===============================")
    # visualize_model(model)
    predict_and_calculate_aupr(model, x_test, y_test)
    cnt, ap = predict_on_pbm(model, pbm_data)
    return cnt, ap


def loop_over_all():
    import os.path
    import time
    wSave = model.get_weights()
    for i in range(1, 124):
        pbm_file_name = 'train/TF%d_pbm.txt'%(i)
        selex_list = []
        for j in range(7):
            selex_file_name = 'train/TF%d_selex_%d.txt'%(i,j)
            if not os.path.isfile(selex_file_name):
                break
            selex_list.append(selex_file_name)
        t = time.time()
        model.set_weights(wSave)
        try:
            cnt, ap = main(pbm_file_name, selex_list)
        except:
            print("Fail at TF{}".format(i))
            cnt = [0] * 100
            ap = 0
        f = open('result.csv', 'a')
        f.write(str(time.time() - t) + ',' + str(sum(cnt[0:100])) + ',' + str(ap) + '\n')
        f.close()



if __name__ == '__main__':
    loop_over_all()
    # PBM_FILE, SELEX_FILES = 'train/TF1_pbm.txt', [0, 1, 2, 3, 4]  # get_argv()
    # PBM_FILE, SELEX_FILES = parse_args(PBM_FILE, SELEX_FILES)
    # main(PBM_FILE, SELEX_FILES)

