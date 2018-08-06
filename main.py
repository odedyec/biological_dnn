#!/usr/bin/env python3

from dataset_generator import *
from model import *
import sys
from result_analyzer import *


''' Global variables '''
SELEX_SIZE = 36
TRAIN = True
GENERATE_DATASET = True
TRAIN_SIZE = 30000
TEST_SIZE = 60000

''' model variable - global for looping purposes '''
model = None
model = build_model(SELEX_SIZE)
model.summary()


def main(PBM_FILE, SELEX_FILES):
    global model
    ''' Generate dataset '''
    x_train, x_test, y_train, y_test, pbm_data = generate_data(PBM_FILE, SELEX_FILES, GENERATE_DATASET, train_size=TRAIN_SIZE, SELEX_SIZE=SELEX_SIZE, test_size=TEST_SIZE)
    x_train, y_train = suffle_data_label(x_train, y_train)
    x_test, y_test = suffle_data_label(x_test, y_test)

    """ Setup model """
    if TRAIN:  # Train network
        model, history = train(model, x_train, y_train, debug=False)
        save_network(model)
        plot_acc_loss(history)
    else:      # Load network from file
        model = load_model(model)
    """ Predict selex classification """
    predict_and_calculate_aupr(model, x_test, y_test)
    """ Rank PBM """
    cnt, ap = predict_on_pbm(model, pbm_data)#, PBM_FILE)
    return cnt, ap


def loop_over_all():
    """
    This function can be the main loop that loads all of the data from within the python script
    To use this, uncomment in main
    :return:
    """
    import os.path
    import time
    wSave = model.get_weights()
    for i in range(0, 124):
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
    # import time
    # t = time.time()
    # PBM_FILE, SELEX_FILES = 'train/TF2_pbm.txt', [0, 1, 2, 3, 4]  #get_argv()  #
    # PBM_FILE, SELEX_FILES = parse_args(PBM_FILE, SELEX_FILES)
    # main(PBM_FILE, SELEX_FILES)
    # print("Took "+str(time.time() - t)+" seconds for program\n")

