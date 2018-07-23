#!/usr/bin/env python

from dataset_generator import *
from model import *
import sys
from result_analyzer import *

TRAIN_SIZE = 100000
SELEX_SIZE = 36
TRAIN = True
GENERATE_DATASET = True
LOAD_ENTIRE_MODEL = False



def get_argv():
    """
    Get input from sys.argv
    :return:
    """
    if len(sys.argv) < 3:
        print "Length of input arguments is ", len(sys.argv)
        print "\nUsage:\n python main.py pbm_file SELEX_FILE_0 SELEX_FILE_1 ..."
        print "\nUsage2:\n python main.py pbm_file #of_selex_0 #of_selex_1 ..."
        sys.exit(0)

    PBM_FILE = sys.argv[1]
    SELEX_FILES = [sys.argv[i] for i in range(2, len(sys.argv))]
    if SELEX_FILES[0].isdigit():
        SELEX_FILES = map(int, SELEX_FILES)
    return PBM_FILE, SELEX_FILES


def parse_args(PBM_FILE, SELEX_FILES):
    """
    Transform selex numbers to filenames.
    :param PBM_FILE: Required for full path
    :param SELEX_FILES: List of numbers of SELEX cycles, or filenames
    :return: Filenames of everything
    """
    if len(SELEX_FILES) < 1:
        parse_args()
    if type(SELEX_FILES[0]) == int:
        base = PBM_FILE.split('_')[0]
        selex = [base+'_selex_'+str(i)+'.txt' for i in SELEX_FILES]
    return PBM_FILE, selex


def main(PBM_FILE, SELEX_FILES):

    pbm_data = pbm_dataset_generator(PBM_FILE)
    if GENERATE_DATASET:  # load data and OneHot encode data
        print pbm_data.shape
        selex_4, _ = selex_dataset_generator(SELEX_FILES[-1])
        selex_0, _ = selex_dataset_generator(SELEX_FILES[0])

        selex_4 = selex_4.reshape((len(selex_4), SELEX_SIZE, 4, 1))
        selex_0 = selex_0.reshape((len(selex_0), SELEX_SIZE, 4, 1))

        x_train, x_test, y_train, y_test = split_train_test(selex_0, selex_4, TRAIN_SIZE)
        save_dataset(x_train, x_test, y_train, y_test)
    else:  # Load from data_tf1.hdf5 file
        x_train, x_test, y_train, y_test = load_dataset()
    x_train, y_train = suffle_data_label(x_train, y_train)
    x_test, y_test = suffle_data_label(x_test, y_test)
    print "Train size", x_train.shape, y_train.shape
    print "Test size", x_test.shape, y_test.shape
    # np.savetxt('x_test.csv', x_test, fmt='%.3f', newline=os.linesep)
    np.savetxt('y_test.csv', y_test, fmt='%.3f', newline=os.linesep)
    # np.savetxt('x_train.csv', x_train, fmt='%.3f', newline=os.linesep)
    np.savetxt('y_train.csv', y_train, fmt='%.3f', newline=os.linesep)
    """ Setup model """
    model = None
    if LOAD_ENTIRE_MODEL:
        model = load_entire_model()
        model.summary()
    else:
        model = build_model(SELEX_SIZE)
        model.summary()
        if TRAIN:  # Train network
            model = train(model, x_train, y_train)
            save_network(model)
        else:      # Load network from file
            model = load_model(model)
        print "==============================="

    # predict_and_calculate_aupr(model, x_test, y_test)
    predict_on_pbm(model, pbm_data)


if __name__ == '__main__':
    PBM_FILE, SELEX_FILES = get_argv()
    print PBM_FILE
    print SELEX_FILES
    # PBM_FILE, SELEX_FILES = parse_args('train/TF1_pbm.txt', [0, 1, 2, 3, 4])
    main(PBM_FILE, SELEX_FILES)

