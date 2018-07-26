#!/usr/bin/env python3

from dataset_generator import *
from model import *
import sys
from result_analyzer import *

SELEX_SIZE = 36
TRAIN = True
GENERATE_DATASET = True
LOAD_ENTIRE_MODEL = False





def main(PBM_FILE, SELEX_FILES):
    x_train, x_test, y_train, y_test, pbm_data = generate_data(PBM_FILE, SELEX_FILES, GENERATE_DATASET, TRAIN_SIZE, SELEX_SIZE)

    x_train, y_train = suffle_data_label(x_train, y_train)
    x_test, y_test = suffle_data_label(x_test, y_test)
    print("Train size", x_train.shape, y_train.shape)
    print("Test size", x_test.shape, y_test.shape)
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
            model, history = train(model, x_train, y_train)
            save_network(model)
            plot_acc_loss(history)
        else:      # Load network from file
            model = load_model(model)
        print("===============================")
    visualize_model(model)
    predict_and_calculate_aupr(model, x_test, y_test)
    predict_on_pbm(model, pbm_data)


if __name__ == '__main__':
    PBM_FILE, SELEX_FILES = 'train/TF1_pbm.txt', [0, 1, 2, 3, 4]  # get_argv()
    print(PBM_FILE)
    print(SELEX_FILES)
    # PBM_FILE, SELEX_FILES =
    PBM_FILE, SELEX_FILES = parse_args(PBM_FILE, SELEX_FILES)
    print(PBM_FILE)
    print(SELEX_FILES)
    main(PBM_FILE, SELEX_FILES)

