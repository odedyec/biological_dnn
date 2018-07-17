from dataset_generator import *
from model import *
import sys
import numpy as np
from sklearn.metrics import average_precision_score


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
    print pbm_data.shape
    selex_4, _ = selex_dataset_generator(SELEX_FILES[-1])
    selex_0, _ = selex_dataset_generator(SELEX_FILES[0])

    selex_4 = selex_4.reshape((len(selex_4), 20, 4, 1))
    selex_0 = selex_0.reshape((len(selex_0), 20, 4, 1))

    x_train, x_test, y_train, y_test = split_train_test(selex_0, selex_4, 10000)
    print "Train size", x_train.shape, y_train.shape
    print "Test size", x_test.shape, y_test.shape
    """ Setup model """
    model = build_model()
    model.summary()
    model = train(model, x_train, y_train)
    save_network(model)

    # model = load_model(model)
    print "==============================="
    p1 = np.argmax(predict(model, x_test), axis=1)
    print p1



if __name__ == '__main__':
    # get_argv()
    PBM_FILE, SELEX_FILES = parse_args('train/TF1_pbm.txt', [0, 1, 2, 3, 4])
    main(PBM_FILE, SELEX_FILES)




    # from sklearn.metrics import average_precision_score
    # predict=model.predict(np.array(test))
    # true=[int(x) for x in np.append(np.ones(100), np.zeros(len(test)-100), axis=0)]
    # print(average_precision_score(true, predict))
