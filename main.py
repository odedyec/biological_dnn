from dataset_generator import *
from model import *

import sys



if len(sys.argv) < 4:
    print "Length of input arguments is ", len(sys.argv)
    print "Usage:\n python main.py train/predict pbm_file SELEX_FILE_0 SELEX_FILE_1 ..."
    sys.exit(0)


TRAIN = True if "train" in sys.argv[1] else False
PREDICT = True if "predict" in sys.argv[1] else False

if not (TRAIN or PREDICT):
    print sys.argv
    print "Usage:\n python main.py train/predict pbm_file SELEX_FILE_0 SELEX_FILE_1 ..."
    sys.exit(0)

PBM_FILE = sys.argv[2]
SELEX_FILES = [sys.argv[i] for i in range(3, len(sys.argv))]

if TRAIN:
    print "Training on:"
    print PBM_FILE
    print "Using: "
    print SELEX_FILES

elif PREDICT:
    print "Predicting on:"
    print PBM_FILE
    print "Using: "
    print SELEX_FILES




