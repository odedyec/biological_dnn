from dataset_generator import *
# from model import *
import sys



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
    selex_4, cnt4  = selex_dataset_generator(SELEX_FILES[-1])
    print selex_4.shape
    print cnt4.shape


if __name__ == '__main__':
    # get_argv()
    PBM_FILE, SELEX_FILES = parse_args('train/TF1_pbm.txt', [0, 1, 2, 3, 4])
    main(PBM_FILE, SELEX_FILES)

