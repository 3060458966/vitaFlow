'''
Prg
'''
import codecs
import os
import sys
from logging import error

from sklearn.model_selection import train_test_split

def split_data(in_directory, out_directory):
    import shutil
    data = os.listdir(in_directory)
    # Split the files into train test and val
    train_val, test = train_test_split(data, test_size = 0.2)
    train, val = train_test_split(train_val, test_size = 0.1)

    #move the files into train folder
    for i in train:
        shutil.copy(os.path.join(in_directory,i), os.path.join(out_directory,'train'))
    #move the files into val folder
    for i in val:
        shutil.copy(os.path.join(in_directory, i), os.path.join(out_directory, 'val'))
    #move the files into test folder
    for i in test:
        shutil.copy(os.path.join(in_directory, i), os.path.join(out_directory, 'test'))




def argparser():
    import argparse
    ap = argparse.ArgumentParser(description='Split the files into train test and val',
                                 usage='%(prog)s IN-DIRECTORY OUT-DIRECTORY')
    ap.add_argument('indirectory')
    ap.add_argument('outdirectory')
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])
    if not os.path.isdir(args.indirectory):
        error('Not a directory: {}'.format(args.indirectory))
        return 1
    if not os.path.isdir(args.outdirectory):
        error('Not a directory: {}'.format(args.outdirectory))
        return 1

    split_data(args.indirectory, args.outdirectory)

if __name__ == '__main__':
    sys.exit(main(sys.argv))