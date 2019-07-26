import numpy as np
import pickle
import sys

GLOVE_PATH = '/project/glove/'
OUTPUT_PATH = '/project/dissertation/output/'
GLOVE_6B_50D_PATH = GLOVE_PATH + 'glove.6B.50d.txt'
GLOVE_840B_300D_PATH = GLOVE_PATH + 'glove.840B.300d.txt'
encoding = "utf-8"


def load_glove(glove_file, output_file):
    """
    Load and pickle glove file
    Sample command line run:
        python utils/load_glove.py [glove_file] [output_file]
    Input file is downloaded from :https://github.com/stanfordnlp/GloVe
    :param glove_file: GloVe file path (ex. glove.6B.50d.txt)
    :param output_file: Pickled dictionary file
    :return:
    """
    print('Exporting ' + glove_file)
    glove_dict = {}
    # all_words = set(w for words in X for w in words)
    with open(glove_file, "rb") as infile:
        for line in infile:
            parts = line.split()
            word = parts[0].decode(encoding)
            #         if (word in all_words):
            nums = np.array(parts[1:], dtype=np.float32)
            glove_dict[word] = nums

    print('Found ', len(glove_dict), ' words.')
    with open(output_file, 'wb') as output:
        pickle.dump(glove_dict, output)

    print('Export ', output_file, ' done!')


if __name__ == '__main__':
    # Map command line arguments to function arguments.
    load_glove(*sys.argv[1:])

# load_glove(GLOVE_6B_50D_PATH, OUTPUT_PATH+'GLOVE_6B_50D.p')
# load_glove(GLOVE_840B_300D_PATH, OUTPUT_PATH+'GLOVE_840B_300D.p')

