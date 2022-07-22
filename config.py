"""
input file, output file, model params and super params
"""
# config-> data_pre_process -> train
# origin files dir(software brat output .txt file and .ann file)
ORIGIN_DIR = './input/origin/'
# conll files
ANNOTATION_DIR = './output/annotation/'


# dataset files
TRAIN_PATH = './output/train.txt'
TEST_PATH = './output/test.txt'


# generate vocab file(tokens), label file from TRAIN_PATH
VOCAB_PATH = './output/vocab.txt'
LABEL_PATH = './output/label.txt'

# pad same size
WORD_PAD = '<PAD>'
WORD_PAD_ID = 0
# unknown vocabulary
WORD_UNK = '<UNK>'
WORD_UNK_ID = 1
# vocab list size
VOCAB_SIZE = 2309
# O id
LABEL_O_ID = 22

# max seg size
SEG_MAX_LEN = 50

# word embedding vector size
EMBEDDING_DIM = 100
HIDDEN_SIZE = 256
TARGET_SIZE = 55
LR = 1e-3
EPOCH = 100

MODEL_PATH = './output/model/'


