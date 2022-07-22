# read format file
from glob import glob
import os
import random
import pandas as pd
import config


def get_annotation(ann_path):
    ...


def generate_vocab():
    train_data_vocab = pd.read_csv(config.TRAIN_PATH, header=None, sep=' ').iloc[:, 0]
    train_data_vocab = [config.WORD_PAD, config.WORD_UNK] + train_data_vocab.value_counts().keys().tolist()
    train_data_vocab = train_data_vocab[:config.VOCAB_SIZE]
    print(min(len(train_data_vocab), config.VOCAB_SIZE))
    # vocab = dict(zip(train_data_vocab, range(len(train_data_vocab))))
    vocab = {v: index for index, v in enumerate(train_data_vocab)}
    vocab = pd.DataFrame(list(vocab.items()))
    vocab.to_csv(config.VOCAB_PATH, header=None, index=None)


def generate_label():
    train_data = pd.read_csv(config.TRAIN_PATH, header=None, sep=' ').iloc[:, 1]
    train_data = train_data.value_counts().keys().tolist()
    label_map = {label: index for index, label in enumerate(train_data)}
    label_df = pd.DataFrame(label_map.items())
    label_df.to_csv(config.LABEL_PATH, header=None, index=None)


if __name__ == '__main__':
    generate_vocab()
    generate_label()
    print('file generate ok next file train.py')
