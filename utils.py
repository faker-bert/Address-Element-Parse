import torch
from torch.utils import data
import config
import pandas as pd


def get_vocab_id():
    vocab = pd.read_csv(config.VOCAB_PATH, names=['word', 'id'])
    return list(vocab['word']), dict(vocab.values)


def get_label_id():
    label = pd.read_csv(config.LABEL_PATH, names=['label', 'id'])
    return list(label['label']), dict(label.values)


class Dataset(data.Dataset):
    def __init__(self, train: bool = True, base_len=50):
        super(Dataset, self).__init__()

        self.base_len = base_len
        self.path = config.TRAIN_PATH if train else config.TEST_PATH

        self.data = pd.read_csv(self.path, sep=' ', names=['word', 'label'])

        _, self.vocab2id = get_vocab_id()
        _, self.label2id = get_label_id()

        self.cut_point = [0]
        self.count = 0
        self.get_address_point_size()

    def __getitem__(self, index):
        address = self.data.iloc[self.cut_point[index]:self.cut_point[index + 1], :]
        address_record = [self.vocab2id.get(word, config.WORD_UNK_ID) for word in address['word']]
        target = [self.label2id.get(label, self.label2id['O']) for label in address['label']]
        return address_record, target

    def __len__(self):
        return self.count

    def get_address_point_size(self):
        with open(self.path, encoding='utf-8') as file:
            file_iter = file.readlines()

        for index, word in enumerate(file_iter):
            if word == '\n':
                self.cut_point.append(index-self.count)
                self.count += 1


def collate_fn(batch):
    """
    padding every address and target for keeping same size and mask for crf
    :param batch: [batch, record(address_enc, target_enc)]
    :return: (address_padding_enc, target_padding_enc, mask_enc)
    """
    address_es = []
    target_s = []
    mask = []

    batch.sort(key=lambda x: len(x[0]), reverse=True)
    max_len = len(batch[0][0])

    for item in batch:
        pad_len = max_len - len(item[0])

        address_es.append(item[0] + [config.WORD_PAD_ID] * pad_len)
        target_s.append(item[1] + [config.LABEL_O_ID] * pad_len)
        # for crf calc
        mask.append([1] * len(item[0]) + [0] * pad_len)
    return torch.tensor(address_es), torch.tensor(target_s), torch.tensor(mask).bool()


if __name__ == '__main__':
    for addresses, targets, masks in data.DataLoader(Dataset(), batch_size=10, shuffle=True, collate_fn=collate_fn):
        print(addresses.shape)
        print(targets.shape)
        print(masks.shape)
    print(get_vocab_id())

