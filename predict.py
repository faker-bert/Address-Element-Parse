import torch.cuda
from utils import *
from model import *
from config import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def extract(label, text):
    i = 0
    res = []
    while i < len(label):
        if label[i] != 'O':
            prefix, name = label[i].split('-')
            start = end = i
            i += 1
            while i < len(label) and (label[i] == 'I-' + name or label[i] == 'E-'+name):
                end = i
                i += 1
            res.append([name, text[start:end + 1]])
        else:
            i += 1
    return res


def predict(text):
    _, vocab2id = get_vocab_id()
    input = torch.tensor([[vocab2id.get(w, WORD_UNK_ID) for w in text]]).to(device)
    mask = torch.tensor([[1] * len(text)]).bool().to(device)

    model = torch.load(MODEL_PATH + 'model_11.pth').to(device)
    y_pred = model(input, mask)
    id2label, _ = get_label_id()

    label = [id2label[l] for l in y_pred[0]]
    print(text)
    print(label)

    info = extract(label, text)
    print(info)
