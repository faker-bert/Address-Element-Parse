from torch import nn
import config
from torchcrf import CRF


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embed = nn.Embedding(config.VOCAB_SIZE, config.EMBEDDING_DIM, config.WORD_PAD_ID)
        self.lstm = nn.LSTM(
            config.EMBEDDING_DIM,
            config.HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True
        )
        # map to label   2*config.HIDDEN_SIZE due to bi-lstm
        self.linear = nn.Linear(2*config.HIDDEN_SIZE, config.TARGET_SIZE)

        self.crf = CRF(config.TARGET_SIZE, batch_first=True)

    def _get_bi_lstm_feature(self, x):
        """

        :param x: [batch, seq_len]
        :return: bi-lstm output [batch, seq_len, label_prop]
        """
        out = self.embed(x)
        out, _ = self.lstm(out)
        return self.linear(out)

    def forward(self, x, mask):
        """

        :param x: [batch, seq_len]
        :param mask: [batch, seq_len] for crf judge
        :return:
        """
        out = self._get_bi_lstm_feature(x)
        return self.crf.decode(out, mask)

    def loss_fn(self, x, target, mask):
        y_pred = self._get_bi_lstm_feature(x)
        return -self.crf(y_pred, target, mask, reduction='mean')


