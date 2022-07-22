import torch
from torch.utils import data
import config
from utils import Dataset, collate_fn

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(train=False)
    dataloader = data.DataLoader(dataset,
                                 batch_size=64,
                                 collate_fn=collate_fn)
    with torch.no_grad():
        model = torch.load(config.MODEL_PATH + 'model_11.pth').to(device)
        # real labels
        y_true_list = []
        # pred_labels
        y_pred_list = []

        for index, (X, target, mask) in enumerate(dataloader):
            # to gpu
            X, target, mask = X.to(device), target.to(device), mask.to(device)
            # pred X
            y_pred = model(X, mask)
            loss = model.loss_fn(X, target, mask)

            print(f'>> batch loss: {loss.item()}')
            for lst in y_pred:
                y_pred_list += lst
            for y, m in zip(target, mask):
                y_true_list += y[m == True].tolist()
        y_true_tensor = torch.tensor(y_true_list)
        y_pred_tensor = torch.tensor(y_pred_list)
        accuracy = (y_true_tensor == y_pred_tensor).sum()/len(y_true_tensor)

        print(f'>> total {len(y_true_tensor)}, accuracy:{accuracy}')
