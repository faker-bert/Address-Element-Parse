import torch
from torch.utils import data
from model import Model
import config
from utils import Dataset, collate_fn


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Dataset()
    dataloader = data.DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

    model = Model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    for epoch in range(config.EPOCH):
        for index, (X, target, mask) in enumerate(dataloader):
            X, target, mask = X.to(device), target.to(device), mask.to(device)

            y_pred = model(X, mask)
            loss = model.loss_fn(X, target, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if index % 10 == 0:
                # 100 batch
                print(f'>>epoch: {epoch} | batch num: {index} | loss: {loss.item()}')
        with open('loss_record.txt', mode='a') as file:
            file.write(f'epoch:{epoch}...loss:{loss.item()}...\n')
        torch.save(model, config.MODEL_PATH + f'model_{epoch}.pth')
        print('save model in local...')




