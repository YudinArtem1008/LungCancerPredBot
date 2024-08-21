# Каким-то макаром надо сделать так, чтобы был доступ к классу модели
# upd: Пофиксил это таким образом

import torch
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
from PIL import Image
import numpy as np
from collections import Counter, OrderedDict
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.nn import Linear, ReLU, Softmax, MaxPool2d, Flatten, Sequential
from IPython.display import clear_output
print(torch.__version__)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class LungCancerDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        image = self.x[index]
        if self.transform is not None and type(image) != torch.Tensor:
            image = self.transform(image)
        return image, self.y[index]

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_transform(self):
        return self.transform


class ResidualBlock(torch.nn.Module):
    def __init__(
            self,
            input_c,
            output_c,
            kernel_size,
            activation=torch.nn.ReLU,
    ):
        super().__init__()

        self.activation = activation()

        padding_size = (kernel_size - 1) // 2

        self.conv = torch.nn.Conv2d(
            in_channels=input_c,
            out_channels=output_c,
            kernel_size=kernel_size,
            padding=padding_size,
            padding_mode="zeros",
        )

        if input_c != output_c:
            self.correct_channels = torch.nn.Conv2d(
                in_channels=input_c,
                out_channels=output_c,
                kernel_size=1,
            )
        else:
            self.correct_channels = torch.nn.Identity()

    def forward(self, x):
        return self.activation(self.conv(x) + self.correct_channels(x))


class LinearBlock(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation=ReLU,
    ):
        super().__init__()

        if activation == Softmax:
            self.activation = activation(dim=-1)
        else:
            self.activation = activation()

        self.lin = Linear(
            in_features=in_features,
            out_features=out_features,
        )

    def forward(self, x):
        return self.activation(self.lin(x))


conv_layers = [
    ("Residual_block_1", ResidualBlock(
        input_c=3,
        output_c=10,
        kernel_size=5
    )),
    ("MaxPool2D_1", MaxPool2d(
        kernel_size=2
    )),
    ("Residual_block_2", ResidualBlock(
        input_c=10,
        output_c=16,
        kernel_size=5
    )),
    ("MaxPool2D_2", MaxPool2d(
        kernel_size=2
    )),
    ("Residual_block_3", ResidualBlock(
        input_c=16,
        output_c=32,
        kernel_size=5
    )),
    ("MaxPool2D_3", MaxPool2d(
        kernel_size=2
    )),
    ("Residual_block_4", ResidualBlock(
        input_c=32,
        output_c=64,
        kernel_size=5
    )),
    ("MaxPool2D_4", MaxPool2d(
        kernel_size=2
    )),
    ("Residual_block_5", ResidualBlock(
        input_c=64,
        output_c=128,
        kernel_size=5
    ))
]

linear_layers = [
    ("LinearBlock_1", LinearBlock(
        in_features=32768,
        out_features=100,
    )),
    ("LinearBlock_2", LinearBlock(
        in_features=100,
        out_features=40,
    )),
    ("LinearBlock_3", LinearBlock(
        in_features=40,
        out_features=3,
        activation=Softmax,
    )),
]

layers = conv_layers + [("Flatten", Flatten())] + linear_layers
model = Sequential(OrderedDict(layers))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = model

    def forward(self, x):
        return self.model(x).log()

    def __str__(self):
        return self.model


def train_val_dataset(dataset, val_split=0.25):
    X, y, transform = dataset.get_x(), dataset.get_y(), dataset.get_transform()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_split)
    datasets = {}
    datasets['train'] = LungCancerDataset(X_train, y_train, transform=transform)
    datasets['val'] = LungCancerDataset(X_test, y_test, transform=transform)
    return datasets


if __name__ == '__main__':
    directory = r'.\dataset\The IQ-OTHNCCD lung cancer dataset\The IQ-OTHNCCD lung cancer dataset'

    categories = ['Bengin cases', 'Malignant cases', 'Normal cases']

    data = []

    for i in categories:
        path = os.path.join(directory, i)
        class_num = categories.index(i)
        for file in os.listdir(path)[1::]:
            filepath = os.path.join(path, file)
            img = np.asarray(Image.open(filepath).resize((256, 256)))
            a = [torch.tensor(img).permute(2, 0, 1), class_num]
            data.append(a)

    random.shuffle(data)

    X, y = [], []
    for feature, label in data:
        X.append(feature)
        y.append(label)

    print('X length:', len(X), len(X[0]), len(X[0][0]), len(X[0][0][0]))
    print('y counts:', Counter(y))
    X = torch.stack((X))

    y = torch.tensor(y)
    X = X / 255.0
    print(X.shape)

    transform = transforms.Compose(
        [transforms.Resize(255),
         transforms.CenterCrop(224),
         transforms.ToTensor()])

    dataset = LungCancerDataset(X, y, transform=transform)
    train_and_test_datasets = train_val_dataset(dataset)

    train_dataset = train_and_test_datasets['train']
    print(train_dataset)
    val_dataset = train_and_test_datasets['val']
    print(val_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=32,
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=32,
                                shuffle=False)

    # print the total no of samples
    # print(train_dataset)
    print('Number of samples of train_dataloader: ', len(train_dataloader))
    print('Number of samples of val_dataloader: ', len(val_dataloader))
    images, labels = next(iter(train_dataloader))
    image = images[2][0]  # load 3rd sample
    # print(image)

    # visualize the image
    plt.imshow(image)

    # print the size of image
    print("Image Size: ", image.size())

    # print the label
    print(labels)

    X_train, y_train = train_dataset.get_x(), train_dataset.get_y()
    X_val, y_val = val_dataset.get_x(), val_dataset.get_y()

    print(X_train.shape)
    print(X_val.shape)

    model = Model().to(dtype=X_train.dtype, device=device)

    num_epochs = 100
    val_every = 1

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-2,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    loss_fn = torch.nn.NLLLoss()

    losses = {"train": [], "val": []}

    for epoch in range(1, num_epochs + 1):
        # В одной эпохе обучения мы проходим по всем объектам из обучающей выборки — классический вариант.
        local_losses = []
        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()

            pred = model(x_batch.to(device))
            loss = loss_fn(pred, y_batch.to(device))

            loss.backward()
            optimizer.step()
            local_losses.append(loss.item())

        losses["train"].append(sum(local_losses) / len(local_losses))

        # Каждые val_every итераций считаем значение loss на валидации.
        if epoch % val_every == 0:
            with torch.no_grad():
                local_losses = []
                for x_batch, y_batch in val_dataloader:
                    val_pred = model(x_batch.to(device))
                    val_loss = loss_fn(val_pred, y_batch.to(device))
                    local_losses.append(val_loss.item())

                losses["val"].append(sum(local_losses) / len(local_losses))

        # Каждые k итераций уменьшаем шаг градиентного спуска.
        if epoch % 30 == 0:
            scheduler.step()

        # Каждые 10 итераций рисуем графики loss.
        if epoch % 1 == 0:
            clear_output(True)
            fig, ax = plt.subplots(figsize=(30, 10))
            plt.title("График ошибки")
            plt.plot(losses["train"], ".-", label="Ошибка на обучении")
            plt.plot(torch.arange(0, epoch, val_every), losses["val"], ".-", label="Ошибка на валидации")
            plt.xlabel("Итерация обучения")
            plt.ylabel("Значение ошибки")
            plt.legend()
            plt.grid()
            plt.show()

    with torch.no_grad():
        accuracy = []
        for x_batch, y_batch in val_dataloader:
            preds = torch.nn.functional.softmax(model(x_batch.to(device)), -1).max(1).indices.cpu().detach()
            accuracy.extend((preds == y_batch).float().tolist())
        accuracy = sum(accuracy) / len(accuracy)
        print(accuracy)

