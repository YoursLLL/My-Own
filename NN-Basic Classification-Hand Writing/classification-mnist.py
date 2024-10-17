import pickle
import gzip
import pathlib
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from rich.progress import Progress, BarColumn, TextColumn
from rich.console import Console
with gzip.open((pathlib.Path("data/mnist") / "mnist.pkl.gz").as_posix(), "rb") as f:
    ((train_images, train_labels), (valid_images, valid_labels), _) = pickle.load(f, encoding="latin-1")

# Tensorise
train_images, train_labels, valid_images, valid_labels = map(
    torch.tensor, (train_images, train_labels, valid_images, valid_labels)
)

# Normalization
train_images = train_images.float() / 255.0
valid_images = valid_images.float() / 255.0

# Turn tensors to the dataset
train_ds, valid_ds = TensorDataset(train_images, train_labels), TensorDataset(valid_images, valid_labels)

# Neural network structure definition
class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_1st = nn.Linear(784, 128)
        self.hidden_2nd = nn.Linear(128, 256)
        self.out = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.hidden_1st(x))
        x = self.dropout(x)
        x = F.relu(self.hidden_2nd(x))
        x = self.dropout(x)
        x = self.out(x)
        return x

loss_func = F.cross_entropy

def get_data(train_ds, valid_ds):
    return (DataLoader(train_ds, batch_size=64, shuffle=True), DataLoader(valid_ds, batch_size=128, shuffle=False))

def get_model():
    model = Mnist_NN()
    return model, torch.optim.Adam(model.parameters(), lr=0.001)

def loss_batch(model, loss_func, input_batch, target_batch, opt=None):
    loss = loss_func(model(input_batch), target_batch)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(),len(input_batch)


console = Console(force_terminal=True)


def fit(steps, model, loss_func, optimizer, train_dl, valid_dl):
    with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TextColumn("验证集损失: {task.fields[validation_los]:.4f}"),
            console=console,
            refresh_per_second=10  # 控制刷新频率
    ) as progress:
        # 初始化进度条
        task = progress.add_task("[red]Training...", total=steps, validation_los=0.0)

        for step in range(steps):
            model.train()
            for input, target in train_dl:
                loss_batch(model, loss_func, input, target, opt=optimizer)

            model.eval()
            with torch.no_grad():
                losses, nums = zip(
                    *[loss_batch(model, loss_func, input, target) for input, target in valid_dl])
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            # 更新进度条中的损失值
            progress.update(task, advance=1, validation_los=val_loss)

train_dl, valid_dl = get_data(train_ds, valid_ds)
model, optimizer = get_model()
fit(25, model, loss_func, optimizer, train_dl, valid_dl)
