import pickle
import gzip
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader
from rich.progress import Progress, BarColumn, TextColumn
from rich.console import Console
from torchvision import datasets, transforms

# Load local dataset
# with gzip.open((pathlib.Path("data/mnist") / "mnist.pkl.gz").as_posix(), "rb") as f:
#     ((train_images, train_labels), (valid_images, valid_labels), _) = pickle.load(f, encoding="latin-1")
#
# # Tensorise
# train_images, train_labels, valid_images, valid_labels = map(
#     torch.tensor, (train_images, train_labels, valid_images, valid_labels)
# )
#
# # Normalization
# train_images = train_images.float() / 255.0
# valid_images = valid_images.float() / 255.0
# Turn tensors to the dataset
# train_ds, valid_ds = TensorDataset(train_images, train_labels), TensorDataset(valid_images, valid_labels)

# auto download dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_ds = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root='data', train=False, download=True, transform=transform)

# divide datasets into train set and test set (80% 4 train 20 4 test)
train_size = int(0.8 * len(train_ds))
val_size = len(train_ds) - train_size
train_ds, valid_ds = random_split(train_ds, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(valid_ds, batch_size=128, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

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
            refresh_per_second=10
    ) as progress:
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
            progress.update(task, advance=1, validation_los=val_loss)


def evaluate(model, data_loader, loss_func):
    model.eval()
    losses, nums = [], []

    with torch.no_grad():
        for input, target in data_loader:
            loss, num = loss_batch(model, loss_func, input, target)
            losses.append(loss * num)
            nums.append(num)

    val_loss = np.sum(losses) / np.sum(nums)
    accuracy = sum(torch.argmax(model(input), dim=1).eq(target).sum().item() for input, target in data_loader) / np.sum(
        nums)
    return val_loss, accuracy

train_dl, valid_dl = get_data(train_ds, valid_ds)
model, optimizer = get_model()
fit(25, model, loss_func, optimizer, train_dl, valid_dl)

val_loss, val_accuracy = evaluate(model, valid_dl, loss_func)
print(f"验证集损失: {val_loss:.4f}, 验证集准确率: {val_accuracy:.4f}")
test_loss, test_accuracy = evaluate(model, test_loader, loss_func)
print(f"测试集损失: {test_loss:.4f}, 测试集准确率: {test_accuracy:.4f}")