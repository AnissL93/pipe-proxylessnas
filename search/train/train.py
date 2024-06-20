"""Train CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import OneCycleLR

import os
import argparse

from pathlib import Path
import sys

from tensorboardX import SummaryWriter

sys.path.append("/home/huiying/projects/nas/pipeproxylessnas/search")

from models import *
from train_utils import progress_bar

from from_net import read_from_config

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
parser.add_argument("--path", "-p", type=str, help="path to the model config file")
parser.add_argument("--pretrain", type=bool, help="if accept pretrain", default=False)
parser.add_argument("--dropout", type=float, help="if use dropout", default=0.0)
parser.add_argument(
    "--lr_sch",
    default="cosine",
    help="lr scheduler",
    type=str,
    choices=["onecycle", "cosine"],
)
parser.add_argument("--test", default=False, action="store_true")

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print("==> Preparing data..")
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="~/data/dataset", train=True, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=32
)

testset = torchvision.datasets.CIFAR10(
    root="~/data/dataset", train=False, download=True, transform=transform_test
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=True, num_workers=32
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Model
print("==> Building model..")
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()

path = Path(args.path)
net = read_from_config(path)
print(net)
net = net.to(device)

if device == "cuda":
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    ckp = path / "checkpoint"
    assert os.path.isdir(ckp), "Error: no checkpoint directory found!"
    checkpoint = torch.load(ckp / "ckpt.pth")
    net.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

if args.lr_sch == "onecycle":
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=len(trainloader), epochs=500
    )
elif args.lr_sch == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
else:
    raise ValueError("lr scheduler not supported")

writer = SummaryWriter()

# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )
    writer.add_scalar("train/loss", train_loss / (batch_idx + 1), epoch)
    writer.add_scalar("train/acc", 100.0 * correct / total, epoch)


def test(epoch, save=True):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(testloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

    writer.add_scalar("test/loss", test_loss / (batch_idx + 1), epoch)
    writer.add_scalar("test/acc", 100.0 * correct / total, epoch)

    if save:
        # Save checkpoint.
        acc = 100.0 * correct / total
        if acc > best_acc:
            print("Saving..")
            state = {
                "net": net.state_dict(),
                "acc": acc,
                "epoch": epoch,
            }
            ckp = path / "checkpoint"
            ckp.mkdir(parents=True, exist_ok=True)
            torch.save(state, ckp / "ckpt.pth")
            best_acc = acc


if args.test:
    test(0, False)
else:
    for epoch in range(start_epoch, start_epoch + 500):
        print(f"epoch: {epoch}")
        train(epoch)
        test(epoch)
        scheduler.step()

writer.close()