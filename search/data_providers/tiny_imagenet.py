# DataProvider for tiny imagenet
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
from PIL import Image
from datasets import load_dataset
from datasets import load_from_disk

import os

from data_providers.base_provider import *

DATA_PATH = "/home/huiying/data/dataset/tiny_image"

class TinyImagenetDataProvider(DataProvider):

    def info(self):
        class_names = self.train_dataset.classes
        # Initialize a dictionary to store the counts for each label
        label_counts = {class_name: 0 for class_name in class_names}
        # Iterate over the dataset and count the samples for each label
        for _, label in self.train_dataset:
            class_name = class_names[label]
            label_counts[class_name] += 1
        # Print the counts for each label
        for class_name, count in label_counts.items():
            print(f"Class: {class_name}, Number of samples: {count}")

    def __init__(self, dataset="tiny_imagenet", save_path=DATA_PATH, train_batch_size=256, test_batch_size=512, valid_size=256,
                 n_worker=32, resize_scale=0.08, distort_color=None, skip_transform=False):
        self.dataset = dataset.lower()
        self._save_path = save_path

        self.train_dataset = load_from_disk(save_path +  '/train')
        self.test_dataset = load_from_disk(save_path + '/test')

        if valid_size is not None:
            if isinstance(valid_size, float):
                valid_size = int(valid_size * len(self.train_dataset))
            else:
                assert isinstance(
                    valid_size, int), 'invalid valid_size: %s' % valid_size
            
            train_indexes, valid_indexes = self.random_sample_valid_set(
                self.train_dataset["label"], valid_size, self.n_classes,
            )
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                train_indexes)
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                valid_indexes)

            self.train = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=test_batch_size, sampler=valid_sampler,
                num_workers=n_worker, pin_memory=True,
            )
        else:
            self.train = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=train_batch_size, shuffle=True,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = None

        self.test = torch.utils.data.DataLoader(self.test_dataset,
                                                batch_size=test_batch_size, shuffle=False, 
                                                num_workers=n_worker, pin_memory=True)

        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'tiny_imagenet'

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return 200

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = '/home/datasets/tiny_imagenet/tiny_imagenet'
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download Tiny ImageNet')

    @property
    def train_path(self):
        return os.path.join(self.save_path, 'train')

    @property
    def valid_path(self):
        return os.path.join(self.save_path, 'val')

    @property
    def normalize(self):
        return transforms.Normalize((0.4802, 0.4481, 0.3975), (0.0564, 0.0547, 0.0552))

    def build_train_transform(self, distort_color, resize_scale):
        print('Color jitter: %s' % distort_color)
        if distort_color == 'strong':
            color_transform = transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        elif distort_color == 'normal':
            color_transform = transforms.ColorJitter(
                brightness=32. / 255., saturation=0.5)
        else:
            color_transform = None

        if color_transform is None:
            train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(
                    self.image_size, scale=(resize_scale, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(
                    self.image_size, scale=(resize_scale, 1.0)),
                transforms.RandomHorizontalFlip(),
                color_transform,
                transforms.ToTensor(),
                self.normalize,
            ])
        return train_transforms

    @property
    def resize_value(self):
        return 64

    @property
    def image_size(self):
        return 64

    def cls_in_words(self, cls):
        pass

    @property
    def cls(self):
        words_file = os.path.join(self.save_path, "words.txt")
        ret = {}
        with open(words_file) as fp:
            for line in fp:
                w = line.split()
                cls_code, cls_name = w[0], w[1]
                ret[cls_code] = cls_name
        return ret


def test_tiny_imagenet_data_provider():
    # data_provider = TinyImagenetDataProvider(
    #     save_path="/home/huiying/data/dataset/ttt/tiny-imagenet-200", distort_color='strong')
    data_provider = TinyImagenetDataProvider(save_path="tiny_image", distort_color='strong')
    # data_provider.info()
    print(data_provider.cls)
    print("Data shape:", data_provider.data_shape)
    print("Number of classes:", data_provider.n_classes)

    train_loader = data_provider.train
    valid_loader = data_provider.valid
    test_loader = data_provider.test

    print("Train dataset size:", len(train_loader.dataset))
    print("Valid dataset size:", len(valid_loader.dataset))
    print("Test dataset size:", len(test_loader.dataset))

    # Iterate over a few batches to check if the data is loaded correctly
    for batch_idx, ds in enumerate(train_loader):
        img = ds["image"]
        label = ds["label"]
        print("Batch:", batch_idx, "Data shape:", img.shape, "Target shape:", label.shape)
        if batch_idx >= 2:  # Stop after a few batches
            break

    for batch_idx, ds in enumerate(valid_loader):
        data = ds["image"]
        target = ds["label"]
        print("Batch:", batch_idx, "Data shape:",
              data.shape, "Target shape:", target.shape)
        if batch_idx >= 2:  # Stop after a few batches
            break

    for batch_idx, ds in enumerate(test_loader):
        data = ds["image"]
        target = ds["label"]
        print("Batch:", batch_idx, "Data shape:",
              data.shape, "Target shape:", target.shape)
        print(target)
        if batch_idx >= 2:  # Stop after a few batches
            break


def train_with_resnet(provider: TinyImagenetDataProvider):
    model = resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, provider.n_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_loader = provider.train
    valid_loader = provider.valid
    test_loader = provider.test
    # Train the model
    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            inputs = data["image"]
            labels = data["label"]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validate the model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valid_loader:
                inputs = data["image"]
                labels = data["label"]
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {correct/total*100:.2f}%')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs = data["image"]
            labels = data["label"]
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {correct/total*100:.2f}%')

    pass

if __name__ == "__main__":
    # test_tiny_imagenet_data_provider()
    dataset_path = "/home/huiying/data/dataset/tiny_image"
    data_provider = TinyImagenetDataProvider(save_path="~/data/tiny_image", distort_color='strong')
    train_with_resnet(data_provider)
    pass
