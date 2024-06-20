import os
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data_providers.base_provider import *

_cifar10_classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
_cifar100_classes = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]


class CifarDataProvider(DataProvider):
    def __init__(
        self,
        dataset="cifar10",
        save_path="~/data/dataset",
        train_batch_size=128,
        test_batch_size=100,
        valid_size=None,
        n_worker=32,
    ):

        self._save_path = save_path
        self.dataset = dataset.lower()
        train_transforms = self.build_train_transform()

        if dataset.lower() == "cifar10":
            # Change to CIFAR-10 dataset
            train_dataset = datasets.CIFAR10(
                root=save_path, train=True, download=True, transform=train_transforms
            )
            n_classes = 10
        elif dataset.lower() == "cifar100":
            # Change to CIFAR-100 dataset
            train_dataset = datasets.CIFAR100(
                root=save_path, train=True, download=True, transform=train_transforms
            )
            n_classes = 100
        else:
            raise ValueError(
                "Invalid dataset name. Supported values are 'cifar10' and 'cifar100'."
            )

        if valid_size is not None and valid_size > 0:
            if isinstance(valid_size, float):
                valid_size = int(valid_size * len(train_dataset))
            else:
                assert isinstance(valid_size, int), (
                    "invalid valid_size: %s" % valid_size
                )

            train_indexes, valid_indexes = self.random_sample_valid_set(
                train_dataset.targets,
                valid_size,
                n_classes,
            )
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)

            # Validation dataset
            valid_dataset = (
                datasets.CIFAR10(
                    root=save_path,
                    train=True,
                    download=True,
                    transform=transforms.Compose(
                        [
                            transforms.ToTensor(),
                            self.normalize,
                        ]
                    ),
                )
                if dataset.lower() == "cifar10"
                else datasets.CIFAR100(
                    root=save_path,
                    train=True,
                    download=True,
                    transform=transforms.Compose(
                        [
                            transforms.ToTensor(),
                            self.normalize,
                        ]
                    ),
                )
            )

            self.train = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                sampler=train_sampler,
                num_workers=n_worker,
                pin_memory=True,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=test_batch_size,
                sampler=valid_sampler,
                num_workers=n_worker,
                pin_memory=True,
            )
        else:
            self.train = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=n_worker,
                pin_memory=True,
            )
            self.valid = None

        # Test dataset
        test_dataset = (
            datasets.CIFAR10(
                root=save_path,
                train=False,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        self.normalize,
                    ]
                ),
            )
            if dataset.lower() == "cifar10"
            else datasets.CIFAR100(
                root=save_path,
                train=False,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        self.normalize,
                    ]
                ),
            )
        )
        self.test = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=n_worker,
            pin_memory=True,
        )

        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return "cifar"

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = "/dataset/cifar"
        return self._save_path

    @property
    def data_url(self):
        return (
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            if self.dataset.lower() == "cifar10"
            else "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        )

    @property
    def train_path(self):
        return os.path.join(self.save_path, "train")

    @property
    def valid_path(self):
        return os.path.join(self._save_path, "val")

    @property
    def normalize(self):
        # https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
        if self.dataset.lower() == "cifar10":
            return transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
            )
        elif self.dataset.lower() == "cifar100":
            return transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            )

    def build_train_transform(self):
        train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(self.resize_value, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ]
        )
        return train_transforms

    @property
    def resize_value(self):
        return 32  # For CIFAR datasets, set to the appropriate size

    @property
    def image_size(self):
        return 32  # For CIFAR datasets, set to the appropriate size

    @property
    def classes(self):
        # Define the class labels based on the dataset
        return (
            _cifar10_classes if self.dataset.lower() == "cifar10" else _cifar100_classes
        )

    @property
    def n_classes(self):
        # Define the class labels based on the dataset
        return 10 if self.dataset.lower() == "cifar10" else 100


# Define CIFAR-10 and CIFAR-100 class labels


def get_label(loader):
    ret = []
    for img, lab in loader:
        ret.append(lab)


if __name__ == "__main__":
    import torch
    import torchvision
    import torchvision.transforms as transforms

    # Initialize the CIFAR-10 data provider
    cifar10_provider = CifarDataProvider(dataset="cifar100", valid_size=128)

    # Load the training, validation, and test datasets
    train_loader = cifar10_provider.train
    valid_loader = cifar10_provider.valid
    test_loader = cifar10_provider.test

    # Display a batch of images from the training set
    def imshow(img):
        import matplotlib.pyplot as plt

        img = img / 2 + 0.5  # Unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig("cat.pdf")

    print(f"{train_loader.batch_size=}")
    print(f"{cifar10_provider.n_classes=}")
    # Get some random training images
    # for images, labels in train_loader:
    #     imshow(torchvision.utils.make_grid(images))
    #     print(' '.join('%5s' % cifar10_provider.classes[labels[j]] for j in range(4)))  # Display labels for the first 4 images
    #     break
