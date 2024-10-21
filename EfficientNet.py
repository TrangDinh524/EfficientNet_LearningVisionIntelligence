from collections import defaultdict
import math

import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP


NUM_EPOCHS = 90
NUM_WORKERS = 4
BATCH_SIZE = 128
NUM_CLASSES = 100
IMG_SIZE = 224
MODEL_CKPT_PATH = None
NORM_MEAN = (0.4914, 0.4822, 0.4465)
NORM_STD = (0.2470, 0.2435, 0.2616)
TTA = True  # test time augmentation

TRAIN_VAL_SPLIT = 0.0
TRAIN_VAL_SPLIT_SEED = 542

# Optimizer
LR = 1.25e-2 * int(BATCH_SIZE / 32)
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
NESTEROV = True

# Learning rate scheduler
LR_SCHEDULER_MILESTONES = [30, 60, 80]
LR_SCHEDULER_GAMMA = 0.1

BEST_CKPT_SAVE_PATH = "/home/cipher/Trang/run/best.pth"
LOSS_PLOT_SAVE_PATH = "/home/cipher/Trang/run/loss.png"
ACCURACY_PLOT_SAVE_PATH = "/home/cipher/Trang/run/accuracy.png"
LR_PLOT_SAVE_PATH = "/home/cipher/Trang/run/lr.png"


superclass_mapping = {
    'aquatic mammals': [4, 30, 55, 72, 95],
    'fish': [1, 32, 67, 73, 91],
    'flowers': [54, 62, 70, 82, 92],
    'food containers': [9, 10, 16, 28, 61],
    'fruit and vegetables': [0, 51, 53, 57, 83],
    'household electrical devices': [22, 39, 40, 86, 87],
    'household furniture': [5, 20, 25, 84, 94],
    'insects': [6, 7, 14, 18, 24],
    'large carnivores': [3, 42, 43, 88, 97],
    'large man-made outdoor things': [12, 17, 37, 68, 76],
    'large natural outdoor scenes': [23, 33, 49, 60, 71],
    'large omnivores and herbivores': [15, 19, 21, 31, 38],
    'medium-sized mammals': [34, 63, 64, 66, 75],
    'non-insect invertebrates': [26, 45, 77, 79, 99],
    'people': [2, 11, 35, 46, 98],
    'reptiles': [27, 29, 44, 78, 93],
    'small mammals': [36, 50, 65, 74, 80],
    'trees': [47, 52, 56, 59, 96],
    'vehicles 1': [8, 13, 48, 58, 90],
    'vehicles 2': [41, 69, 81, 85, 89]
}


params = {
    "efficientnet_b0": (1.0, 1.0, 224, 0.2),
    "efficientnet_b1": (1.0, 1.1, 240, 0.2),
    "efficientnet_b2": (1.1, 1.2, 260, 0.3),
    "efficientnet_b3": (1.2, 1.4, 300, 0.3),
    "efficientnet_b4": (1.4, 1.8, 380, 0.4),
    "efficientnet_b5": (1.6, 2.2, 456, 0.4),
    "efficientnet_b6": (1.8, 2.6, 528, 0.5),
    "efficientnet_b7": (2.0, 3.1, 600, 0.5),
}


"""
Building blocks
"""


class SwishActivation(nn.Module):
    """Custom Swish activation module, as detailed in paper
        "Searching for Activation Functions" https://arxiv.org/abs/1710.05941
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvBNSwishCustom(nn.Sequential):
    """
    Convolutional layer followed by Batch Normalization and Swish activation, which uses custom padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        p = max(kernel_size - stride, 0)
        pad = [p // 2, p - p // 2, p // 2, p - p // 2]
        super().__init__(
            nn.ZeroPad2d(pad),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                bias=False,
                padding=0,
                groups=groups,
            ),
            nn.BatchNorm2d(out_channels),
            SwishActivation(),
        )


class SaENet(nn.Module):
    """
    Squeeze-and-Excitation block, as detailed in paper
        "Squeeze and Excitation Networks" https://arxiv.org/abs/1709.01507
    """
    def __init__(self, in_channels, inter_dim):
        super().__init__()
        self.sae = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_dim, 1),
            SwishActivation(),
            nn.Conv2d(inter_dim, in_channels, 1),
            nn.Sigmoid(),
        )   # Sigmoid activation function is used to output a value between 0 and 1

    def forward(self, x):
        return self.sae(x) * x


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck block, as detailed in paper
        "MobileNetV2: Inverted Residuals and Linear Bottlenecks" https://arxiv.org/abs/1801.04381
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        expand_prop,
        kernel_size,
        stride,
        red_prop=4,
        res_rate=0.2,
    ) -> None:
        super().__init__()
        self.res_rate = res_rate
        self.is_residual = in_channels == out_channels and stride == 1

        hidden_dim = in_channels * expand_prop
        reduced_dim = max(1, int(in_channels / red_prop))

        # The expansion layer
        layers = []
        if in_channels != hidden_dim:
            layers += [ConvBNSwishCustom(in_channels, hidden_dim, 1)]

        # Finally, the depthwise convolution
        layers += [
            ConvBNSwishCustom(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim),
            SaENet(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        self.layers = nn.Sequential(*layers)

    def _drop_connect(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        keep_prob = 1.0 - self.res_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_residual:
            return x + self._drop_connect(self.layers(x))
        else:
            return self.layers(x)


"""
Main EfficientNet model
"""


def get_num_div(value, divisor=8):
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def get_out_channels(filters, width_mult):
    if width_mult == 1.0:
        return filters
    return math.ceil(get_num_div(filters * width_mult))


def get_repeats(repeats, depth_mult):
    if depth_mult == 1.0:
        return repeats
    return math.ceil(depth_mult * repeats)


def weight_init(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            fan_out = m.weight.size(0)
            init_range = 1.0 / math.sqrt(fan_out)
            nn.init.uniform_(m.weight, -init_range, init_range)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class EfficientNet(nn.Module):
    def __init__(self, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes

        # Default settings, as detailed in the paper
        # t: expansion factor of the bottleneck layer
        # c: number of output channels
        # n: number of times the block is repeated
        # s: stride of the first layer
        # k: kernel size of the depthwise separable convolution
        settings = [
            # t, c, n, s, k
            [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112
            [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56
            [6,  40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28
            [6,  80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7
            [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   7 ->   7
        ]
        out_channels = get_out_channels(32, width_mult)
        layers = [ConvBNSwishCustom(3, out_channels, 3, stride=2)]

        in_channels = out_channels
        for t, c, n, s, k in settings:
            out_channels = get_out_channels(c, width_mult)
            repeats = get_repeats(n, depth_mult)
            for i in range(repeats):
                stride = s if i == 0 else 1
                layers += [
                    MBConvBlock(
                        in_channels,
                        out_channels,
                        expand_prop=t,
                        stride=stride,
                        kernel_size=k,
                    )
                ]
                in_channels = out_channels

        last_channels = get_out_channels(1280, width_mult)
        layers += [ConvBNSwishCustom(in_channels, last_channels, 1)]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes),
        )
        weight_init(self.modules())

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=[2, 3])
        x = self.classifier(x)
        return x


def eval(model, test_loader, device, do_print=True):
    # 反向映射
    class_to_superclass = {}
    for superclass, classes in superclass_mapping.items():
        for class_idx in classes:
            class_to_superclass[class_idx] = superclass

    model.eval()
    correct = 0
    total = 0
    top1_correct = 0
    top5_correct = 0
    superclass_correct = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    superclass_correct_dict = defaultdict(int)
    superclass_total = defaultdict(int)

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if TTA:
                inputs_flipped = torch.flip(inputs, [3])  # [B, C, H, W]
                outputs_flipped = model(inputs_flipped)
                outputs = (outputs + outputs_flipped) / 2

            # Top-1 Accuracy
            _, top1_pred = outputs.max(1)
            top1_correct += top1_pred.eq(labels).sum().item()

            total += labels.size(0)
            correct += top1_pred.eq(labels).sum().item()
            # Top-5 Accuracy
            _, top5_pred = outputs.topk(5, 1, largest=True, sorted=True)
            top5_correct += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()

            # Superclass Accuracy
            for i, label in enumerate(labels):
                superclass = class_to_superclass[label.item()]
                superclass_total[superclass] += 1
                if top1_pred[i].item() in superclass_mapping[superclass]:
                    superclass_correct += 1
                    superclass_correct_dict[superclass] += 1

            # Per-class accuracy
            for i, label in enumerate(labels):
                class_total[label.item()] += 1
                if top1_pred[i] == label:
                    class_correct[label.item()] += 1

    accuracy = 100. * correct / total
    top1_accuracy = 100. * top1_correct / total
    top5_accuracy = 100. * top5_correct / total
    superclass_accuracy = 100. * superclass_correct / total

    if do_print:
        print(f'Test Accuracy: {accuracy:.2f}%')
        print(f'Top-1 Accuracy: {top1_accuracy:.2f}%')
        print(f'Top-5 Accuracy: {top5_accuracy:.2f}%')
        print(f'Superclass Accuracy: {superclass_accuracy:.2f}%')

        # 打印每个超类的准确率
        for superclass in superclass_mapping:
            if superclass_total[superclass] > 0:
                superclass_acc = 100. * superclass_correct_dict[superclass] / superclass_total[superclass]
                print(f'{superclass} Accuracy: {superclass_acc:.2f}%')

    return accuracy


def train_one_epoch(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, device, device_id):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, disable=device_id != 0):
        inputs, labels = inputs.to(device), labels.to(device)

        # Reset lại gradient
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass và tối ưu hóa
        loss.backward()
        optimizer.step()

        # Thống kê loss và accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)

    # Compute train accuracy
    total = torch.tensor(total, dtype=torch.float32, device=device)
    correct = torch.tensor(correct, dtype=torch.float32, device=device)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    epoch_accuracy = 100 * correct.item() / total.item()

    # Validate the model
    if len(val_loader) > 0:
        if device_id == 0:
            val_accuracy = eval(model, val_loader, device, do_print=False)
        else:
            val_accuracy = 0
    else:
        val_accuracy = epoch_accuracy

    # Update learning rate
    lr_scheduler.step()

    return epoch_loss, epoch_accuracy, val_accuracy


class CIFAR100TrainingDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.dataset = datasets.CIFAR100(*args, **kwargs)
        rng = np.random.RandomState(TRAIN_VAL_SPLIT_SEED)
        indices = np.arange(len(self.dataset))
        rng.shuffle(indices)
        split_idx = int(len(indices) * TRAIN_VAL_SPLIT)
        self.indices = indices[split_idx:].tolist()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class CIFAR100ValidationDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.dataset = datasets.CIFAR100(*args, **kwargs)
        rng = np.random.RandomState(TRAIN_VAL_SPLIT_SEED)
        indices = np.arange(len(self.dataset))
        rng.shuffle(indices)
        split_idx = int(len(indices) * TRAIN_VAL_SPLIT)
        self.indices = indices[:split_idx].tolist()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class AlbumentationsTransformWrapper:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        img = np.asarray(img)
        return self.transforms(image=img)["image"]


def run(world_size, device_id):
    # Load CIFAR-100 dataset with transformation to tensor
    # transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
    #                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    # Adapted from
    # https://albumentations.ai/docs/examples/pytorch_classification/
    train_transform = AlbumentationsTransformWrapper(A.Compose(
        [
            A.SmallestMaxSize(max_size=[224, 256, 512]),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomCrop(height=IMG_SIZE, width=IMG_SIZE),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ToTensorV2(),
        ]
    ))
    test_transform = AlbumentationsTransformWrapper(A.Compose(
        [
            A.SmallestMaxSize(max_size=IMG_SIZE),
            A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ToTensorV2(),
        ]
    ))

    # Datasets and dataloaders
    training_dataset = CIFAR100TrainingDataset('./data_src', train=True, download=True, transform=train_transform)
    train_sampler = DistributedSampler(
        training_dataset,
        num_replicas=world_size,
        rank=device_id,
        shuffle=True
    )
    train_loader = DataLoader(
        dataset=training_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        sampler=train_sampler,
        pin_memory=True
    )

    val_dataset = CIFAR100ValidationDataset('./data_src', train=True, download=True, transform=test_transform)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_dataset = datasets.CIFAR100('./data_src', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # Load the EfficientNet-B0 model
    device = torch.device(f"cuda:{device_id}")
    model = EfficientNet(1.0, 1.0, 0.2, num_classes=NUM_CLASSES).to(device)

    # Optimizer with a smaller learning rate
    optimizer = optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=NESTEROV
    )
    model = DDP(model, device_ids=[device_id])

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=LR_SCHEDULER_MILESTONES,
        gamma=LR_SCHEDULER_GAMMA
    )

    # Loss function remains the same
    criterion = nn.CrossEntropyLoss()

    # model training
    best_accuracy = 0
    epoch_losses = []
    epoch_accuracies = []
    val_accuracies = []
    lrs = []

    for epoch in range(NUM_EPOCHS):
        lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)

        epoch_loss, epoch_accuracy, val_accuracy = train_one_epoch(
            model, train_loader, val_loader, criterion, optimizer, lr_scheduler, device, device_id)
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)
        val_accuracies.append(val_accuracy)

        if device_id == 0:
            print(
                f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, '
                f'Validation accuracy: {val_accuracy:.2f}%, Learning Rate: {lr:.4f}'
            )
        if device_id == 0 and val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), BEST_CKPT_SAVE_PATH)
            print(f'Found new best model with accuracy on validation set: {best_accuracy:.2f}%, epoch: {epoch+1}')

    print("Finish training. Start evaluating")
    # Load the best model
    if device_id == 0:
        model.load_state_dict(torch.load(BEST_CKPT_SAVE_PATH))
        eval(model, test_loader, device)

    # Plot statistics
    if device_id == 0:
        plt.plot(epoch_losses)
        plt.title("Training loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(LOSS_PLOT_SAVE_PATH)
        plt.clf()

        plt.plot(epoch_accuracies, label="Train accuracy")
        if len(val_loader) > 0:
            plt.plot(val_accuracies, label="Validation accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(ACCURACY_PLOT_SAVE_PATH)
        plt.clf()

        plt.plot(lrs)
        plt.title("Learning rate")
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.tight_layout()
        plt.savefig(LR_PLOT_SAVE_PATH)
        plt.clf()

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running DDP on rank {rank}.")
    device_id = rank % torch.cuda.device_count()
    world_size = dist.get_world_size()
    run(world_size, device_id)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
