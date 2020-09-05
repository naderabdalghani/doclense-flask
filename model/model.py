import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split

batch_size = 20
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset_lowercase = datasets.ImageFolder('../dataset/lowercase', transform=data_transforms)
dataset_uppercase = datasets.ImageFolder('../dataset/uppercase', transform=data_transforms)

dataset = ConcatDataset([dataset_lowercase, dataset_uppercase])

dataset_len = len(dataset)
train_len = int(0.6 * dataset_len)
valid_len = int(0.2 * dataset_len)
test_len = dataset_len - train_len - valid_len

train_data, valid_data, test_data = random_split(dataset, [train_len, valid_len, test_len])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

loaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

model = models.wide_resnet50_2(pretrained=True)
print(model)
