import torch
import timm, tome
from tqdm.auto import tqdm
import torch
import torch.nn as nn

import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics import MulticlassAccuracy
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

from cjm_pytorch_utils.core import set_seed
import copy

import argparse

def val(model, dataloader, metric, device):
    model.eval()
    metric.reset()
    progress_bar = tqdm(total=len(dataloader), desc="Eval")
    with torch.no_grad():
        for batch_id, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) 
            metric.update(outputs.detach().cpu(), labels.detach().cpu())
            progress_bar.set_postfix(accuracy=metric.compute().item())
            progress_bar.update()
    progress_bar.close()
    return metric.compute()

parser = argparse.ArgumentParser(description='Đánh giá mô hình')
parser.add_argument('--model_name', type=str, required=True,
                    help='Tên của mô hình cần đánh giá')
parser.add_argument('--dataset_name', type=str, required=True,
                    help='Checkpoint')
parser.add_argument('--image_size', type=int, default=224, required=True,
                    help='Checkpoint')
args = parser.parse_args()

size = args.image_size
t = []
t.append(transforms.Resize(size, interpolation=3), )
t.append(transforms.CenterCrop(224))
t.append(transforms.ToTensor())
t.append(transforms.Normalize(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD))
transform = transforms.Compose(t)

def load_dataset(dataset_name, transform):
    if dataset_name == 'flower102':
        dataset_train = datasets.Flowers102('log', split = 'train', transform=transform, download = True)
        dataset_val = datasets.Flowers102('log', split = 'val', transform=transform, download = True)
        dataset_test = datasets.Flowers102('log', split = 'test', transform=transform, download = True)
        nb_classes = 102
    elif dataset_name == 'resisc45':
        dataset = ImageFolder('/kaggle/input/resisc45/NWPU-RESISC45', transform=transform)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.5 * (len(dataset) - train_size))
        test_size = len(dataset) - train_size - val_size
        dataset_train, dataset_val = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
        dataset_val, dataset_test = torch.utils.data.random_split(dataset_val, [val_size, test_size])
        nb_classes = 45
    elif dataset_name == 'pets37':
        dataset = datasets.OxfordIIITPet('log', split = 'trainval', transform=transform, download = True)
        dataset_train, dataset_val = torch.utils.data.random_split(dataset, [3312, 368])
        dataset_test = datasets.OxfordIIITPet('log', split = 'test', transform=transform, download = True)
        nb_classes = 37
    elif dataset_name == 'cifar100':
        dataset_test = datasets.CIFAR100('log', train = False, transform=transform, download = True)
        dataset = datasets.CIFAR100('log', train = True, transform=transform, download = True)
        dataset_train, dataset_val = torch.utils.data.random_split(dataset, [45000, 5000])
        nb_classes = 100
    elif dataset_name == 'dtd':
        dataset_train = datasets.DTD('log', split = 'train', transform=transform, download = True)
        dataset_val = datasets.DTD('log', split = 'val', transform=transform, download = True)
        dataset_test = datasets.DTD('log', split = 'test', transform=transform, download = True)
        nb_classes = 47
    elif dataset_name == 'eurosat':
        dataset = ImageFolder('/kaggle/input/eurosat/2750', transform=transform)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.5 * (len(dataset) - train_size))
        test_size = len(dataset) - train_size - val_size
        dataset_train, dataset_val = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
        dataset_val, dataset_test = torch.utils.data.random_split(dataset_val, [val_size, test_size])
        nb_classes = 10
    elif dataset_name == 'svhn':
        dataset = datasets.SVHN('log', split = 'train', transform=transform, download = True)
        train_size = int(0.9 * len(dataset))
        val_size = (len(dataset) - train_size)
        dataset_train, dataset_val = torch.utils.data.random_split(dataset, [train_size, val_size])
        dataset_test = datasets.SVHN('log', split = 'test', transform=transform, download = True)
        nb_classes = 10
    elif dataset_name == 'fer2013':
        transform = transforms.Compose(
                [
                    transforms.Resize(size),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        dataset = datasets.FER2013('/kaggle/working/log_1/', split = 'train', transform=transform)#, download = True)
        train_size = int(0.9 * len(dataset))
        val_size = (len(dataset) - train_size)
        dataset_train, dataset_val = torch.utils.data.random_split(dataset, [train_size, val_size])
        dataset_test = datasets.FER2013('/kaggle/working/log_1/', split = 'test', transform=transform)#, download = True)
        nb_classes = 7
    elif dataset_name == 'pcam':
        dataset_train = ImageFolder(os.path.join('/kaggle/input/pcam-data-1', 'train'), transform=transform)
        dataset_val = ImageFolder(os.path.join('/kaggle/input/pcam-data-1', 'val'), transform=transform)
        dataset_test = ImageFolder(os.path.join('/kaggle/input/pcam-data-1', 'test'), transform=transform)
        nb_classes = 2
    elif dataset_name == 'isic2019':
        dataset = ImageFolder('/kaggle/input/isic2019-1/kaggle/working/train', transform=transform)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.5 * (len(dataset) - train_size))
        test_size = len(dataset) - train_size - val_size
        dataset_train, dataset_val = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
        dataset_val, dataset_test = torch.utils.data.random_split(dataset_val, [val_size, test_size])
        nb_classes = 8
    return dataset_train, dataset_val, dataset_test, nb_classes
    
dataset_train, dataset_val, dataset_test, nb_classes = load_dataset(args.dataset_name)
data_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)
seed = 1234
set_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = timm.create_model(args.model_name.split('.')[0] , pretrained=True, num_classes = num_classes)
model.load_state_dict(torch.load('/home/nhan-softzone/Tome_new/checkpoints/' + args.model_name + '_' + args.dataset_name + '.bin'))
model.to(device)
model.eval()

print('Load model. Done')

metric = MulticlassAccuracy()

baseline_acc = val(model, data_loader, metric, device)

import tome
tome_acc = []
metric = MulticlassAccuracy()
for r in range(1, 21):
    model_tome = copy.deepcopy(model)
    tome.patch.timm(model_tome)
    model_tome.r = r
    model_tome.method = 'none'
    tome_acc.append(val(model_tome, data_loader, metric, device).item())
    
import tome_x_attn
tome_x_attn_acc = []
metric = MulticlassAccuracy()
for r in range(1, 21):
    model_new_tome = copy.deepcopy(model)
    tome_x_attn.patch.timm(model_new_tome)
    model_new_tome.r = r
    tome_x_attn_acc.append(val(model_new_tome, data_loader, metric, device).item())
    
import pandas as pd
report_df = pd.DataFrame()
report_df['ToFu'] = tome_acc
report_df['Tome X_attn'] = tome_x_attn_acc

import json
def save_as_json(df, model_name, dataset_name, baseline_acc):
    data = df.to_dict(orient='list')
    data['model_name'] = model_name
    data['dataset_name'] = dataset_name
    data['baseline_acc'] = baseline_acc.item()
    with open('/kaggle/working/' + model_name + '_' + dataset_name + '.json', 'w') as f:
        json.dump(data, f)
save_as_json(report_df, args.model_name, args.dataset_name, baseline_acc)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(tome_acc, label='ToFu', color='blue', linewidth=2)
plt.plot(tome_x_attn_acc, label='Tome X_attn', color='orange', linewidth=2)

plt.title('Comparison of Tome vs ToFu')
plt.xlabel('R')
plt.ylabel('Accuracy')

plt.legend()

new_chart_path = args.model_name + '_' + args.dataset_name + '.png'
plt.savefig(new_chart_path)

plt.show()

new_chart_path