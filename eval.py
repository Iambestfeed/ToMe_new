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

from cjm_pytorch_utils.core import set_seed, pil_to_tensor, tensor_to_pil, get_torch_device, denorm_img_tensor
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
args = parser.parse_args()

size = 256
t = []
t.append(transforms.Resize(size, interpolation=3), )
t.append(transforms.CenterCrop(224))
t.append(transforms.ToTensor())
t.append(transforms.Normalize(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD))
transform = transforms.Compose(t)

if args.dataset_name == 'cifar100':
    imagenet_data = datasets.CIFAR100('log', train = False, transform=transform, download = True)
    data_loader = DataLoader(imagenet_data, batch_size=32, shuffle=False)
    num_classes = 100
elif args.dataset_name == 'resisc45':
    dataset = ImageFolder('/kaggle/input/resisc45/NWPU-RESISC45', transform=transform)
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [25200, 6300])
    dataset_val, dataset_test = torch.utils.data.random_split(dataset_val, [3150, 3150])
    data_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)
    num_classes = 45
elif args.dataset_name == 'flower102':
    dataset_train = datasets.Flowers102('log', split = 'train', transform=transform, download = True)
    dataset_val = datasets.Flowers102('log', split = 'val', transform=transform, download = True)
    dataset_test = datasets.Flowers102('log', split = 'test', transform=transform, download = True)
    data_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)
    num_classes = 102
elif args.dataset_name == 'pets37':
    dataset = datasets.OxfordIIITPet('log', split = 'trainval', transform=transform, download = True)
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [3312, 368])
    dataset_test = datasets.OxfordIIITPet('log', split = 'test', transform=transform, download = True)
    data_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)
    num_classes = 37
    
seed = 1234
set_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = timm.create_model(args.model_name, pretrained=True, num_classes = num_classes)
if 'base' in args.model_name:
    model.load_state_dict(torch.load('/kaggle/input/checkpoint-finetune-vit-for-benchmark-diffrate/' + args.model_name + '_' + args.dataset_name + '.bin'))
elif 'tiny' in args.model_name:
    model.load_state_dict(torch.load('/kaggle/input/checkpoint-deit-tiny/' + args.model_name + '_' + args.dataset_name + '.bin'))
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

new_chart_path = '/kaggle/working/' + args.model_name + '_' + args.dataset_name + '.png'
plt.savefig(new_chart_path)

plt.show()

new_chart_path
