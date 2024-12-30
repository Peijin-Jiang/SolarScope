import os
import torch
import timm
import wandb
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datetime import datetime
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, precision_recall_curve, auc, precision_score, recall_score
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np


def adjust_learning_rate(optimizer, epoch, lr, min_lr=1e-6, epochs=10, warmup_epochs=1):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs 
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def make_transforms(image_size):
  print("Setting image size to ", image_size)
  TRAIN_TRANSFORM = transforms.Compose([
      transforms.RandomResizedCrop(image_size, scale=(0.5, 1.5), ratio=(0.75, 1.33)),
      transforms.RandomHorizontalFlip(),
      transforms.RandomVerticalFlip(),
      transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
      transforms.RandomRotation(degrees=15),
      transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])

  VAL_TRANSFORM = transforms.Compose([
      transforms.Resize((image_size, image_size)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])
  return TRAIN_TRANSFORM, VAL_TRANSFORM


def make_models_transforms(model_name, image_size, freeze_backbone=False):
    train_transforms, val_transforms = None, None

    if "resnet" in model_name or "inception" in model_name:
        # Existing code for ResNet models
        model = timm.create_model(model_name, pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True

    elif "dinov2" in model_name or "clip" in model_name or "swin" in model_name:
        # Customizing for DINOv2 models
        if "swin" in model_name:
          model = timm.create_model(model_name, pretrained=True, num_classes=1)
          # get model specific transforms (normalization, resize)
          data_config = timm.data.resolve_model_data_config(model)
          timm_transforms = timm.data.create_transform(**data_config, is_training=False)
          train_transforms, val_transforms =timm_transforms, timm_transforms
        else:
          model = timm.create_model(model_name, pretrained=True)
          model.head = nn.Linear(model.num_features, 1)
          if freeze_backbone:
              for param in model.parameters():
                  param.requires_grad = False
              for param in model.head.parameters():
                  param.requires_grad = True
          # Resolve data configuration for the model
          data_config = timm.data.resolve_model_data_config(model)
          mean = data_config['mean']
          std = data_config['std']
          # Define training transforms with additional augmentations
          # augmentation 1
          # train_transforms = transforms.Compose([
          #     transforms.RandomResizedCrop(image_size, scale=(0.8, 1.2), ratio=(0.75, 1.33)),
          #     transforms.RandomHorizontalFlip(),
          #     transforms.RandomVerticalFlip(),
          #     transforms.RandomRotation(degrees=15),
          #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
          #     transforms.ToTensor(),
          #     transforms.Normalize(mean=mean, std=std),
          # ])
          # augmentation 2, stronger augmentation
          train_transforms = transforms.Compose([
              transforms.RandomResizedCrop(image_size, scale=(0.5, 1.5), ratio=(0.75, 1.33)),
              transforms.RandomHorizontalFlip(),
              transforms.RandomVerticalFlip(),
              transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
              transforms.RandomRotation(degrees=15),
              transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
              transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
              transforms.ToTensor(),
              transforms.Normalize(mean=mean, std=std),
          ])
          val_transforms = transforms.Compose([
              transforms.Resize(int(image_size * 1.1)),
              transforms.CenterCrop(image_size),
              transforms.ToTensor(),
              transforms.Normalize(mean=mean, std=std),
          ])

    else:
        raise NotImplementedError(f"Model {model_name} is not supported.")

    # Fallback to default transforms if not set
    if train_transforms is None or val_transforms is None:
        train_transforms, val_transforms = make_transforms(image_size)

    return model, train_transforms, val_transforms



class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = int(self.data.iloc[idx, 1])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def classification_metrics(y_true, y_pred, y_scores):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_scores)
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)
    
    # Calculate overall precision and recall
    overall_precision = precision_score(y_true, y_pred)
    overall_recall = recall_score(y_true, y_pred)

    return acc, bacc, auroc, auprc, overall_precision, overall_recall


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, warmups=1, lr=0.001, save_dir="", resume_ckpt="", start_epoch=None):
    if resume_ckpt != "":
      if start_epoch is None:
        start_epoch = int(resume_ckpt.split("_")[-1].split(".")[0]) + 1
      model.load_state_dict(torch.load(resume_ckpt))
      print(f"Resuming training from epoch {start_epoch}", "Load model from", resume_ckpt)
    else:
        start_epoch = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # set up the fp16 scaler
    fp16_scaler = torch.cuda.amp.GradScaler()
    torch.save(model.state_dict(), os.path.join(save_dir, f"model_init.pth"))
    best_val_loss, best_model_state = float('inf'), None
    for epoch in range(start_epoch, epochs):
        train_loss, val_loss = 0.0, 0.0
        model.train()
        pbar = tqdm(train_loader, desc="Training for one epoch")
        batch_idx = 0
        for images, labels in pbar:
            # adjust learning rate
            epoch_ = epoch + batch_idx / len(train_loader)
            adjust_learning_rate(optimizer, epoch_, lr, epochs=epochs, warmup_epochs=warmups)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                images, labels = images.to(device), labels.to(device).float().view(-1, 1)
                optimizer.zero_grad()
                outputs = model(images)
            loss = criterion(outputs, labels)
            # backprop
            fp16_scaler.scale(loss).backward()
            # update the parameters with gradient accumulation
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
            optimizer.zero_grad()
            
            train_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
            batch_idx += 1

        model.eval()
        y_true, y_pred, y_scores = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float().view(-1, 1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend((torch.sigmoid(outputs).cpu().numpy() > 0.5).astype(int))
                y_scores.extend(torch.sigmoid(outputs).cpu().numpy().flatten())

        # save the model
        if save_dir != "":
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_{epoch}.pth"))

        acc, bacc, auroc, auprc, precision, recall = classification_metrics(y_true, y_pred, y_scores)
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        # picking the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_acc": acc, "val_bacc": bacc, "val_auroc": auroc, "val_auprc": auprc, "val_precision": precision, "val_recall": recall})
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        print(f'Accuracy: {acc:.4f}, Balanced Accuracy: {bacc:.4f}, AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

def evaluate_model(model, test_loader):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(images)
            scores = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (scores > 0.5).astype(int)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)
            y_scores.extend(scores)
    acc, bacc, auroc, auprc, precision, recall = classification_metrics(y_true, y_pred, y_scores)
    wandb.log({"test_acc": acc, "test_bacc": bacc, "test_auroc": auroc, "test_auprc": auprc, "test_precision": precision, "test_recall": recall})
    print(f'Accuracy: {acc:.4f}, Balanced Accuracy: {bacc:.4f}', f'AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    return y_true, y_pred, y_scores


def train_main(dataset_dir: str, model_name="vit_small_patch14_dinov2.lvd142m", image_size=224, batch_size=32, lr=0.001, epochs=10, warmups=1, weight_decay=0.01, save_dir="outputs/", resume_ckpt="", freeze_backbone=False, exp_name="", start_epoch=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Training on device: ", device)
    # create save dir
    exp_date = datetime.now().strftime("%y-%m-%d")
    exp_code = '{}_'.format(model_name.replace("/", "_").replace(".", "_")) + f'{exp_date}' + "_" + exp_name
    save_dir = os.path.join(save_dir, exp_code)
    os.makedirs(save_dir, exist_ok=True)
    # create wandb
    writer = wandb.init(project="Solar-PV-classification-finetuning-" + exp_code)
    # create dataset
    train_csv = os.path.join(dataset_dir, "train.csv")
    val_csv = os.path.join(dataset_dir, "val.csv")
    test_csv = os.path.join(dataset_dir, "test.csv")
    # create model
    model, TRAIN_TRANSFORM, VAL_TRANSFORM = make_models_transforms(model_name, image_size, freeze_backbone)
    model = model.to(device)
    print("The number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # create loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_dataset = ImageDataset(csv_file=train_csv, transform=TRAIN_TRANSFORM)
    val_dataset = ImageDataset(csv_file=val_csv, transform=VAL_TRANSFORM)
    test_dataset = ImageDataset(csv_file=test_csv, transform=VAL_TRANSFORM)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=40)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=40)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=40)
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=epochs, warmups=warmups, lr=lr, save_dir=save_dir, resume_ckpt=resume_ckpt, start_epoch=start_epoch)
    evaluate_model(model, test_loader)
    # save model
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
