import os
import cv2
import math
import numpy as np
import wandb
import torch
import data, models, losses, metrics # type: ignore

from tqdm import tqdm
from torch.utils.data import DataLoader


def get_criterion(args):
    criteria = {
        "bce": losses.BCECriterion(),
        "dice": losses.DiceCriterion(),
        "focal": losses.FocalCriterion(),
        "mse": losses.MSECriterion(),
        "dice+bce": losses.ComboDiceBCECriterion(),
        "dice+focal": losses.ComboDiceFocalCriterion(),
    }
    assert args.loss.lower() in criteria, f"Unknown loss type {args.loss}."
    criterion = criteria[args.loss.lower()]
    return criterion


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def log_image_samples(writer, split, imgs, labels, predictions, image_size):
    cnt = min(len(imgs), 8)
    writer.log(
        {
            f"examples/inputs/{split}": [
                wandb.Image(cv2.resize(
                    np.transpose(imgs[i].detach().cpu().numpy(), axes=(1, 2, 0)), (image_size, image_size),
                ), caption=f"input {i}") for i in range(cnt)
            ],
            f"examples/labels/{split}": [
                wandb.Image(labels[i].view(image_size, image_size, 1).detach().cpu().numpy(), caption=f"target {i}") for i in range(cnt)
            ],
            f"examples/predictions/{split}": [
                wandb.Image(predictions[i].view(image_size, image_size, 1).detach().cpu().numpy(), caption=f"prediction {i}") for i in range(cnt)
            ]
        }
    )


def train_one_epoch(model, train_dl, epoch, criterion, optimizer, args):
    model.train()
    train_stats = {}
    preds_gather, labels_gather = [], []

    for i, batch in enumerate(tqdm(train_dl, desc="Start training the model for one epoch...")):

        # adjust the learning rate
        adjust_learning_rate(optimizer, float(i) / len(train_dl) + epoch, args)

        if "sam" in args.model_name:
          outputs = model(
              pixel_values=batch["pixel_values"].to(args.device),
              input_boxes=batch["input_boxes"].to(args.device),
              multimask_output=False,
          )
          predicted_masks = outputs.pred_masks.squeeze()
        elif "segformer" in args.model_name:
          outputs = model(batch["pixel_values"].to(args.device))
          predicted_masks = outputs.logits.squeeze()
        else:
          outputs = model(batch["pixel_values"].to(args.device))
          predicted_masks = outputs.squeeze()

        ground_truth_masks = batch["ground_truth_mask"].float().to(args.device)
        if len(predicted_masks.shape) == 2:
          predicted_masks = predicted_masks.unsqueeze(0)
        loss_info = criterion(predicted_masks, ground_truth_masks)
        loss = loss_info["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if len(train_stats) == 0:  # first batch
            train_stats = {k: v.item() for k,v in loss_info.items()}
        else:
            train_stats = {k: train_stats[k] + loss_info[k].item() for k in train_stats}

        if (i + 1) % 20 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Batch {i+1} | Loss: {loss_info['loss'].item():.4f} | Learning rate: {lr: .6f}")

        # preds_gather.append(predicted_masks.detach().cpu())
        # labels_gather.append(ground_truth_masks.detach().cpu())

    train_stats = {k: train_stats[k] / len(train_dl) for k in train_stats}
    # preds_gather = torch.cat(preds_gather, dim=0) > 0.0  # also in logit space
    # labels_gather = torch.cat(labels_gather, dim=0).bool()

    return train_stats["loss"], train_stats


def eval_one_epoch(model, val_dl, criterion, epoch, writer, image_size, args):
    model.eval()
    val_stats = {}
    preds_gather, labels_gather = [], []

    with torch.no_grad():
        for batch_idx, val_batch in tqdm(enumerate(val_dl)):
            if "sam" in args.model_name:
              outputs = model(
                  pixel_values=val_batch["pixel_values"].to(args.device),
                  input_boxes=val_batch["input_boxes"].to(args.device),
                  multimask_output=False,
              )
              predicted_masks = outputs.pred_masks.squeeze()
            elif "segformer" in args.model_name:
              outputs = model(val_batch["pixel_values"].to(args.device))
              predicted_masks = outputs.logits.squeeze()
            else:
              outputs = model(val_batch["pixel_values"].to(args.device))
              predicted_masks = outputs.squeeze()
            ground_truth_masks = val_batch["ground_truth_mask"].float().to(args.device)
            if len(predicted_masks.shape) == 2:
              predicted_masks = predicted_masks.unsqueeze(0)
            loss_info = criterion(predicted_masks, ground_truth_masks)

            if len(val_stats) == 0:  # first batch
                val_stats = {k: v.item() for k, v in loss_info.items()}
            else:
                val_stats = {k: val_stats[k] + loss_info[k].item() for k in val_stats}

            if batch_idx == 0 and args.report_to == "wandb":
                predicted_masks_ = torch.nn.functional.interpolate(predicted_masks.unsqueeze(1), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze()
                ground_truth_masks_ = torch.nn.functional.interpolate(ground_truth_masks.unsqueeze(1), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze()
                log_image_samples(writer, "val" if epoch != -1 else "test", val_batch["pixel_values"], ground_truth_masks_, predicted_masks_, image_size=image_size)

            preds_gather.append(predicted_masks.detach().cpu())
            labels_gather.append(ground_truth_masks.detach().cpu())

    val_stats = {k: val_stats[k] / len(val_dl) for k in val_stats}
    preds_gather = (torch.cat(preds_gather, dim=0) > 0.0).cpu().numpy()
    labels_gather = torch.cat(labels_gather, dim=0).bool().cpu().numpy()
    seg_metrics = metrics.segmentation_metrics(preds_gather, labels_gather)

    for met in seg_metrics:
        val_stats[met] = seg_metrics[met].item()
    return val_stats["loss"], val_stats


def train_model(train_path, val_path, test_path, writer, args):

    model, processor, image_size, mask_size = models.load_model(args.model_name, args.device)
    train_data = data.SegmentationDataset(train_path, image_size=image_size, mask_size=mask_size, transform=processor, augmentation=args.augmentation, model_name=args.model_name)
    val_data = data.SegmentationDataset(val_path, image_size=image_size, mask_size=mask_size, transform=processor, model_name=args.model_name)
    test_data = data.SegmentationDataset(test_path, image_size=image_size, mask_size=mask_size, transform=processor, model_name=args.model_name)
    # test_data[0]
    # raise
    criterion = get_criterion(args)
    if "sam" in args.model_name:
      optimizer = torch.optim.Adam(model.mask_decoder.parameters(), lr=args.lr, weight_decay=0)
    else:
      optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0) # finetune all parameters

    train_dl = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    test_dl = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    best_val_loss, best_val_model = None, None

    # if resume the checkpoint
    start_epoch = 0
    if args.resume_ckpt:
      start_epoch = int(args.resume_ckpt.split("model_")[-1].replace(".pth", "")) + 1
      resume_ckpt = torch.load(args.resume_ckpt)
      model.load_state_dict(resume_ckpt)
      print(f"Resume from previous checkpoint: {args.resume_ckpt}, starting from {start_epoch}...")

    best_model_state = model.state_dict()
    for i in range(start_epoch, args.epochs):
        # train
        train_loss, train_stats = train_one_epoch(model, train_dl, i, criterion, optimizer, args)
        # validation
        val_loss, val_stats = eval_one_epoch(model, val_dl, criterion, i, writer, image_size, args)
        if best_val_loss is None or val_loss < best_val_loss:
          best_val_loss = val_loss
          best_model_state = model.state_dict()
        # print the loss
        print(f"Epoch: {i}, train loss: {train_loss}, val_loss: {val_loss}" + ", val_dice: {}, val_IoU: {}".format(val_stats["dice"], val_stats["iou"]))
        # log train/val stats
        if args.report_to == "wandb":
          log_stats = {"train_" + k: v for k, v in train_stats.items()}
          log_stats.update({"val_" + k: v for k, v in val_stats.items()})
          log_stats["epoch"] = i
          wandb.log(log_stats)
        # save the model
        model_dir = os.path.join(args.save_dir, args.exp_code)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_dir, 'model_{}.pth'.format(i)))
    
    # start evaluation on test
    model.load_state_dict(best_model_state)
    test_loss, test_stats = eval_one_epoch(model, test_dl, criterion, -1, writer, image_size, args)
    print(f"Epoch: {args.epochs}, test_loss: {test_loss}" + ", test_dice: {}, test_IoU: {}".format(test_stats["dice"], test_stats["iou"]))
    if args.report_to == "wandb":
      wandb.log({"test_" + k: v for k, v in test_stats.items()})
