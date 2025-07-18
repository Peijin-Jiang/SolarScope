# config.py

# Input data settings
dataset_csv = "/content/drive/MyDrive/solar_PV_prediction/src/dataset_csv/20241013"

# Model settings
model_name = "segformer-b5"# deeplabv3-resnet101", "sam""segformer-b5""unet"
loss = "Dice+BCE"  # Options: ["BCE", "Dice", "Combo", "Focal", "Dice+Focal", "Dice+BCE"]

# Training settings
seed = 0
device = "cuda"  # Options: ['cuda', 'cpu']
epochs = 200
warmup_epochs = 5
batch_size = 2
max_len = 256
training_ratio = 1 # Training ratio
lr = 2e-4  # Learning rate
min_lr = 1e-6  # Minimum learning rate
lr_scheduler = "cosine"  # Options: ['cosine', 'fixed']
augmentation = True # Whether to use augmentation
gc = 1  # Gradient accumulation
folds = 0  # Number of folds
prop = 1  # Proportion of training data
optim = "adamw"  # Options: ['adam', 'adamw']
optim_wd = 0  # Weight decay
save_dir = "/content/drive/MyDrive/solar_PV_prediction/outputs_segAug"
report_to = 'wandb'  # Options: ['wandb', 'tensorboard', None]
resume_ckpt = "/content/drive/MyDrive/solar_PV_prediction/outputs_segAug/segformer-b5_25-06-06_lr_5e-05_opt_adamw_bsz_4_ratio_1_loss_Dice+BCE/model_19.pth" # Resume the previous checkpointing
workers = 12  # Number of dataloader workers