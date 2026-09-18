import os
import numpy as np
import segmentation_models_pytorch as smp
from glob import glob
from my_code_cleanrepo.train_eomt import train_model
from load_data import get_train_val_dataloaders
from utils.transforms import sar_transform, imagenet_transform, basic_transform
import torch
import random
from datetime import datetime


"""
Encoder list - https://smp.readthedocs.io/en/latest/encoders.html#
-resnet101 (w=imagenet) (dilation)
-timm-efficientnet-l2 (weights=noisy-student) (dilation)
-resnext101_32x8d (weights=instagram) (dilation)
-convnextv2 (dilation)
-coatnext
-maxxvitv2
-maxvit
-mit_b5 (w=imagenet) (dilation)
-pvt_v2_b5
-tu-xception71

Final list:
-ResNet101 
-efficientnet-b3
-resnext101_32x4d
-tu-convnextv2_tiny
-CoatNext
-tu-maxvit_tiny_tf_512
-mit_b3
-PvTv2

Segmentation head list - https://smp.readthedocs.io/en/latest/models.html
-UNetPlusPlus
-DeepLabV3Plus
-UPerNet
-Segformer
"""

TS = 512
PS = 20
TS_PATH = f'data/pytorch_samples/{TS}_{PS}'
REGION = None # one region or None (all)
TYPE = 'seep' # one oil slick type or all


# TRAIN_FOLDERS = [ # use this if only split by regions
#     os.path.join(TS_PATH, x, 'train') for x in os.listdir(TS_PATH) 
# ]
# VAL_FOLDERS = [ # use this if only split by regions
#     os.path.join(TS_PATH, x, 'val') for x in os.listdir(TS_PATH) 
# ]

# TRAIN_FOLDERS = [os.path.join(TS_PATH, 'northsea', 'train')]
# VAL_FOLDERS = [os.path.join(TS_PATH, 'northsea', 'val')]

RESIZE_SIZE = 384
NUM_CLASSES = 2
DATASET = f'{TS}-{PS}: slicks_world_4sept26 -ft seeps'
ENCODER = 'mit_b4'
WEIGHTS = 'imagenet'
FREEZE_BB = True
SEG_MODEL='Segformer'
BATCH_SIZE = 32
LR = '6e-5'
DIFF_LR_ENCO_DECO = False
DROPOUT_P = None
MAX_EPOCHS = 40
SUBSET = None # Int or None
LOSS_FNC = 'wcombo' # 'ce', 'wce', 'combo', 'wcombo'
CHECKPOINT = 'models/512rsz384-20_100epochs/Segformer_mit_b4_unfrz_bs18_lr2e-5_wce-loss/best_model_epoch26_dice0.882.pth'

#___________________
if not REGION:
    REGION = '*'
if not TYPE:
    TYPE = '*'
TRAIN_FOLDERS = [ # use this if split by regions and type
    x for x in glob(os.path.join(TS_PATH, REGION,'train', TYPE)) if os.path.basename(x) not in ['images','labels'] 
]

VAL_FOLDERS = [ # use this if split by regions and type
    x for x in glob(os.path.join(TS_PATH, REGION, 'val', TYPE)) if os.path.basename(x) not in ['images','labels'] 
]

if TYPE!='*':
    TRAIN_FOLDERS.extend(glob(os.path.join(TS_PATH, REGION, 'train', 'bg')))
    VAL_FOLDERS.extend(glob(os.path.join(TS_PATH, REGION, 'val', 'bg')))

seed = 24
random.seed(seed)          
np.random.seed(seed)        
torch.manual_seed(seed)       
torch.cuda.manual_seed(seed)
if FREEZE_BB:
    ff = 'frz'
else:
    ff = 'unfrz'

if __name__ == '__main__':
    device = "cuda:0"
    lr = float(LR)

    triple = True
    if triple:
        in_bands = 3
    else:
        in_bands = 1
    
    if CHECKPOINT is None:
        if not RESIZE_SIZE:
            RESIZE_SIZE=TS
        if RESIZE_SIZE==TS:
            out_log_dir = f'models/{TS}-{PS}_{MAX_EPOCHS}epochs/{SEG_MODEL}_{ENCODER}_{ff}_bs{BATCH_SIZE}_lr{LR}_{LOSS_FNC}-loss'
        else:
            out_log_dir = f'models/{TS}rsz{RESIZE_SIZE}-{PS}_{MAX_EPOCHS}epochs/{SEG_MODEL}_{ENCODER}_{ff}_bs{BATCH_SIZE}_lr{LR}_{LOSS_FNC}-loss'

        model = smp.create_model(
            arch=SEG_MODEL,                     # name of the architecture, # see args at https://smp.readthedocs.io/en/latest/models.html#deeplabv3plus
            encoder_name=ENCODER,
            encoder_weights=WEIGHTS,
            in_channels=in_bands,
            classes=NUM_CLASSES,
        ) 

    else:
        model = smp.from_pretrained(CHECKPOINT)
        out_log_dir = CHECKPOINT + '_ft2' #NOTE if path gets too long, need to use below
        # out_log_dir = f'models/{TS}rsz{RESIZE_SIZE}-{PS}_{MAX_EPOCHS}epochs/{SEG_MODEL}_{ENCODER}_{ff}_bs{BATCH_SIZE}_lr{LR}_{LOSS_FNC}-loss'

    if SUBSET is not None:
        out_log_dir = out_log_dir + f'_subset{SUBSET}'
    
    if DROPOUT_P:
        for module in model.decoder.blocks:
            module.dropout = torch.nn.Dropout2d(p=DROPOUT_P)

    print(datetime.now())
    print('out log_dir:',os.path.abspath(out_log_dir))
    # model.freeze_encoder() ## Doesnt work? https://smp.readthedocs.io/en/latest/insights.html#freezing-and-unfreezing-the-encoder
    model = model.to(device)

    # train_folder = os.path.join(TS_PATH,'train') 
    # val_folder = os.path.join(TS_PATH,'val')
    # val_folder = 'data/training_samples/pytorch_512_20_n/val' #NOTE: use same val folder as not pruned

    transform = sar_transform(resize_size=RESIZE_SIZE, triple=triple)
    # transform = basic_transform(resize_size=RESIZE_SIZE)

    train_loader, val_loader = get_train_val_dataloaders(
        path_train_dirs=TRAIN_FOLDERS,
        path_val_dirs=VAL_FOLDERS,
        batch_size=BATCH_SIZE, 
        num_workers=4,
        transform_train=transform,
        transform_val=transform,
        data_aug=True, # flips for SAR
        subset=SUBSET,
        random_seed=seed+1
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=MAX_EPOCHS,
        num_classes=NUM_CLASSES,
        loss_fnc=LOSS_FNC,
        type_scheduler='cosine_annealing',
        weight_decay=5e-4,
        monitor_metric='dice',
        log_dir=out_log_dir,
        log_dataset={'dataset log': DATASET, 'train': TRAIN_FOLDERS, 'val': VAL_FOLDERS},
        log_batchsize=BATCH_SIZE,
        log_frozen_backbone=FREEZE_BB,
        log_transform=transform.name,
        log_pretrained_weights=WEIGHTS,
        freeze_backbone=FREEZE_BB,
        lr=lr,
        label_smoothing=True,
        device=device,
        n_tensorboard_plot=min(BATCH_SIZE,24),
        diff_lr_encoder_decoder=DIFF_LR_ENCO_DECO
    )