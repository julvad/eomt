# Create the same architecture
import torch
from timm.models.vision_transformer import vit_small_patch16_224
from my_code_cleanrepo.utils.transforms import imagenet_transform
from models.eomt import EoMT
import os
import numpy as np
from glob import glob
from my_code_cleanrepo.train_eomt import train_model
from my_code_cleanrepo.load_data import get_train_val_dataloaders
import random
from datetime import datetime


VIT_PATH = r"C:\Users\juvad3723\.1\--Projects\GitHub\clean_repo\models\ssl\dinov2_vit_small_patch16_224_testrunOlivia.pth"
transform = imagenet_transform(224)


# Create the same architecture
vit = vit_small_patch16_224(
            pos_embed="learn",
            dynamic_img_size=True,
            init_values=1e-5,
            in_chans=3,
            num_classes=0 # remove classification head
        )

# Load weights
state_dict = torch.load(VIT_PATH)

state_dict.pop("head.weight", None)
state_dict.pop("head.bias", None)

vit.load_state_dict(state_dict, strict=False)

missing, unexpected = vit.load_state_dict(
    state_dict,
    strict=False
)

eomt_model = EoMT(
    encoder=vit,
    num_classes=2,
    num_q=200,
    num_blocks=4,
)


# Dataset_____________________________________________________________
TS = 512
PS = 20
TS_PATH = rf'c:\Users\juvad3723\.1\--Projects\GitHub\clean_repo\data\pytorch_samples\{TS}_{PS}'
REGION = None # one region or None (all)
TYPE = 'seep' # one oil slick type or all
DATASET = f'{TS}-{PS}: slicks_world_4sept26 -seeps'
#_____________________________________________________________________

RESIZE_SIZE = 224
NUM_CLASSES = 2


FREEZE_BB = False
BATCH_SIZE = 24
LR = '2e-5'
DIFF_LR_ENCO_DECO = False
DROPOUT_P = None # NOTE: not implemented for eomt
MAX_EPOCHS = 40
SUBSET = None # Int or None
LOSS_FNC = 'wcombo' # 'ce', 'wce', 'combo', 'wcombo'
CHECKPOINT = None

#___________________
if not REGION:
    REGION = '*'
if not TYPE:
    TYPE = '*'
TRAIN_FOLDERS = [ # use this if split by regions and type
    x for x in glob(os.path.join(TS_PATH, REGION, 'train', TYPE)) if os.path.basename(x) not in ['images','labels'] 
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
            out_log_dir = f'my_trained_models/{TS}-{PS}_{MAX_EPOCHS}epochs/vits16_{ff}_bs{BATCH_SIZE}_lr{LR}_{LOSS_FNC}-loss'
        else:
            out_log_dir = f'my_trained_models/{TS}rsz{RESIZE_SIZE}-{PS}_{MAX_EPOCHS}epochs/vits16_{ff}_bs{BATCH_SIZE}_lr{LR}_{LOSS_FNC}-loss'

    else:
        out_log_dir = CHECKPOINT + '_ft' #NOTE if path gets too long, need to use below
        # out_log_dir = f'models/{TS}rsz{RESIZE_SIZE}-{PS}_{MAX_EPOCHS}epochs/{SEG_MODEL}_{ENCODER}_{ff}_bs{BATCH_SIZE}_lr{LR}_{LOSS_FNC}-loss'

    if SUBSET is not None:
        out_log_dir = out_log_dir + f'_subset{SUBSET}'
    
    if DROPOUT_P:
        for module in eomt_model.decoder.blocks:
            module.dropout = torch.nn.Dropout2d(p=DROPOUT_P)

    print(datetime.now())
    print('out log_dir:',os.path.abspath(out_log_dir))

    train_loader, val_loader = get_train_val_dataloaders(
        path_train_dirs=TRAIN_FOLDERS,
        path_val_dirs=VAL_FOLDERS,
        batch_size=BATCH_SIZE, 
        num_workers=4,
        transform_train=transform,
        transform_val=transform,
        data_aug=True, # flips for SAR
        subset=SUBSET,
        random_seed=seed
    )

    eomt_model = eomt_model.to(device)
    
    train_model(
        model=eomt_model,
        train_loader=train_loader,
        val_loader=val_loader,
        rsz_size=RESIZE_SIZE,
        num_epochs=MAX_EPOCHS,
        num_classes=NUM_CLASSES,
        loss_fnc=LOSS_FNC,
        type_scheduler='cosine_annealing',
        weight_decay=5e-3,
        monitor_metric='dice',
        log_dir=out_log_dir,
        log_dataset={'dataset log': DATASET, 'train': TRAIN_FOLDERS, 'val': VAL_FOLDERS},
        log_batchsize=BATCH_SIZE,
        log_frozen_backbone=FREEZE_BB,
        log_transform=transform.name,
        log_pretrained_weights=VIT_PATH,
        freeze_backbone=FREEZE_BB,
        lr=lr,
        label_smoothing=True,
        device=device,
        n_tensorboard_plot=min(BATCH_SIZE,24),
        diff_lr_encoder_decoder=DIFF_LR_ENCO_DECO
    )