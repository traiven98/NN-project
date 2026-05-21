import os

DATASET_PATH    = os.path.join(os.path.dirname(__file__), '..', 'data')
CSV_PATH        = os.path.join(DATASET_PATH, 'HAM10000_metadata.csv')
IMG_DIRS        = [
    os.path.join(DATASET_PATH, 'HAM10000_images_part_1'),
    os.path.join(DATASET_PATH, 'HAM10000_images_part_2'),
]

IMG_SIZE        = 224
BATCH_SIZE      = 32
NUM_CLASSES     = 7
EPOCHS_FROZEN   = 5
EPOCHS_FINETUNE = 15
LR_FROZEN       = 1e-3
LR_FINETUNE     = 3e-5
SEED            = 42

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_LABELS = {
    'akiec': 'Actinic keratosis',
    'bcc':   'Basal cell carcinoma',
    'bkl':   'Benign keratosis',
    'df':    'Dermatofibroma',
    'mel':   'Melanoma',
    'nv':    'Melanocytic nevus',
    'vasc':  'Vascular lesion',
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CHECKPOINTS_DIR = 'checkpoints'
PLOTS_DIR       = 'plots'
