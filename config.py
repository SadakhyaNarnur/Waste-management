import torch
import os

BATCH_SIZE = 16 # increase / decrease according to GPU memeory
RESIZE_TO = 512 # resize the image for training and transforms
NUM_EPOCHS = 50 # number of epochs to train for

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DATA_PATH = os.path.join(os.path.dirname(__file__),"../data")
# training images and XML files directory
TRAIN_DIR = os.path.join(DATA_PATH,'train')
# validation images and XML files directory
VALID_DIR = os.path.join(DATA_PATH,'valid')

CLASSES = [
    'background','paper','glass','plastic','cardboard','metal','trash'
]
NUM_CLASSES = 7

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False
# location to save model and plots
OUT_DIR = os.path.join(os.path.dirname(__file__),"../outputs")
SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs