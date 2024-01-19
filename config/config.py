import os
from utils.utils import custom_print

mode = "train"
# mode = "test"

root_dir = "D:\\Desktop\zongliang\\processed_data\\our_busv"
train_data_dir = root_dir + "\\train"
val_data_dir = [
    root_dir + "\\val",
]
image_dir_name = "image"
mask_dir_name = "mask"
image_ext = [".jpg", ".jpg", ".jpg", ".jpg"]
mask_ext = [".png", ".bmp", ".jpg", ".png"]

vgg16_ckp = "./checkpoints/vgg16_bn_feat.pth"
image_size = 224
learning_rate = 2e-5
learning_rate_decay = 20000
batch_size = 4
group_size = 1
epochs = 100000
log_interval = 100
val_interval = 1000

log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, "log.txt")
custom_print("$" * 50 + " start log " + "$" * 50, log_file, "w")

model_save_dir = "checkpoints"
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
model_train_last = os.path.join(model_save_dir, "_last.pth")
model_train_best = os.path.join(model_save_dir, "_best.pth")
