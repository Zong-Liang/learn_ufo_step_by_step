import os
from utils.utils import custom_print
from pprint import pprint


mode = "train"
# mode = "test"

root_dir = r"D:\Desktop\zongliang\processed_data\our_busv_1"
train_data_dir = root_dir + r"\train"
val_data_dir = [
    root_dir + r"\val",
]
image_dir_name = "image"
mask_dir_name = "mask"
image_ext = [".jpg", ".jpg", ".jpg", ".jpg"]
mask_ext = [".png", ".bmp", ".jpg", ".png"]

vgg16_pth = r"D:\Desktop\zongliang\paper_code\learn_ufo_step_by_step\checkpoints\vgg16_bn_feat.pth"
image_size = 224
learning_rate = 2e-5
learning_rate_decay = 20000
batch_size = 4
group_size = 5
epochs = 100000
log_interval = 100
val_interval = 1000

log_dir = r"D:\Desktop\zongliang\paper_code\learn_ufo_step_by_step\logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, "log.txt")
custom_print("#" * 50 + " start log " + "#" * 50, log_file, "w")

model_save_dir = r"D:\Desktop\zongliang\paper_code\learn_ufo_step_by_step\checkpoints"
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
model_train_last = os.path.join(model_save_dir, "model_last.pth")
model_train_best = os.path.join(model_save_dir, "model_best.pth")

if __name__ == "__main__":
    print("#" * 50 + " in image_dir " + "#" * 50)
    print(train_data_dir)
    print(os.path.join(train_data_dir, os.listdir(train_data_dir)[0]))
    pprint(os.listdir(os.path.join(train_data_dir, os.listdir(train_data_dir)[0])))
    print("#" * 50 + " in mask_dir " + "#" * 50)
    pprint(os.listdir(os.path.join(train_data_dir, os.listdir(train_data_dir)[1])))
    print(os.path.join(train_data_dir, os.listdir(train_data_dir)[1]))
