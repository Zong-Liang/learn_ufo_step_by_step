import torch
from torch import nn
from models.video_model import build_model, weights_init
from config.config import *
from engine.train import train_step
import os
from datetime import datetime

torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if mode == "train":
    device = torch.device("cuda")
    net = build_model(device).to(device)
    net.apply(weights_init)
    net.base.load_state_dict(torch.load(vgg16_ckp))
    net = nn.DataParallel(net, device_ids=[0])
    net.train()

    print(datetime.now().strftime("%F %T") + " start training on our_busv: ")
    train_step(
        net,
        train_data_dir,
        device,
        batch_size,
        log_file,
        val_data_dir,
        model_train_best,
        model_train_last,
        learning_rate,
        learning_rate_decay,
        epochs,
        log_interval,
        val_interval,
    )
    print(datetime.now().strftime("%F %T") + " done!")
