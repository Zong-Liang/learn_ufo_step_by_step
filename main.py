import torch
from torch import nn
from models.model import build_model, weights_init
from config.config import *
from engine.train import train_step
from engine.test import test_step
import os
from datetime import datetime

torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if mode == "train":
    net = build_model(device).to(device)
    net.apply(weights_init)
    net.base.load_state_dict(torch.load(vgg16_pth))
    net = nn.DataParallel(net, device_ids=[0])
    net.train()

    print(
        "$" * 50 + " " + datetime.now().strftime("%F %T") + " start train " + "$" * 50
    )
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
    print("$" * 50 + " " + datetime.now().strftime("%F %T") + " done! " + "$" * 50)

elif mode == "test":
    net = build_model(device).to(device)
    net = torch.nn.DataParallel(net, device_ids=[0])
    net.load_state_dict(torch.load(model_best_pth))
    net.eval()

    print("$" * 50 + " " + datetime.now().strftime("%F %T") + " start test " + "$" * 50)
    test_step(
        device,
        net,
        test_data_dir,
        test_output_dir,
        group_size,
        image_size,
        image_dir_name,
    )
    print("$" * 50 + " " + datetime.now().strftime("%F %T") + " done! " + "$" * 50)
else:
    print("wrong mode! Your mode must be 'train' or 'test'!")
