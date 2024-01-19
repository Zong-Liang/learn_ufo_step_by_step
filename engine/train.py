from loss.loss import Loss
from torch.optim import Adam
from utils.utils import custom_print
import datetime
import torch
from engine.val import val_step
from torch.utils.data import DataLoader
from load_data.load_data_1 import VideoDataset
from config.config import *


def train_step(
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
):
    optimizer = Adam(net.parameters(), learning_rate, weight_decay=1e-6)
    train_loader = DataLoader(
        VideoDataset(train_data_dir, epochs * batch_size, image_size, group_size),
        num_workers=0,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=False,
    )
    loss = Loss().to(device)
    best_d = 0
    ave_loss, ave_i_loss, ave_m_loss, ave_c_loss, ave_s_loss = 0, 0, 0, 0, 0
    epoch = 0
    for data, mask in train_loader:
        epoch += 1
        # print(epoch)

        data = data.view(-1, data.shape[2], data.shape[3], data.shape[4])
        mask = mask.view(-1, mask.shape[2], mask.shape[3])
        # print(data.shape)
        # print(mask.shape)

        img, cls_gt, mask_gt = data, torch.rand(batch_size, 78), mask

        net.zero_grad()
        img, cls_gt, mask_gt = img.to(device), cls_gt.to(device), mask_gt.to(device)
        pred_cls, pred_mask = net(img)
        # print(pred_mask.shape)
        # print(pred_cls.shape)

        all_loss, i_loss, m_loss, c_loss, s_loss = loss(
            pred_mask, mask_gt, pred_cls, cls_gt
        )
        all_loss.backward()
        epoch_loss = all_loss.item()
        i_l = i_loss.item()
        m_l = m_loss.item()
        c_l = c_loss.item()
        s_l = s_loss.item()
        ave_loss += epoch_loss
        ave_i_loss += i_l
        ave_m_loss += m_l
        ave_c_loss += c_l
        ave_s_loss += s_l
        optimizer.step()

        if epoch % log_interval == 0:
            # print(img.shape)
            # print(mask_gt.shape)
            # print(pred_cls.shape)
            # print(pred_mask.shape)

            ave_loss = ave_loss / log_interval
            ave_i_loss = ave_i_loss / log_interval
            ave_m_loss = ave_m_loss / log_interval
            ave_c_loss = ave_c_loss / log_interval
            ave_s_loss = ave_s_loss / log_interval
            custom_print(
                datetime.datetime.now().strftime("%F %T")
                + " lr: %e | epoch: [%d/%d] | all_loss: [%.4f] | i_loss: [%.4f] | m_loss: [%.4f] | c_loss: "
                "[%.4f] | s_loss: [%.4f]"
                % (
                    learning_rate,
                    epoch,
                    epochs,
                    ave_loss,
                    ave_i_loss,
                    ave_m_loss,
                    ave_c_loss,
                    ave_s_loss,
                ),
                log_file,
                "a+",
            )

            ave_loss, ave_i_loss, ave_m_loss, ave_c_loss, ave_s_loss = 0, 0, 0, 0, 0

        if epoch % val_interval == 0:
            net.eval()
            with torch.no_grad():
                custom_print(
                    datetime.datetime.now().strftime("%F %T")
                    + " now is evaluating our_busv_1: ",
                    log_file,
                    "a+",
                )
                ave_d, ave_i = val_step(
                    net,
                    val_data_dir,
                    device,
                    group_size,
                    image_size,
                    image_dir_name,
                    mask_dir_name,
                    image_ext,
                    mask_ext,
                )
                if ave_d[0] > best_d:
                    # follow your save condition
                    best_d = ave_d[0]
                    torch.save(net.state_dict(), model_train_best)
                torch.save(net.state_dict(), model_train_last)

                custom_print("-" * 100, log_file, "a+")
                custom_print(
                    datetime.datetime.now().strftime("%F %T")
                    + " [our_busv]  dice: [%.4f] | iou: [%.4f] | the best dice: [%.4f]"
                    % (ave_d[0], ave_i[0], best_d),
                    log_file,
                    "a+",
                )
                custom_print("-" * 100, log_file, "a+")
            net.train()

        if epoch % learning_rate_decay == 0:
            optimizer = Adam(net.parameters(), learning_rate / 2, weight_decay=1e-6)
            learning_rate = learning_rate / 2
