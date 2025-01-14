import math

import torch
from torch.cuda.amp import autocast

import utils
import os
import json



def mlm(text, text_input, tokenizer, device, mask_generator, config):
    text_masked = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                            return_tensors="pt").to(device)
    text_ids_masked = text_masked.input_ids
    masked_pos = torch.empty((text_ids_masked.shape[0], config['max_masks']), dtype=torch.int64, device=device)
    masked_ids = torch.empty((text_ids_masked.shape[0], config['max_masks']), dtype=torch.long, device=device)
    for index, text_id in enumerate(text_ids_masked):
        text_ids_masked_, masked_pos_ = mask_generator(text_id)
        masked_ids_ = [text_input.input_ids[index][p].item() for p in masked_pos_]
        n_pad = config['max_masks'] - len(masked_ids_)
        masked_pos_ = masked_pos_ + [0] * n_pad
        masked_pos_ = torch.tensor(masked_pos_, dtype=torch.int64).to(device)
        masked_ids_ = masked_ids_ + [-100] * n_pad
        masked_ids_ = torch.tensor(masked_ids_, dtype=torch.long).to(device)
        masked_pos[index] = masked_pos_
        masked_ids[index] = masked_ids_
    return text_ids_masked, masked_pos, masked_ids


def train_model(model, data_loader, optimizer, scaler, tokenizer, epoch, device, scheduler, config, mask_generator, log_folder):
    os.makedirs(log_folder, exist_ok=True)
    model.train()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mim', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    log_list = []
    for i, (image, text, text_eda, idx) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        image = image.to(device, non_blocking=True)
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)
        text_ids_masked, masked_pos, masked_ids = mlm(text, text_input, tokenizer, device, mask_generator, config)

        text_input_eda = tokenizer(text_eda, padding='max_length', truncation=True, max_length=config['max_tokens'],
                                   return_tensors="pt").to(device)
        text_ids_eda = text_input_eda.input_ids
        text_atts_eda = text_input_eda.attention_mask

        idx = idx.to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss_itc, loss_itm, loss_mlm, loss_mim = \
                model(image, text_input.input_ids, text_input.attention_mask,
                      text_ids_masked=text_ids_masked, masked_pos=masked_pos, masked_ids=masked_ids,
                      idx=idx, text_ids_eda=text_ids_eda, text_atts_eda=text_atts_eda,)
            loss = loss_itc + loss_itm + loss_mlm + loss_mim * 0.1356

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()
        skip_lr_sched = (scale > scaler.get_scale())
        if not skip_lr_sched:
            scheduler.step()
        optimizer.zero_grad()

        metric_logger.update(loss_itc=loss_itc.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_mim=loss_mim.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if (i % 50==0):
            log_list.append({
                'loss_itc': loss_itc.item(),
                'loss_itm': loss_itm.item(),
                'loss_mlm': loss_mlm.item(),
                'loss_mim': loss_mim.item(),
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]["lr"]
            })
            with open(f'{log_folder}/{epoch+1}.json', 'w') as f:
                json.dump(log_list, f, indent=4)
        
    log_list.append({
        "Averaged stats:": metric_logger.global_avg()
    })
    
    with open(f'{log_folder}/{epoch+1}.json', 'w') as f:
        json.dump(log_list, f, indent=4)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

