import os
from pathlib import Path
import time
import datetime
import argparse
import json
import math
import random
import numpy as np
from ruamel.yaml import YAML
yaml = YAML(typ='safe')
from prettytable import PrettyTable

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer

from transformers import BertTokenizer

import utils
from models.model_search import Search

from dataset import create_dataset, create_sampler, create_loader
from dataset.search_dataset import TextMaskingGenerator
from scheduler import create_scheduler
from optim import create_optimizer

from train import train_model
from eval import evaluation_itm, mAP, evaluation_itc


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    world_size = utils.get_world_size()

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("### Creating model")
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    model = Search(config=config)
    if config['load_pretrained']:
        model.load_pretrained(args.checkpoint)
    model = model.to(device)
    print("Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("### Creating search dataset")
    test_dataset = create_dataset(config, evaluate=False, inference=True)

    print("### Start inferencing")
    test_loader = create_loader([test_dataset], [None],
                                batch_size=[config['batch_size_test']],
                                num_workers=[4],
                                is_trains=[False],
                                collate_fns=[None])[0]

    sims_matrix_t2i, image_embeds, text_embeds, text_atts = evaluation_itc(
        model_without_ddp, data_loader=test_loader, tokenizer=tokenizer, device=device, config=config)
    
    sims_matrix_t2i = torch.load(args.beit3_score)

    score_test_t2i = evaluation_itm(model_without_ddp, device, config, args,
                                    sims_matrix_t2i, image_embeds, text_embeds, text_atts, args.blip2_weight, args.blip2_score, args.blip2_weight, args.clip_score, args.clip_weight)

    similarity = torch.tensor(score_test_t2i)
    indices = torch.argsort(similarity, dim=1, descending=True)
    # print(indices.shape)
    # print(indices)
    
    # Get the top 10 indices
    top_10_indices = indices[:, :10]

    # Output to a text file
    with open(f"{args.output_file}", "w") as f:
        for i, top10 in enumerate(top_10_indices):
            string = ' '.join([test_dataset.g_pids[id] for id in top10])
            f.write(f"{string}\n")

    print(f"Top 10 results saved to {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--bs', default=16, type=int, help="mini batch size")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--beit3_score',type=str)
    parser.add_argument('--beit3_weight', default=0.925, type=float)
    parser.add_argument('--blip2_score', type=str)
    parser.add_argument('--blip2_weight', default=0.9, type=float)
    parser.add_argument('--clip_score', type=str)
    parser.add_argument('--clip_weight', default=0.9, type=float)

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print("### output_dir:", args.output_dir)

    config = yaml.load(open(args.config, 'r'))
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
