import os
import re
import json
import random
import numpy as np

import utils
from dataset.eda import *


def pre_caption(caption, max_words, is_eda=False, eda_p=0.5, re_text=False):
    if re_text:
        caption = re.sub(
            r'[^0-9a-z]+',
            ' ',
            caption.lower(),
        )

    # eda
    if is_eda and random.random() < eda_p:
        caption = eda(caption, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=1)[0]

    caption_words = caption.split()
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])
    if not len(caption):
        raise ValueError("pre_caption yields invalid text")

    return caption


def read_json_to_list(filename):
    data_list = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line.strip())
            data_list.append(item)
    print(f"from {filename} loading {len(data_list)} data")
    return data_list