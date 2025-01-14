import os
import random
from random import randint, shuffle
from random import random as rand
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

from dataset.utils import pre_caption, read_json_to_list


class TextMaskingGenerator:
    def __init__(self, tokenizer, mask_prob, mask_max, skipgram_prb=0.2, skipgram_size=3, mask_whole_word=True,
                 use_roberta=False):
        self.id2token = {i: w for w, i in tokenizer.get_vocab().items()}
        self.use_roberta = use_roberta
        for i in range(len(self.id2token)):
            assert i in self.id2token.keys()  # check
        self.cls_token_id = tokenizer.cls_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.mask_max = mask_max
        self.mask_prob = mask_prob
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word

    def get_random_word(self):
        i = randint(0, len(self.id2token) - 1)
        return i

    def __call__(self, text_ids):  # tokens: [CLS] + ...
        n_pred = min(self.mask_max, max(1, int(round(len(text_ids) * self.mask_prob))))

        # candidate positions of masked tokens
        assert text_ids[0] == self.cls_token_id
        special_pos = set([0])  # will not be masked
        cand_pos = list(range(1, len(text_ids)))

        shuffle(cand_pos)
        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end

                if self.use_roberta:
                    while (new_st > 1) and (self.id2token[text_ids[new_st].item()][0] != 'Ġ'):
                        new_st -= 1
                    while (new_end < len(text_ids)) and (self.id2token[text_ids[new_end].item()][0] != 'Ġ'):
                        new_end += 1
                else:
                    # bert, WordPiece
                    while (new_st >= 0) and self.id2token[text_ids[new_st].item()].startswith('##'):
                        new_st -= 1
                    while (new_end < len(text_ids)) and self.id2token[text_ids[new_end].item()].startswith('##'):
                        new_end += 1

                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        n_real_pred = len(masked_pos)
        if n_real_pred > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                text_ids[pos] = self.mask_token_id
            elif rand() < 0.5:  # 10%
                text_ids[pos] = self.get_random_word()

        return text_ids, masked_pos


class search_train_dataset(Dataset):
    def __init__(self, config, transform):
        self.image_root = config['image_root']
        self.transform = transform
        self.max_words = config['max_words']
        self.eda_p = config['eda_p']

        ann_file = config['train_file']
        self.ann = []
        for f in ann_file:
            anns = read_json_to_list(f)
            for item in anns:
                self.ann.append(item)

        self.img_ids = {}
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        print('image ids:', n)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        img_id = ann['image_id']

        cap = ann['caption']
        caption = pre_caption(cap, self.max_words)

        caption_eda = pre_caption(cap, self.max_words, True, self.eda_p)


        return image, caption, caption_eda, self.img_ids[img_id]


class search_test_dataset(Dataset):
    def __init__(self, config, transform):
        ann_file = config['test_file']
        self.transform = transform
        self.image_root = config.get('image_root_test', config['image_root'])
        self.max_words = config['max_words']

        self.ann = read_json_to_list(ann_file)
        self.text = []
        self.image = []
        self.g_pids = []
        self.q_pids = []
        for img_id, ann in enumerate(self.ann):
            self.g_pids.append(ann['image_id'])
            self.image.append(ann['image'])
            for i, caption in enumerate(ann['caption']):
                self.q_pids.append(ann['image_id'])
                self.text.append(pre_caption(caption, self.max_words))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index


class search_inference_dataset(Dataset):
    def __init__(self, config, transform):
        ann_file = config['test_file']
        self.transform = transform
        self.image_root = config.get('image_root_test', config['image_root'])
        self.max_words = config['max_words']

        self.ann = read_json_to_list(ann_file)
        self.text = []
        self.image = []
        self.g_pids = []
        self.q_pids = []
        for img_id, file in enumerate(os.listdir(self.image_root)):
            self.g_pids.append(file.split('.')[0])
            self.image.append(file)
            
        for i, ann in enumerate(self.ann):
            self.q_pids.append(ann['query_index'])
            self.text.append(pre_caption(ann['caption'], self.max_words))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image[index])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index