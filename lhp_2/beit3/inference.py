import os
import numpy as np
import pandas as pd
import torch
import csv
from PIL import Image
from tqdm import tqdm
import modeling_finetune
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from transformers import XLMRobertaTokenizer
from timm import create_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import argparse
import time
import torch.nn.functional as F
import json

parser = argparse.ArgumentParser(description='Inference BEiT3')
parser.add_argument('--checkpoint', default = './checkpoint/lhp/lhp_beit3.pth', type=str)
parser.add_argument('--tokenizer', default = './checkpoint/lhp/beit3.spm', type=str)
parser.add_argument('--image_folder', default = './data/PAB/name-masked_test-set/gallery', type=str)
parser.add_argument('--annotation', default = './data/PAB/name-masked_test-set/query.json', type=str)
parser.add_argument('--save_score', default = './sims_score/score_beit3_reproduce.pt', type=str)
parser.add_argument('--output_file', default = './predictions/score_beit3.txt', type=str)
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(args.checkpoint)
model = create_model('beit3_large_patch16_384_retrieval')
model.load_state_dict(checkpoint['model'])
model.eval().to(device)
tokenizer = XLMRobertaTokenizer(args.tokenizer)
print('Load model successfully')


transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])
loader = default_loader

# Function to extract features from an image
def extract_image_features(image_path):
    image = loader(image_path)
    image = transform(image)
    # print(image.shape)
    image = image.unsqueeze(0).to(device)
    # print(image.shape)
    with torch.no_grad():
        image_feats, _ = model(image=image, only_infer=True)
        
    return image_feats
    
def get_text_segment(text, max_len=64):
    tokens = tokenizer.tokenize(text)
    tokens = tokenizer.convert_tokens_to_ids(tokens)

    if len(tokens) > max_len - 2:
        tokens = tokens[:max_len - 2]

    tokens = [tokenizer.bos_token_id] + tokens[:] + [tokenizer.eos_token_id]
    num_tokens = len(tokens)
    padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
    language_tokens = tokens + [tokenizer.pad_token_id] * (max_len - num_tokens)
    return torch.tensor([language_tokens]),  torch.tensor([padding_mask]), num_tokens

def extract_text_features(text):
    language_tokens, padding_mask, _ = get_text_segment(text)
    language_tokens = language_tokens.to(device)
    padding_mask = padding_mask.to(device)
    
    with torch.no_grad():
        _, language_cls = model(text_description=language_tokens, padding_mask=padding_mask, only_infer=True)
    
    return language_cls


def read_csv(csv_file):
    image_paths = []
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            image_paths.append(row[0])
    return image_paths

def compare_text_and_image_features(image_path, text_description):
    image_features = extract_image_features(image_path)
    text_features = extract_text_features(text_description)
    
    image_features = image_features / np.linalg.norm(image_features) 
    text_features = text_features / np.linalg.norm(text_features)
    
    similarity = np.dot(image_features, text_features.T)  
    
    return similarity

def read_json_to_list(filename):
    data_list = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line.strip())
            data_list.append(item['caption'])
    print(f"from {filename} loading {len(data_list)} data")
    return data_list

def compare_text_and_image_testset(image_folder, text_folder):
    image_feats = []
    for image in tqdm(image_folder):
        image_feat = extract_image_features(image)
        image_feats.append(image_feat)
    
    image_feats = torch.cat(image_feats, dim=0)
    
    text_feats = []
    for text_description in tqdm(text_folder):
        text_feat = extract_text_features(text_description)
        text_feats.append(text_feat)
    
    text_feats = torch.cat(text_feats, dim=0)
    
    sims_matrix = image_feats @ text_feats.t()
    sims_matrix_t2i = sims_matrix.t()
    
    similarity = torch.tensor(sims_matrix_t2i)
    
    if args.save_score is not None:
        torch.save(similarity, args.save_score)
    
    indices = torch.argsort(similarity, dim=1, descending=True)
    
    top_10_indices = indices[:, :10]
    
    with open(f"{args.output_file}", "w") as f:
        for i, top10 in enumerate(top_10_indices):
            string = ' '.join([image_folder[id].split('/')[-1][:-4] for id in top10])
            f.write(f"{string}\n")

    print(f"Top 10 results saved to {args.output_file}")
    return None
    
image_folder = [os.path.join(args.image_folder,image_path) for image_path in os.listdir(args.image_folder)]
annotation_path = args.annotation
text_folder = read_json_to_list(annotation_path)

compare_text_and_image_testset(image_folder, text_folder)

