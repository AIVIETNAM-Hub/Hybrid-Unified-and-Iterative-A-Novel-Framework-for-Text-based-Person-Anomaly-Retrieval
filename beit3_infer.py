import torch
from lhp.beit3 import modeling_finetune
import sys
import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import time
from torchvision import transforms
from transformers import XLMRobertaTokenizer
from timm import create_model
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Processor():
    def __init__(self, tokenizer):
        self.image_processor = transforms.Compose([
            transforms.Resize((384, 384), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])
        
        self.tokenizer = tokenizer
    
    def process(self, image=None, text=None):
        assert (image is not None) or (text is not None)
        language_tokens = None
        padding_mask = None
        if image is not None:
            image = self.image_processor(image)
            image = image.unsqueeze(0)
        if text is not None:
            language_tokens, padding_mask, _ = self.get_text_segment(text)
        return {'image': image, 'text_description': language_tokens, 'padding_mask': padding_mask}
            
        
    def get_text_segment(self, text, max_len=64):
        tokens = self.tokenizer.tokenize(text)
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.tokenizer.bos_token_id] + tokens[:] + [self.tokenizer.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        language_tokens = tokens + [self.tokenizer.pad_token_id] * (max_len - num_tokens)
        return torch.tensor([language_tokens]),  torch.tensor([padding_mask]), num_tokens

def load_model_and_processor(model_checkpoint='unilm/beit3/beit3_large_patch16_384_coco_retrieval.pth', tokenizer_checkpoint='unilm/beit3/beit3.spm'):
    checkpoint = torch.load(model_checkpoint)
    model = create_model('beit3_large_patch16_384_retrieval')
    model.load_state_dict(checkpoint['model'])
    tokenizer = XLMRobertaTokenizer(tokenizer_checkpoint)
    processor = Processor(tokenizer)
    return model, processor

    
def cosine_similarity(vector_a, vector_b):

    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    dot_product = np.dot(vector_a, vector_b)
    
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cosine_sim = dot_product / (norm_a * norm_b)
    return cosine_sim

def beit_score(image_path, text, processor, model):
    start_time = time.time()
    image_feature = encode_image(image_path, processor, model)
    text_feature = encode_text(text, processor, model)

    cs = cosine_similarity(image_feature, text_feature)
    
    end_time = time.time()
    print(end_time - start_time)
    return cs

def encode_image(image, vis_processors, model):
    try:
        raw_image = Image.open(image).convert("RGB")
    except:
        raw_image = image
    inputs = vis_processors.process(image=raw_image, text=None)
    with torch.no_grad():
        image_feature, _ = model(image=inputs['image'].to(device), only_infer=True)
    image_feature /= image_feature.norm(dim=-1, keepdim=True)
    
    return image_feature[0].cpu().numpy().astype(np.float32)                       

def encode_text(query_text, txt_processors, model):
    inputs = txt_processors.process(image=None, text=query_text)
    with torch.no_grad():
        _, text_feature = model(text_description=inputs['text_description'].to(device), padding_mask=inputs['padding_mask'].to(device), only_infer=True)
    text_feature /= text_feature.norm(dim=-1, keepdim=True)
    
    return text_feature[0].cpu().numpy().astype(np.float32)

def read_json_to_list(filename):
    data_list = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line.strip())
            data_list.append(item['caption'])
    print(f"from {filename} loading {len(data_list)} data")
    return data_list

def compare_text_and_image_testset(image_folder, text_folder, vis_processors, txt_processors, model):
    # Extract image features
    image_feats = []
    for image in tqdm(image_folder):
        image_feat = encode_image(image, vis_processors, model)
        # image_feat = image_feat / np.linalg.norm(image_feat)  # Normalize
        image_feats.append(image_feat)
    
    image_feats = np.vstack(image_feats)  # Stack into a single matrix

    # Extract text features
    text_feats = []
    for text_description in tqdm(text_folder):
        text_feat = encode_text(text_description, txt_processors, model)
        # text_feat = text_feat / np.linalg.norm(text_feat)  # Normalize
        text_feats.append(text_feat)
    
    text_feats = np.vstack(text_feats)  # Stack into a single matrix

    # Compute similarity matrix (text-to-image)
    sims_matrix = np.dot(image_feats, text_feats.T)
    sims_matrix_t2i = sims_matrix.T  # Transpose for text-to-image comparison
    
    sims_tensor = torch.from_numpy(sims_matrix_t2i)
    torch.save(sims_tensor,'score_beit.pt')

    # Get top 10 indices for each text
    indices = np.argsort(-sims_matrix_t2i, axis=1)  # Sort in descending order
    top_10_indices = indices[:, :10]  # Get top 10 indices
    print(top_10_indices)
    # Save results to file
    with open(f"{args.output_file}.txt", "w") as f:
        for i, top10 in enumerate(top_10_indices):
            string = ' '.join([image_folder[id].split('/')[-1][:-4] for id in top10])
            f.write(f"{string}\n")

    print(f"Top 10 results saved to {args.output_file}.txt")
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()

    model, processor = load_model_and_processor()
    data_folder = '/home/s48gb/Desktop/GenAI4E/pab/data/PAB/name-masked_test-set/gallery'
    image_folder = [os.path.join(data_folder,image_path) for image_path in os.listdir(data_folder)]
    annotation_path = '/home/s48gb/Desktop/GenAI4E/pab/data/PAB/name-masked_test-set/gallery/query.json'
    text_folder = read_json_to_list(annotation_path)
    compare_text_and_image_testset(image_folder, text_folder, processor, processor, model)
    



    