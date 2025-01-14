import sys
sys.path.append('./blip')
from blip.blip2 import init_model
import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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

def blip_score(image_path, text, vis_processors, txt_processors, model):
    start_time = time.time()
    image_feature = encode_image(image_path, vis_processors, model)
    text_feature = encode_text(text, txt_processors, model)

    cs = cosine_similarity(image_feature, text_feature)
    
    end_time = time.time()
    print(end_time - start_time)
    return cs

def encode_image(image_path, vis_processors, model):
    raw_image = Image.open(image_path).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    image_features = model.extract_features({"image": image}, mode="image")
    image_feature = image_features.image_embeds_proj
    image_feature /= image_feature.norm(dim=-1, keepdim=True)
    image_feature = image_feature.squeeze(dim=0)

    return image_feature[0].cpu().numpy().astype(np.float32)                        

def encode_text(query_text, txt_processors, model):

    processed_text = txt_processors["eval"](query_text)
    text_features = model.extract_features(
        {"text_input": [processed_text]},
        mode = "text"
    )
    text_features = text_features.text_embeds_proj[:, 0, :]
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    return text_features.cpu().numpy().astype(np.float32)[0]

def read_json_to_list(filename):
    data_list = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line.strip())
            data_list.append(item['caption'])
    print(f"from {filename} loading {len(data_list)} data")
    return data_list

def compare_text_and_image_testset(image_folder, text_folder, vis_processors, txt_processors, model):
    image_feats = []
    for image in tqdm(image_folder):
        image_feat = encode_image(image, vis_processors, model)
        image_feats.append(image_feat)
    
    image_feats = np.vstack(image_feats)

    text_feats = []
    for text_description in tqdm(text_folder):
        text_feat = encode_text(text_description, txt_processors, model)
        text_feats.append(text_feat)
    
    text_feats = np.vstack(text_feats)

    sims_matrix = np.dot(image_feats, text_feats.T)
    sims_matrix_t2i = sims_matrix.T 
    
    sims_tensor = torch.from_numpy(sims_matrix_t2i)
    torch.save(sims_tensor,args.save_score)

    indices = np.argsort(-sims_matrix_t2i, axis=1) 
    top_10_indices = indices[:, :10]  # Get top 10 indices

    # Save results to file
    with open(f"{args.output_file}", "w") as f:
        for i, top10 in enumerate(top_10_indices):
            string = ' '.join([image_folder[id].split('/')[-1][:-4] for id in top10])
            f.write(f"{string}\n")

    print(f"Top 10 results saved to {args.output_file}")
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default = './data/PAB/name-masked_test-set/gallery', type=str)
    parser.add_argument('--annotation', default = './data/PAB/name-masked_test-set/query.json', type=str)
    parser.add_argument('--save_score', default = './sims_score/score_blip2_reproduce.pt', type=str)
    parser.add_argument('--output_file', default = './predictions/score_blip2.txt', type=str)
    args = parser.parse_args()

    model, vis_processors, txt_processors = init_model()
    image_folder = [os.path.join(args.image_folder,image_path) for image_path in os.listdir(args.image_folder)]
    annotation_path = args.annotation
    text_folder = read_json_to_list(annotation_path)

    compare_text_and_image_testset(image_folder, text_folder, vis_processors, txt_processors, model)
    



