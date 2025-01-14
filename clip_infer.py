import torch
import clip
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import time
import os
import argparse
device = "cuda" if torch.cuda.is_available() else "cpu"

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

def clip_score(model, preprocess, image_path, text):
    start_time = time.time()
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize([text], context_length=77, truncate=True).to(device)
    image_features = encode_image(model, image)
    text_features = encode_text(model, text)

    cs = cosine_similarity(image_features[0], text_features[0])
    
    end_time = time.time()
    print(end_time - start_time )
    return cs

def encode_image(model, image):
    with torch.no_grad():
        image_features = model.encode_image(image).cpu().numpy().astype(np.float32)
    return image_features

def encode_text(model, text):
    with torch.no_grad():
        text_features = model.encode_text(text).cpu().numpy().astype(np.float32)
    return text_features

def read_json_to_list(filename):
    data_list = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line.strip())
            data_list.append(item['caption'])
    print(f"from {filename} loading {len(data_list)} data")
    return data_list

def compare_text_and_image_testset(image_folder, text_folder, output_file, preprocess, model):
    # Extract image features
    image_feats = []
    for image in tqdm(image_folder):
        image = preprocess(Image.open(image)).unsqueeze(0).to(device)
        image_feat = encode_image(model, image)
        image_feat = image_feat / np.linalg.norm(image_feat)  # Normalize
        image_feats.append(image_feat)
    
    image_feats = np.vstack(image_feats)  # Stack into a single matrix

    # Extract text features
    text_feats = []
    for text_description in tqdm(text_folder):
        text = clip.tokenize([text_description], context_length=77, truncate=True).to(device)
        text_feat = encode_text(model, text)
        text_feat = text_feat / np.linalg.norm(text_feat)  # Normalize
        text_feats.append(text_feat)
    
    text_feats = np.vstack(text_feats)  # Stack into a single matrix

    # Compute similarity matrix (text-to-image)
    sims_matrix = np.dot(image_feats, text_feats.T)
    sims_matrix_t2i = sims_matrix.T  # Transpose for text-to-image comparison
    
    sims_tensor = torch.from_numpy(sims_matrix_t2i)
    torch.save(sims_tensor,args.save_score)

    # Get top 10 indices for each text
    indices = np.argsort(-sims_matrix_t2i, axis=1)  # Sort in descending order
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
    parser.add_argument('--save_score', default = './sims_score/score_clip_reproduce.pt', type=str)
    parser.add_argument('--output_file', default = './predictions/score_clip.txt', type=str)
    args = parser.parse_args()
    
    # data_folder = '/home/s48gb/Desktop/GenAI4E/pab/WWW2025Workshop_mimcode/data/PAB/name-masked_test-set/gallery'
    image_folder = [os.path.join(args.image_folder,image_path) for image_path in os.listdir(args.image_folder)]
    annotation_path = args.annotation
    text_folder = read_json_to_list(annotation_path)
    
    model, preprocess = clip.load("ViT-L/14@336px", device=device)   
    compare_text_and_image_testset(image_folder, text_folder, output_file=args.output_file, preprocess=preprocess, model=model) 
    # image_path = '/home/s48gb/Desktop/GenAI4E/pab/WWW2025Workshop_mimcode/data/PAB/train/imgs_0/goal/2.jpg'
    # text = "A man with short brown hair and a beard is holding a black and white cat. He is wearing a blue t-shirt with a graphic design on it. The background is plain and light-colored."
    # cs = clip_score(model, preprocess, image_path, text)
    # print(cs)

