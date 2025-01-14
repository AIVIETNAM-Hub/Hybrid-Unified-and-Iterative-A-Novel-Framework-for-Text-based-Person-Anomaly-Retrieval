import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
import sys
# print(sys.path)

def init_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=device)
    return model, vis_processors, txt_processors

# if __name__ == '__main__':
#      model, vis_processors, txt_processors = init_model()
#     cs = score('/home/s48gb/Desktop/GenAI4E/pab/WWW2025Workshop_mimcode/data/PAB/train/imgs_0/goal/0.jpg', "The image shows a band performing on stage under a large tent. The stage is decorated with green garlands. There are four musicians visible: one playing the saxophone, one playing the guitar, one playing the drums, and one playing the bass. The saxophonist and guitarist are wearing black shirts and pants, while the drummer and bassist are also dressed in black. The background features a grassy area with trees and a clear sky.", vis_processors, txt_processors, model)
#     print(cs)

