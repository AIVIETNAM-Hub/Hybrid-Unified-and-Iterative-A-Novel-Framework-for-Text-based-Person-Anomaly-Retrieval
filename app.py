import streamlit as st
import numpy as np
import faiss
import torch
from PIL import Image
import os
import pickle
import time
import shutil
import beit3_infer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_data
def load_model():
    """Cached model loading function"""
    model, processor = beit3_infer.load_model_and_processor('checkpoint/lhp/lhp_beit3.pth', 'checkpoint/lhp/beit3.spm')
    model.eval().to(device)
    return model, processor

def load_index():
    try:
        index = faiss.read_index('index_storage/image_index.faiss')
    except:
        index = None
    try:
        with open('index_storage/image_paths.pkl', 'rb') as f:
            image_paths = pickle.load(f)
    except:
        image_paths = None
    return index, image_paths

def clear_storage():
    """Remove all files in index storage"""
    # Create and save Faiss index
    os.makedirs('index_storage', exist_ok=True)
    
    index = faiss.IndexFlatIP(1024)
    image_paths = []
    faiss.write_index(index, 'index_storage/image_index.faiss')
    with open('index_storage/image_paths.pkl', 'wb') as f:
        pickle.dump(image_paths, f)
    st.success("Storage cleared successfully")
    return index, image_paths

def extract_folder(index, image_paths, model, processor, image_folder):
    """Extract embeddings for all images in a folder"""
    total_images = [
        f for f in os.listdir(image_folder) 
        if os.path.isfile(os.path.join(image_folder, f)) and 
        f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ]
    
    embeddings = []
    
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    for processed_count, filename in enumerate(total_images, 1):
        filepath = os.path.join(image_folder, filename)
        
        # Update progress
        progress = processed_count / len(total_images)
        progress_bar.progress(progress)
        status_text.text(f"Processing {filename} ({processed_count}/{len(total_images)})")
        
        # Extract embedding
        embedding = beit3_infer.encode_image(filepath, processor, model)
        if embedding is not None:
            embeddings.append(embedding)
            image_paths.append(filepath)
        
        time.sleep(0.1)
    
    index.add(np.array(embeddings))
    faiss.write_index(index, 'index_storage/image_index.faiss')
    
    with open('index_storage/image_paths.pkl', 'wb') as f:
        pickle.dump(image_paths, f)
    
    progress_bar.progress(1.0)
    status_text.text(f"Completed! Processed {len(total_images)} images")
    st.sidebar.success("Embedding extraction finished successfully")

def extract_image(index, image_paths, model, processor, upload_image):
    image = Image.open(upload_image).convert("RGB")
        
    # Extract embedding
    embedding = beit3_infer.encode_image(image, processor, model)
    imagepath = f"./images/{len(image_paths)}_{upload_image.name}"
    image.save(imagepath)
    if embedding is not None:
        image_paths.append(imagepath)
    
    index.add(np.array(embedding).reshape(1, -1))
    faiss.write_index(index, 'index_storage/image_index.faiss')
    
    with open('index_storage/image_paths.pkl', 'wb') as f:
        pickle.dump(image_paths, f)

    st.sidebar.success("Embedding extraction finished successfully")

def retrieve_similar_items(index, image_paths, model, processor, query, top_k=5, query_type='image'):
    """Retrieve similar images based on image or text query"""
    
    # Prepare embedding
    if query_type == 'image':
        query_embedding = beit3_infer.encode_image(query, processor, model)
    else:
        query_embedding = beit3_infer.encode_text(query, processor, model)
    
    # Search similar images
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    
    return [image_paths[i] for i in indices[0]]

def main():
    st.title('🖼️ Image Retrieval App')
    
    # Load model
    model, processor = load_model()
    index, image_paths = load_index()
    
    # Sidebar for operations
    st.sidebar.header('Operations')
    st.sidebar.subheader('Extract Folder')
    image_folder = st.sidebar.text_input('Image Folder Path', './images')
    
    if st.sidebar.button('Extract Folder'):
        if os.path.exists(image_folder):
            extract_folder(index, image_paths, model, processor, image_folder)
        else:
            st.error("Specified folder does not exist")
    st.sidebar.markdown("---")
    st.sidebar.subheader('Extract Image')
    upload_image = st.sidebar.file_uploader("Choose a query image", type=['png', 'jpg', 'jpeg', 'bmp', 'gif'], key='extract')
    if st.sidebar.button('Extract Image'):
        try:
            extract_image(index, image_paths, model, processor, upload_image)
        except:
            st.error("Specified image does not exist")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Clear The Storage of Image")
    if st.sidebar.button('Clear Storage'):
        index, image_paths = clear_storage()
    
    # Main area for search
    st.header('Similarity Search')
    st.info(f"There are {len(image_paths)} images in the storage")
    # Check if index exists
    if index:
        # Search type selection
        search_type = st.radio('Search Type', ['Image', 'Text'])
        top_k = st.slider('Top K Images', min_value=5, max_value=100, value=10, step=5)
        
        if search_type == 'Image':
            uploaded_file = st.file_uploader("Choose a query image", type=['png', 'jpg', 'jpeg', 'bmp', 'gif'], key='retrieve')
            if uploaded_file is not None:
                query_image = Image.open(uploaded_file).convert("RGB")
                st.image(query_image, caption='Query Image', use_container_width=True)
                
                similar_images = retrieve_similar_items(index, image_paths, model, processor, query_image, top_k, 'image')
        else:
            text_query = st.text_input("Enter text description")
            if st.button('Search'):
                similar_images = retrieve_similar_items(index, image_paths, model, processor, text_query, top_k, 'text')
        
        # Display similar images
        if 'similar_images' in locals():
            st.subheader(f'Top {top_k} Similar Images')
            # Group images into rows of 5
            for i in range(0, len(similar_images), 5):
                row_images = similar_images[i:i+5]  # Group 5 images
                cols = st.columns(len(row_images))  # Create columns for the row
                for col, img_path in zip(cols, row_images):
                    with col:
                        st.image(img_path, use_container_width=True)
                        st.write(os.path.basename(img_path))
    else:
        st.warning("Please extract embeddings first using the sidebar")

if __name__ == "__main__":
    main()
    