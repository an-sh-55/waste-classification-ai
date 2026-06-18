import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import urllib.request
import os

st.title("♻️ Waste Classifier AI")
st.write("Upload a photo of your trash, and our AI will tell you what it is!")

# --- NEW: Download the brain from Google Drive if it isn't here yet ---
MODEL_PATH = 'waste_model.pth'
if not os.path.exists(MODEL_PATH):
    st.write("First time setup: Waking up the AI's brain... (takes a few seconds)")
    
    drive_link = "https://drive.google.com/file/d/1s4qVDXWxWBkxDK9LBGkJLrIavuKstx-d/view?usp=share_link"
    
    try:
        file_id = drive_link.split('/d/')[1].split('/')[0]
        direct_download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
        urllib.request.urlretrieve(direct_download_url, MODEL_PATH)
    except Exception as e:
        st.error("Could not download the brain file. Double check your Google Drive link configuration!")

# Rebuild the brain structure
model = models.resnet18()
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 6) 

# Load the file we just downloaded
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
# ---------------------------------------------------------------------

# --- The rest of the app file ---
uploaded_file = st.file_uploader("Choose a picture...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Your Photo', use_container_width=True)
    st.write("AI is thinking...")
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = test_transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_idx = torch.max(outputs, 1)
    
    final_answer = classes[predicted_idx.item()]
    st.success(f"Looks like this is: {final_answer.upper()}!")