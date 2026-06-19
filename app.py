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

# --- FORCE A FRESH DOWNLOAD BY CHANGING THE NAME TO V1 ---
MODEL_PATH = 'waste_model_v1.pth'
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading the fresh AI brain... (takes a few seconds)"):
        release_url = "https://github.com/an-sh-55/waste-classification-ai/releases/download/v1.0/waste_model.pth"
        try:
            urllib.request.urlretrieve(release_url, MODEL_PATH)
        except Exception as e:
            st.error(f"Could not download the brain file: {e}")

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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = test_transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_idx = torch.max(outputs, 1)
    
    final_answer = classes[predicted_idx.item()]
    st.success(f"Looks like this is: {final_answer.upper()}!")
