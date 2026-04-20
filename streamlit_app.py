# ============================
# 🔹 IMPORTS
# ============================
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

# ============================
# 🔹 MODEL CLASS
# ============================
class DeepfakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ============================
# 🔹 LOAD MODEL
# ============================
@st.cache_resource
def load_model():
    model = DeepfakeModel()
    model.load_state_dict(torch.load("/Users/rajupatel/deepfake_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ============================
# 🔹 IMAGE TRANSFORM
# ============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================
# 🔹 UI
# ============================
st.title("🧠 AI-Based Deepfake Detection System")

st.write("Upload an image to check whether it is Real or Fake.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        prediction = (output > 0.5).item()

    if prediction == 1:
        st.error("❌ Fake Image Detected")
    else:
        st.success("✅ Real Image")