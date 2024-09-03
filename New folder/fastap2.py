# fastapi_app.py

import base64
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import io
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from deepface import DeepFace

# Initialize FastAPI app
app = FastAPI()

# Load the product database
file_path = 'export_skincare.csv'  # Update this path to the correct location
product_df = pd.read_csv(file_path)

# Load the best model
OUT_CLASSES = 3
label_index = {"dry": 0, "normal": 1, "oily": 2}
index_label = {0: "dry", 1: "normal", 2: "oily"}
best_model_path = "best_model.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, OUT_CLASSES)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model = model.to(device)
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_skin_type(img):
    img = transform(img)
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img)
        return out.argmax(1).item()

def predict_age(img_path):
    analysis = DeepFace.analyze(img_path, actions=['age'], enforce_detection=False)
    if isinstance(analysis, list):
        return analysis[0]['age']
    elif isinstance(analysis, dict):
        return analysis['age']
    else:
        raise ValueError("Unexpected analysis result format")

def recommend_products(skin_type, age, category, problems):
    recommendations = product_df[
        (product_df['product_type'].str.contains(category, case=False)) &
        (product_df['skintype'].str.contains(skin_type, case=False)) &
        (product_df['notable_effects'].apply(lambda x: any(problem in x for problem in problems)))
    ]
    return recommendations

class PredictionRequest(BaseModel):
    image_base64: str
    category: str
    problems: str

@app.post("/predict/")
async def predict(request: PredictionRequest):
    problems_list = request.problems.split(',')
    
    # Decode the base64 image
    img_data = base64.b64decode(request.image_base64.split(',')[1])
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    
    # Predict skin type
    skin_type_prediction = predict_skin_type(img)
    skin_type = index_label[skin_type_prediction]
    
    # Save the image temporarily for age prediction
    img_path = "temp.jpg"
    img.save(img_path)
    
    # Predict age
    age_prediction = predict_age(img_path)
    
    # Get product recommendations
    recommendations = recommend_products(skin_type, age_prediction, request.category, problems_list)
    
    # Format recommendations
    recommendations_list = recommendations.to_dict(orient='records')
    
    return JSONResponse(content={
        "skin_type": skin_type,
        "age": age_prediction,
        "recommendations": recommendations_list
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
