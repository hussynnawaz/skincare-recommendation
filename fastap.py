# # fastapi_app.py
# import torch
# from torchvision import transforms
# from torchvision.models import resnet50, ResNet50_Weights
# from PIL import Image
# import numpy as np
# import pandas as pd
# from typing import Optional
# from fastapi import FastAPI, File, UploadFile, Form, Query
# from fastapi.responses import JSONResponse
# import re

# # Initialize FastAPI app
# app = FastAPI()


# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

# # Load the product database
# file_path = 'export_skincare.csv'  # Update this path to the correct location
# product_df = pd.read_csv(file_path)

# # Load the best model
# OUT_CLASSES = 3
# label_index = {"dry": 0, "normal": 1, "oily": 2}
# index_label = {0: "dry", 1: "normal", 2: "oily"}
# best_model_path = "best_model.pth"
# device = "cuda" if torch.cuda.is_available() else "cpu"

# model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, OUT_CLASSES)
# model.load_state_dict(torch.load(best_model_path, map_location=device))
# model = model.to(device)
# model.eval()

# # Define the image transformation
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def predict_skin_type(img):
#     img = transform(img)
#     img = img.unsqueeze(0).to(device)
#     with torch.no_grad():
#         out = model(img)
#         return out.argmax(1).item()

# # def predict_age(img_path):
# #     analysis = DeepFace.analyze(img_path, actions=['age'], enforce_detection=False)
# #     if isinstance(analysis, list):
# #         return analysis[0]['age']
# #     elif isinstance(analysis, dict):
# #         return analysis['age']
# #     else:
# #         raise ValueError("Unexpected analysis result format")

# def recommend_products(skin_type, category, problems):
#     recommendations = product_df[
#         (product_df['product_type'].str.contains(category, case=False)) &
#         (product_df['skintype'].str.contains(skin_type, case=False)) &
#         (product_df['notable_effects'].apply(lambda x: any(problem in x for problem in problems)))
#     ]
#     return recommendations

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...), category: str = Form(...), problems: str = Form(...)):
#     problems_list = problems.split(',')
#     img = Image.open(file.file).convert("RGB")
    
#     # Predict skin type
#     skin_type_prediction = predict_skin_type(img)
#     skin_type = index_label[skin_type_prediction]
    
#     # Save the uploaded image temporarily for age prediction
#     img_path = "temp.jpg"
#     img.save(img_path)
    
#     # Predict age
#     # age_prediction = predict_age(img_path)
    
#     # Get product recommendations
#     recommendations = recommend_products(skin_type, category, problems_list)
    
#     # Format recommendations
#     recommendations_list = recommendations.to_dict(orient='records')
    
#     return JSONResponse(content={
#         "skin_type": skin_type,
#         # "age": age_prediction,
#         "recommendations": recommendations_list
#     })

# # Load the CSV data
# file_path = 'export_skincare 1.csv'
# df = pd.read_csv(file_path)

# # Convert the PKR column to a numeric type
# def convert_pkr_to_numeric(pkr_value):
#     if isinstance(pkr_value, str):
#         # Extract numeric part and remove commas
#         numeric_value = re.sub(r'[^\d.]', '', pkr_value)
#         return float(numeric_value)
#     return 0

# df['PKR_numeric'] = df['PKR'].apply(convert_pkr_to_numeric)

# @app.get("/products/")
# def get_products(
#     product_type: Optional[str] = Query(None, description="Filter by product type"),
#     # skintype: Optional[str] = Query(None, description="Filter by skin type"),
#     # max_price: Optional[float] = Query(None, description="Filter by maximum PKR"),
#     # min_price: Optional[float] = Query(None, description="Filter by minimum PKR"),
# ):
#     filtered_df = df.copy()

#     # Apply filters
#     if product_type:
#         filtered_df = filtered_df[filtered_df['product_type'].str.contains(product_type, case=False)]

#     # if skintype:
#     #     filtered_df = filtered_df[filtered_df['skintype'].str.contains(skintype, case=False)]

#     # if max_price is not None:
#     #     filtered_df = filtered_df[filtered_df['PKR_numeric'] <= max_price]
        
#     # if min_price is not None:
#     #     filtered_df = filtered_df[filtered_df['PKR_numeric'] >= min_price]

#     # Convert to dictionary
#     products = filtered_df[['product_name', 'product_type', 'skintype', 'PKR','description','notable_effects','brand']].to_dict(orient='records')
#     return {"products": products}


# fastapi_app.py




import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
# from deepface import DeepFace
import pandas as pd
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException
from fastapi.responses import JSONResponse
import re
import logging
from fastapi.logger import logger

logging.basicConfig(level=logging.DEBUG)
logger.setLevel(logging.DEBUG)


# Initialize FastAPI app
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[""],  # Replace "" with your frontend's domain for better security
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

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

# def predict_age(img_path):
#     analysis = DeepFace.analyze(img_path, actions=['age'], enforce_detection=False)
#     if isinstance(analysis, list):
#         return analysis[0]['age']
#     elif isinstance(analysis, dict):
#         return analysis['age']
#     else:
#         raise ValueError("Unexpected analysis result format")

def recommend_products(skin_type, category, problems):
    recommendations = product_df[
        (product_df['product_type'].str.contains(category, case=False)) &
        (product_df['skintype'].str.contains(skin_type, case=False)) &
        (product_df['notable_effects'].apply(lambda x: any(problem in x for problem in problems)))
    ]
    return recommendations

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...), category: str = Form(...), problems: str = Form(...)):
#     problems_list = problems.split(',')
#     img = Image.open(file.file).convert("RGB")
    
#     # Predict skin type
#     skin_type_prediction = predict_skin_type(img)
#     skin_type = index_label[skin_type_prediction]
    
#     # Save the uploaded image temporarily for age prediction
#     img_path = "temp.jpg"
#     img.save(img_path)
    
#     # Predict age
#     # age_prediction = predict_age(img_path)
    
#     # Get product recommendations
#     recommendations = recommend_products(skin_type, category, problems_list)
    
#     # Format recommendations
#     recommendations_list = recommendations.to_dict(orient='records')
    
#     return JSONResponse(content={
#         "skin_type": skin_type,
#         # "age": age_prediction,
#         "recommendations": recommendations_list
#     })

from fastapi import HTTPException
import base64
from io import BytesIO

@app.post("/predict/")
async def predict(image_base64: str = Form(...), category: str = Form(...), problems: str = Form(...)):
    problems_list = problems.split(',')
    print(image_base64)
    print(category)

    try:
        # Decode the base64 string into an image
        image_data = base64.b64decode(image_base64.split(",")[1])
        img = Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # Predict skin type
    skin_type_prediction = predict_skin_type(img)
    skin_type = index_label[skin_type_prediction]

    # Get product recommendations
    recommendations = recommend_products(skin_type, category, problems_list)

    # Format recommendations
    recommendations_list = recommendations.to_dict(orient='records')

    return JSONResponse(content={
        "skin_type": skin_type,
        "recommendations": recommendations_list
    })




# Load the CSV data
file_path = 'export_skincare 1.csv'
df = pd.read_csv(file_path)

# Convert the PKR column to a numeric type
def convert_pkr_to_numeric(pkr_value):
    if isinstance(pkr_value, str):
        # Extract numeric part and remove commas
        numeric_value = re.sub(r'[^\d.]', '', pkr_value)
        return float(numeric_value)
    return 0

df['PKR_numeric'] = df['PKR'].apply(convert_pkr_to_numeric)

@app.get("/products/")
def get_products(
    product_type: Optional[str] = Query(None, description="Filter by product type"),
    # skintype: Optional[str] = Query(None, description="Filter by skin type"),
    # max_price: Optional[float] = Query(None, description="Filter by maximum PKR"),
    # min_price: Optional[float] = Query(None, description="Filter by minimum PKR"),
):
    filtered_df = df.copy()

    # Apply filters
    if product_type:
        filtered_df = filtered_df[filtered_df['product_type'].str.contains(product_type, case=False)]

    # if skintype:
    #     filtered_df = filtered_df[filtered_df['skintype'].str.contains(skintype, case=False)]

    # if max_price is not None:
    #     filtered_df = filtered_df[filtered_df['PKR_numeric'] <= max_price]
        
    # if min_price is not None:
    #     filtered_df = filtered_df[filtered_df['PKR_numeric'] >= min_price]

    # Convert to dictionary
    products = filtered_df[['product_name', 'product_type', 'skintype', 'PKR','description','notable_effects','brand']].to_dict(orient='records')
    return {"products": products}