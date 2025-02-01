from flask import Flask, request, jsonify
from PIL import Image
import base64
import io
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import pandas as pd
import re

app = Flask(__name__)

# Load the product database
file_path = 'products.csv'  # Update this path to the correct location
df = pd.read_csv(file_path)

# Convert the PKR column to a numeric type
def convert_pkr_to_numeric(pkr_value):
    if isinstance(pkr_value, str):
        # Extract numeric part and remove commas
        numeric_value = re.sub(r'[^\d.]', '', pkr_value)
        return float(numeric_value)
    return 0

df['PKR_numeric'] = df['PKR'].apply(convert_pkr_to_numeric)

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

def recommend_products(skin_type, category, problems):
    recommendations = df[
        (df['product_type'].str.contains(category, case=False)) &
        (df['skintype'].str.contains(skin_type, case=False)) &
        (df['notable_effects'].apply(lambda x: any(problem in x for problem in problems)))
    ]
    return recommendations
@app.route('/predict/', methods=['POST'])
def predict():
    data = request.form
    image_base64 = data.get('image')  # Use request.form for multipart/form-data
    category = data.get('category')
    problems = data.get('problems', '').split(',')

    if not image_base64:
        return jsonify({"error": "No image data provided"}), 400

    try:
        # Decode the base64 string into an image
        image_data = base64.b64decode(image_base64.split(",")[1])  # Remove the 'data:image/jpeg;base64,' part
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Invalid image data"}), 400

    # Predict skin type
    skin_type_prediction = predict_skin_type(img)
    skin_type = index_label[skin_type_prediction]

    # Get product recommendations
    recommendations = recommend_products(skin_type, category, problems)

    # Format recommendations
    recommendations_list = recommendations.to_dict(orient='records')

    return jsonify({
        "skin_type": skin_type,
        "recommendations": recommendations_list
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    
