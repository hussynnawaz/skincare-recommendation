from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import pandas as pd

app = Flask(__name__)

# Load model and transform
OUT_CLASSES = 3
index_label = {0: "dry", 1: "normal", 2: "oily"}
device = "cuda" if torch.cuda.is_available() else "cpu"

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, OUT_CLASSES)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the product data from CSV
products_df = pd.read_csv("products.csv")

# Helper function to predict skin type
def predict_skin_type(img):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return index_label[model(img).argmax(1).item()]

# Helper function to provide product recommendations
def get_product_recommendation(skin_type, category=None, problems=[]):
    # Filter products based on skin type
    filtered_products = products_df[products_df['skin_type'] == skin_type]

    # Filter or modify recommendations based on category and problems
    if category:
        filtered_products = filtered_products[filtered_products['category'].str.contains(category, case=False, na=False)]

    if problems:
        filtered_products = filtered_products[filtered_products['problems'].apply(lambda x: any(problem.lower() in x.lower() for problem in problems))]

    # If no products match the filters, return a fallback message
    if filtered_products.empty:
        return ["No suitable recommendations available."]

    # Return the product names (you can adjust this based on what info you want to show)
    return filtered_products['product_name'].tolist()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        img = None
        category = None
        problems = []

        # Check if image file is present
        if 'image' in request.files:
            img = Image.open(request.files['image']).convert("RGB")
        else:
            return jsonify({"error": "No image file provided."}), 400  # Image missing

        # Optional: category and problems fields (category can be a string, problems a comma-separated list)
        category = request.form.get('category', None)
        problems_str = request.form.get('problems', '')
        problems = problems_str.split(',') if problems_str else []

        if img is None:
            return jsonify({"error": "No image provided."}), 400  # Invalid image

        # Predict the skin type using the model
        skin_type = predict_skin_type(img)
        
        # Get the appropriate product recommendations based on skin type, category, and problems
        recommendations = get_product_recommendation(skin_type, category, problems)

        return jsonify({
            "skin_type": skin_type,
            "category": category,
            "problems": problems,
            "product_recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
