import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import pandas as pd
import base64
from io import BytesIO
import re
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the product database
file_path = 'export_skincare.csv'
product_df = pd.read_csv(file_path)

# Load the model
OUT_CLASSES = 3
index_label = {0: "dry", 1: "normal", 2: "oily"}
best_model_path = "best_model.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, OUT_CLASSES)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.to(device)
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_skin_type(img):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return index_label[model(img).argmax(1).item()]

def recommend_products(skin_type, category, problems):
    return product_df[
        (product_df['product_type'].str.contains(category, case=False)) &
        (product_df['skintype'].str.contains(skin_type, case=False)) &
        (product_df['notable_effects'].apply(lambda x: any(p in x for p in problems)))
    ]
@app.route('/predict', methods=['POST'])
def predict():
    try:
        img = None
        category = None
        problems = []

        # Check if the content-type is multipart/form-data (for file upload)
        if 'image' in request.files:  # File Upload (multipart/form-data)
            img = Image.open(request.files['image']).convert("RGB")
            category = request.form.get('category')
            problems_str = request.form.get('problems')
            problems = problems_str.split(',') if problems_str else []

        # Check if the content-type is application/json (for JSON upload with base64 image)
        elif request.content_type == 'application/json':  # JSON with base64 image
            data = request.get_json()

            if data and "image_base64" in data:
                image_base64 = data.get("image_base64")
                category = data.get("category")
                problems = data.get("problems", "").split(',')
                try:
                    image_data = base64.b64decode(image_base64.split(",")[1])
                    img = Image.open(BytesIO(image_data)).convert("RGB")
                except (ValueError, IndexError) as e:
                    return jsonify({"error": f"Invalid base64 image data: {e}"}), 400
            else:
                return jsonify({"error": "Invalid JSON data. 'image_base64' is required."}), 400

        else:
            return jsonify({"error": "Unsupported Media Type.  Use multipart/form-data or application/json."}), 415

        if img is None:
           return jsonify({"error": "No image provided."}), 400

        # Predict skin type and get product recommendations
        skin_type = predict_skin_type(img)
        recommendations = recommend_products(skin_type, category, problems)

        return jsonify({
            "skin_type": skin_type,
            "recommendations": recommendations.to_dict(orient='records')
        })

    except (KeyError, ValueError, TypeError, OSError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred: " + str(e)}), 500

# Load and process product data
file_path = 'export_skincare 1.csv'
df = pd.read_csv(file_path)
df['PKR_numeric'] = df['PKR'].apply(lambda x: float(re.sub(r'[^\d.]', '', str(x))) if isinstance(x, str) else 0)

@app.route('/products', methods=['GET'])
def get_products():
    product_type = request.args.get("product_type")
    filtered_df = df[df['product_type'].str.contains(product_type, case=False)] if product_type else df
    return jsonify({
        "products": filtered_df[['product_name', 'product_type', 'skintype', 'PKR', 'description', 'notable_effects', 'brand']].to_dict(orient='records')
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
