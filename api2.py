from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

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

# Helper function to predict skin type
def predict_skin_type(img):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return index_label[model(img).argmax(1).item()]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        img = None
        category = None
        problems = []

        # Ensure that the 'image' field is included in the form data
        if 'image' in request.files:  # Check for image file upload
            img = Image.open(request.files['image']).convert("RGB")
            category = request.form.get('category')
            problems_str = request.form.get('problems')
            problems = problems_str.split(',') if problems_str else []

        else:
            return jsonify({"error": "No image file provided."}), 400  # Image field missing

        if img is None:
            return jsonify({"error": "No image provided."}), 400

        skin_type = predict_skin_type(img)
        return jsonify({
            "skin_type": skin_type,
            "category": category,
            "problems": problems
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
