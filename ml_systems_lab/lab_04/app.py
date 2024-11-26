"""Done by Shukrullo Nazirjonov"""
import torch
from torchvision import models, transforms
from flask import Flask, request, jsonify
from PIL import Image
import os

app = Flask(__name__)

# Load VGG16 model and weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg16(weights=False)
model.load_state_dict(torch.load('/Users/shuk/Desktop/Random/coding-exercises/vgg16_weights.pth', map_location=device))
model = model.to(device)
model.eval()

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Label mapping
with open('class_map.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]
    labels = [label.split(' ')[-1] for label in labels]

@app.route('/', methods=['GET','POST'])
def predict():
    # Check if an image is uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']

    try:
        image = Image.open(file.stream).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            _, indices = torch.topk(outputs, 5)  # Get top 5 predictions
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs = probabilities[0, indices[0]].cpu().numpy()
            top_indices = indices[0].cpu().numpy()

        # Map indices to labels
        top_classes = [{"class": labels[i], "probability": float(top_probs[j])} 
                       for j, i in enumerate(top_indices)]

        return jsonify(top_classes)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=500)
