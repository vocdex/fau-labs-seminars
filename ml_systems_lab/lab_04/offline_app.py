import os
import torch
import numpy as np
from torchvision import models, transforms
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
from flask import send_from_directory
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load VGG16 model and weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg16(pretrained=False)
model.load_state_dict(torch.load('weights/vgg16_weights.pth', map_location=device))
model = model.to(device)
model.eval()

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

with open('imagenet_classes.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def generate_gradcam(input_tensor, model, target_layer, class_index):
    """
    Generates a Grad-CAM heatmap.
    """
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    # Attach hooks to the target layer
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    # Forward and backward passes
    output = model(input_tensor)
    model.zero_grad()
    class_score = output[0, class_index]
    class_score.backward()

    # Get the gradients and activations
    grads = gradients["value"].cpu().data.numpy()[0]
    acts = activations["value"].cpu().data.numpy()[0]

    # Global average pooling of gradients
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    # Relu on cam
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()  # Normalize between 0 and 1

    handle_forward.remove()
    handle_backward.remove()

    return cam

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))
    
    # Save and process the uploaded image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)

    image = Image.open(image_path).convert('RGB')
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

    # Grad-CAM for the top prediction
    target_layer = model.features[-1]  # Last convolutional layer
    cam = generate_gradcam(input_tensor, model, target_layer, top_indices[0])

    # Overlay Grad-CAM on the original image
    cam = cv2.resize(cam, (image.width, image.height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    original_image = np.array(image)
    overlay = heatmap * 0.4 + original_image[..., ::-1] 
    overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], f"gradcam_{file.filename}")
    cv2.imwrite(overlay_path, overlay)

    return render_template('result.html', image_path=image_path, overlay_path=overlay_path, predictions=top_classes)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=500, debug=False)
