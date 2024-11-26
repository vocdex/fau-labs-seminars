import os
import torch
import numpy as np
from torchvision import models, transforms
from flask import Flask, render_template, Response
from PIL import Image
import cv2

app = Flask(__name__)

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

    # ReLU on cam
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()  # Normalize between 0 and 1

    handle_forward.remove()
    handle_backward.remove()

    return cam

def process_frame(frame):
    """
    Process a frame with the model and generate Grad-CAM overlay.
    """
    image = Image.fromarray(frame[..., ::-1])  # Convert BGR to RGB
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        _, indices = torch.topk(outputs, 1)  # Get the top prediction
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_index = indices[0][0].item()
        top_label = labels[top_index]
        top_prob = probabilities[0, top_index].item()

    # Generate Grad-CAM
    target_layer = model.features[-1]  # Last convolutional layer
    cam = generate_gradcam(input_tensor, model, target_layer, top_index)
    cam = cv2.resize(cam, (frame.shape[1], frame.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Overlay heatmap on the original frame
    overlay = heatmap * 0.4 + frame

    # Add label and probability to the overlay
    label_text = f"{top_label}: {top_prob:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(overlay, label_text, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay

@app.route('/')
def index():
    return render_template('camera_index.html')

def video_stream():
    """
    Capture frames from the webcam and process them in real-time.
    """
    cap = cv2.VideoCapture(0)  # Open the webcam
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to a smaller size (e.g., 640x480)
        frame = cv2.resize(frame, (640, 480))

        # Process the frame with Grad-CAM
        overlay = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', overlay)
        frame_data = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    """
    Route for streaming the video feed with Grad-CAM overlay.
    """
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=500, debug=False)
