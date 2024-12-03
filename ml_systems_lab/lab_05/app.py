import torch
import csv
import torch.nn as nn
from flask import request, jsonify, Flask
import numpy as np
from model import ConvNet
import warnings
from finetune import seed_everything
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")

seed_everything()

app = Flask(__name__)

# Load model and training data
model = ConvNet()
device = "cpu"
model.to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

# Load training data
X_train = torch.load('X_train.pt')
y_train = torch.load('y_train.pt')

# Storage for predictions
stored_predictions = []
prediction_limit = 1000
# Initialize global variables at the start
top1_correct = 0
top5_correct = 0
total_predictions = 0


@app.route('/', methods=['POST'])
def predict():
    global stored_predictions, top1_correct, top5_correct, total_predictions

    if 'index' not in request.form:
        return jsonify({"error": "No index provided"}), 400

    index = int(request.form['index'])
    image = X_train[index]
    true_label = y_train[index].item()

    if image.dim() == 2:
        image = image.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image.unsqueeze(0))
        _, predicted = torch.max(outputs, 1)
        top5_predictions = torch.topk(outputs, 5).indices.squeeze()

    # Update accuracy metrics
    total_predictions += 1
    top1_correct += int(predicted == true_label)
    top5_correct += int(true_label in top5_predictions)

    # Save to CSV when prediction limit is reached
    if total_predictions >= prediction_limit:
        with open('train_model_accuracy.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Metric', 'Accuracy'])
            writer.writeheader()
            writer.writerow({
                'Metric': 'Top-1 Accuracy', 
                'Accuracy': f"{100 * top1_correct / total_predictions:.2f}%"
            })
            writer.writerow({
                'Metric': 'Top-5 Accuracy', 
                'Accuracy': f"{100 * top5_correct / total_predictions:.2f}%"
            })
        print("Accuracy metrics saved")

    return jsonify({
        'prediction': predicted.item(),
        'top_5_predictions': top5_predictions.tolist(),
        'true_label': true_label
    })


if __name__ == '__main__':
    app.run(debug=False, port=5000)