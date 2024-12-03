import torch
import torch.nn as nn
import torch.optim as optim
import requests
import time
import warnings
import os
import csv
import random
import numpy as np
from model import ConvNet
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

seed_everything()

def perform_requests():
    # URL of the Flask server
    url = 'http://localhost:5000/'
    
    # Total number of requests
    total_requests = 1000
    
    # Perform requests
    for i in range(total_requests):
        try:
            # Send POST request with index
            os.system(f"curl -X POST {url} -F 'index={i}'")            
            # Small delay to prevent overwhelming the server
            time.sleep(0.01)
        
        except Exception as e:
            print(f"Error on request {i+1}: {e}")
            continue

def fine_tune_model():
    # Load model and training data
    model = ConvNet()
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    
    X_train = torch.load('X_train.pt')
    y_train = torch.load('y_train.pt')
    
    # Prepare for training
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Fine-tuning loop
    for epoch in range(5):  # 5 epochs
        for i in range(len(X_train)):
            inputs = X_train[i].unsqueeze(0)
            label = y_train[i]
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, label.unsqueeze(0))
            
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1} completed")
    
    # Save fine-tuned model
    torch.save(model.state_dict(), 'model_finetuned.pth')
    print("Fine-tuning completed and model saved")

def compare_models():
    X_test = torch.load('x_test.pt')
    Y_test = torch.load('y_test.pt')
    
    models = {
        'Original': torch.load('model.pth', map_location=torch.device('cpu')),
        'Fine-tuned': torch.load('model_finetuned.pth', map_location=torch.device('cpu'))
    }
    
    results = {}
    
    for model_name, model_state_dict in models.items():
        model = ConvNet()
        model.load_state_dict(model_state_dict)
        model.eval()
        
        top1_correct = 0
        top5_correct = 0
        total = len(X_test)
        
        with torch.no_grad():
            for i in range(total):
                outputs = model(X_test[i].unsqueeze(0))
                
                # Top-1 accuracy
                _, top1_pred = torch.max(outputs, 1)
                top1_correct += (top1_pred == Y_test[i]).item()
                
                # Top-5 accuracy
                top5_preds = torch.topk(outputs.squeeze(), 5).indices
                top5_correct += int(Y_test[i] in top5_preds)
        
        results[model_name] = {
            'Top-1 Accuracy': 100 * top1_correct / total,
            'Top-5 Accuracy': 100 * top5_correct / total
        }
    
    # Save to CSV
    import csv
    with open('compare_models.csv', 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Top-1 Accuracy', 'Top-5 Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for model_name, metrics in results.items():
            writer.writerow({
                'Model': model_name,
                'Top-1 Accuracy': f"{metrics['Top-1 Accuracy']:.2f}%",
                'Top-5 Accuracy': f"{metrics['Top-5 Accuracy']:.2f}%"
            })
    
    # Print results
    for model_name, metrics in results.items():
        print(f"{model_name} Model:")
        print(f"  Top-1 Accuracy: {metrics['Top-1 Accuracy']:.2f}%")
        print(f"  Top-5 Accuracy: {metrics['Top-5 Accuracy']:.2f}%")


def main():
    # Perform 1000 requests
    perform_requests()
    
    # Trigger fine-tuning
    fine_tune_model()

    # Compare models
    compare_models()

if __name__ == '__main__':
    main()