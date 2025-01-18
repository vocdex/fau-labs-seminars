import os
import requests
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def create_config():
    config_content = """
        inference_address=http://0.0.0.0:8080
        management_address=http://0.0.0.0:8081
        metrics_address=http://0.0.0.0:8082
        number_of_netty_threads=4
        job_queue_size=10
        model_store=/tmp/model_store
    """
    with open('config.properties', 'w') as f:
        f.write(config_content.strip())
    
    # Create model store directory if it doesn't exist
    os.makedirs('/tmp/model_store', exist_ok=True)

def start_torchserve(mar_file):
    # Copy .mar file to model store
    os.system(f'cp {mar_file} /tmp/model_store/')
    
    # Start TorchServe
    os.system('torchserve --start --model-store /tmp/model_store --models maskrcnn=maskrcnn.mar --ncs')


def predict_and_visualize(image_path):
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    url = 'http://localhost:8080/predictions/maskrcnn'
    response = requests.post(url, data=image_data, headers={'Content-Type': 'application/octet-stream'})
    
    results = json.loads(response.content)
    
    img = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    for box in results['boxes']:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
    
    if 'labels' in results:
        for i, (box, label) in enumerate(zip(results['boxes'], results['labels'])):
            x1, y1, _, _ = box
            plt.text(x1, y1, label, color='white', backgroundcolor='red')
    
    plt.axis('off')
    plt.savefig('output_detection.png')
    plt.close()

def main(mar_file_path, test_image_path):
    try:
        create_config()
        start_torchserve(mar_file_path)
        
        print("Waiting for TorchServe to start...")
        import time
        time.sleep(10)
        
        print("Running inference...")
        predict_and_visualize(test_image_path)
        
        print("Detection complete! Check output_detection.png for results")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        os.system('torchserve --stop')

if __name__ == "__main__":
    mar_file = "maskrcnn.mar" 
    test_image = "test_image.jpg" 
    main(mar_file, test_image)