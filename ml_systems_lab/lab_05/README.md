# Usage
Run the Flask server:
```bash
python app.py
```
Then, open another terminal and run the following command:
```bash
python finetune.py
```
This will send 1000 POST requests to the server and writes the Top-1 and Top-5 accuracies to the `train_model_accuracy.csv` file. After that, the model will be fine-tuned on test data and the Top-1 and Top-5 accuracies will be written to the `compare_models.csv` file.
