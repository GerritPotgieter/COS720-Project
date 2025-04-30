from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import os

# Initialize Flask app with custom template folder
app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'frontend'))

model_path = "model/model2"
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email_text']
    
    inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1).max().item()

    result = "Phishing" if prediction == 1 else "Not Phishing"
    
    # Here we add a reasoning, for now it's just a placeholder
    reasoning = "Model determined based on typical phishing language patterns."
    
    return jsonify({'prediction': result, 'confidence': confidence, 'reasoning': reasoning})

if __name__ == '__main__':
    app.run(debug=True)
