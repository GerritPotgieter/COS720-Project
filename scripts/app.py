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
    
    inputs = tokenizer(
        email_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    probs = predictions[0].tolist()

    # Safer label extraction using model config
    labels = {
    "legitimate_email": probs[0],
    "phishing_url": probs[1],
    "legitimate_url": probs[2],
    "phishing_url_alt": probs[3]
}

    max_label = max(labels.items(), key=lambda x: x[1])
    
    return jsonify({
        'prediction': max_label[0],
        'confidence': max_label[1],
        'reasoning': labels
    })


    return {
        "prediction": max_label[0],
        "confidence": max_label[1],
        "all_probabilities": labels
    }


    
    email_text = request.form['email_text']
    
    inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1).max().item()

    result = "Phishing" if prediction == 1 else "Not Phishing"
    
    # Here we add a reasoning, for now it's just a placeholder
    reasoning = "Model determined based on typical phishing language patterns."
    
    

if __name__ == '__main__':
    app.run(debug=True)
