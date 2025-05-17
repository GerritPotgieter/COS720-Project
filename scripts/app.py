import re
from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import os

# Init flask with correct frontend folder to fetch the bootstrap page
app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'frontend'))

#load the actual model in stored in the model folder
model_path = "model_output" #Make sure path is correct!! and contains config.json, pytorch_model.bin and tokenizer files
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)

#uses this for the predict function, doesnt really matter but still probably quicker
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#sends the normal bootstrap page
@app.route('/')
def home():
    return render_template('index.html')

# This gets called when the user submits the form

@app.route('/predict', methods=['POST'])
def predict():
    subject = request.form.get('subject', '')
    body = request.form.get('body', '')
    sender = request.form.get('sender', '')
    url = request.form.get('url', '')

    # Combine input fields for model
    email_text = f"[SENDER] {sender} [SUBJECT] {subject} [BODY] {body} [URL] {url}".strip()

    if not email_text:
        return jsonify({'error': 'Email content is required.'})

    # Tokenize
    inputs = tokenizer(
        email_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()

    labels = {
        "legitimate": probs[0],
        "phishing": probs[1]
    }

    prediction_label = "phishing" if probs[1] > probs[0] else "legitimate"
    confidence = round(max(probs), 4)

    # --- Reasoning logic ---
    reasoning = []
    if sender and re.search(r"(noreply|alert|account|support|suspend)", sender, re.IGNORECASE):
        reasoning.append("Suspicious sender address")
    if url and re.search(r"(bit\.ly|\.ru|\.cn|tinyurl|phish)", url, re.IGNORECASE):
        reasoning.append("Suspicious or shortened URL")
    if re.search(r"(password|verify|login|urgent|click here|award|prize|Inheritance|beneficiary )", body, re.IGNORECASE):
        reasoning.append("Phishing-related language in body")
    if not reasoning:
        reasoning.append(" ")

    return jsonify({
        "prediction": prediction_label,
        "confidence": confidence,
        "scores": labels,
        "reasoning": reasoning
    })


    
if __name__ == '__main__':
    app.run(debug=True)
