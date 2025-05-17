import re
from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import os

# Init flask with correct frontend folder to fetch the bootstrap page
app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'frontend'))

#load the actual model in stored in the model folder
model_path = "GerritPot/COS720-Project-Phishing" #Make sure path is correct!! and contains config.json, pytorch_model.bin and tokenizer files
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_path)

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
    body = request.form.get('body', '')
   

    #check that body is not empty
    if not body:
        return jsonify({'error': 'Email body is required.'})

    # Combine input fields for model
    #email_text = f"[SENDER] {sender} [SUBJECT] {subject} [BODY] {body} [URL] {url}".strip()
    email_text = body

  

    # Tokenize
    inputs = tokenizer(
        email_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)

    probs = prediction[0].tolist()

    labels = {
        "legitimate_email": probs[0],
        "phishing_url": probs[1],
        "legitimate_url": probs[2],
        "phishing_url_alt": probs[3]
    }

    max_label = max(labels.items(), key=lambda x: x[1])
    prediction_label = max_label[0]
    confidence = max_label[1]

    #if prediction label is legit email or url then make the label only say legitimate
    if prediction_label == ("legitimate_email" or "legitimate_url"):
        prediction_label = "Legitimate email"
    elif prediction_label ==("phishing_url" or "phishing_url_alt"):
        prediction_label = "Phishing email"



    # --- Reasoning logic ---
    reasoning = []
    if re.search(r"(password|verify|login|urgent|click here|award|prize|Inheritance|beneficiary)", body, re.IGNORECASE):
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
