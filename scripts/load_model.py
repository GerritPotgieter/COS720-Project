from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "cybersectony/phishing-email-detection-distilbert_v2.4.1"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Quick check
print("Model and tokenizer loaded successfully!")
