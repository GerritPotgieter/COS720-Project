import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Constants
DATA_PATH = "data/phishing_email.csv"
BATCH_SIZE = 16
MODEL_PATH = "GerritPot/COS720-Project-Phishing"

# Load model and tokenizer from Huggingface
config = AutoConfig.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, config=config)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
model.eval()

# Load and preprocess data
df = pd.read_csv(DATA_PATH)
df = df.dropna()
df["text"] = df["text_combined"].astype(str).str.replace(r"\s+", " ", regex=True)
df["label"] = df["label"].astype(int)

# Create train/validation split - ONLY USE VALIDATION SET FOR EVAL
_, val_texts, _, val_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# Tokenize validation texts
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# Dataset wrapper
class EmailDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        } | {"labels": torch.tensor(self.labels[idx])}

val_dataset = EmailDataset(val_encodings, val_labels)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Run evaluation on validation set
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

# Print classification report
print("\n=== Classification Report on Validation Set ===")
print(classification_report(all_labels, all_preds, target_names=["Not Phishing", "Phishing"]))
