import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score
import os

MODEL_NAME = "cybersectony/phishing-email-detection-distilbert_v2.4.1"
DATA_PATH = "data/processed_phishing_emails.csv"
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
OUTPUT_DIR = "model/newmodel"
LOG_FILE = os.path.join(OUTPUT_DIR, "training_log.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text_combined", "label"])
df["label"] = df["label"].astype(int)

train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df["text_combined"].tolist(), df["label"].tolist(), test_size=0.2, stratify=df["label"], random_state=42
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

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

train_dataset = EmailDataset(train_encodings, train_labels)
val_dataset = EmailDataset(val_encodings, val_labels)
test_dataset = EmailDataset(test_encodings, test_labels)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

num_training_steps = EPOCHS * len(train_loader)
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

with open(LOG_FILE, "w") as log:
    model.train()
    for epoch in range(EPOCHS):
        loop = tqdm(train_loader, leave=True)
        running_loss = 0

        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()

            loop.set_description(f"Epoch {epoch + 1}")
            loop.set_postfix(loss=loss.item())

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch["labels"].cpu().numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds)
        val_report = classification_report(val_targets, val_preds, digits=4)

        print(f"\nEpoch {epoch + 1} Validation Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}")
        log.write(f"\nEpoch {epoch + 1}\nValidation Accuracy: {val_acc:.4f}\nF1 Score: {val_f1:.4f}\n{val_report}\n")
        model.train()

    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(batch["labels"].cpu().numpy())

    test_acc = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds)
    test_report = classification_report(test_targets, test_preds, digits=4)

    print(f"\nFinal Test Accuracy: {test_acc:.4f} | F1: {test_f1:.4f}")
    print(test_report)
    log.write(f"\nFinal Test Results:\nAccuracy: {test_acc:.4f}\nF1 Score: {test_f1:.4f}\n{test_report}\n")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete and model saved.")
