import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Settings
MODEL_NAME = "cybersectony/phishing-email-detection-distilbert_v2.4.1"
DATA_PATH = "data/CEAS_08.csv"  # Change this if needed
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# 1. Load Dataset
# 1. Load Dataset
df = pd.read_csv(DATA_PATH)
df = df.dropna()

# Optional: Clean weird newlines or stray quote marks
df["body"] = df["body"].astype(str).str.replace(r"\s+", " ", regex=True)
df["subject"] = df["subject"].astype(str).str.replace(r"\s+", " ", regex=True)

# Combine subject + body into a single input field
df["text"] = df["subject"] + " " + df["body"]

# Labels should be 0 or 1
df["label"] = df["label"].astype(int)

# Sanity check
print(df[["text", "label"]].head())


train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2
)

# 2. Tokenizer & Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# 3. PyTorch Dataset
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

# 4. Load Model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# 5. Optimizer + Scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

num_training_steps = EPOCHS * len(train_loader)
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# 6. Training Loop
from sklearn.metrics import accuracy_score, f1_score

# 6. Train Model
model.train()
NUM_EPOCHS = 3
for epoch in range(NUM_EPOCHS):
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

    # 🔍 Validation after each epoch
    model.eval()
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)

            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(batch["labels"].cpu().numpy())

    # ✅ Metrics
    val_accuracy = accuracy_score(val_targets, val_preds)
    val_f1 = f1_score(val_targets, val_preds)

    print(f"✅ Epoch {epoch + 1} Results:")
    print(f"   🧠 Loss: {running_loss / len(train_loader):.4f}")
    print(f"   🎯 Accuracy: {val_accuracy:.4f}")
    print(f"   📈 F1 Score: {val_f1:.4f}\n")

    model.train()  # Set model back to train


# 7. Save Model (Optional)
# model.save_pretrained("model/")
# tokenizer.save_pretrained("model/")

print("✅ Training complete and model saved!")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

# Show report
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, digits=4))
