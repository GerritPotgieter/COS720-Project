import pandas as pd
import os
from tqdm import tqdm
import csv


# Directories
RAW_DIR = "data/raw"
OUTPUT_PATH = "data/processed_phishing_emails.csv"

# Dataset names
simple_datasets = ["Enron.csv", "Ling.csv"]
rich_datasets = ["CEAS_08.csv", "Nazario.csv", "Nigerian_Fraud.csv", "SpamAssasin.csv"]

# Load and process simple datasets (subject, body, label)
simple_rows = []
print("Processing simple datasets...")
for filename in tqdm(simple_datasets):
    path = os.path.join(RAW_DIR, filename)
    df = pd.read_csv(path)
    df = df.dropna(subset=["subject", "body", "label"])

    df["text_combined"] = (
        "[SUBJECT] " + df["subject"].astype(str) + " " +
        "[BODY] " + df["body"].astype(str)
    ).str.replace(r"\s+", " ", regex=True)

    simple_rows.append(df[["text_combined", "label"]])

# Load and process rich datasets (sender, receiver, date, subject, body, url, label)
rich_rows = []
print("Processing rich datasets...")
for filename in tqdm(rich_datasets):
    path = os.path.join(RAW_DIR, filename)
    df = pd.read_csv(path)
    df = df.dropna(subset=["sender", "receiver", "date", "subject", "body", "label"])

    df["text_combined"] = (
        "[SENDER] " + df["sender"].astype(str) + " " +
        "[RECEIVER] " + df["receiver"].astype(str) + " " +
        "[DATE] " + df["date"].astype(str) + " " +
        "[SUBJECT] " + df["subject"].astype(str) + " " +
        "[BODY] " + df["body"].astype(str) + " " +
        "[URL] " + df["urls"].astype(str)
    ).str.replace(r"\s+", " ", regex=True)

    rich_rows.append(df[["text_combined", "label"]])

# Combine and shuffle all
print("Combining and shuffling datasets...")
combined_df = pd.concat(simple_rows + rich_rows, ignore_index=True)
combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# Save to file
os.makedirs("data", exist_ok=True)
combined_df.to_csv(OUTPUT_PATH, index=False ,  quoting=csv.QUOTE_ALL)
print(f"Saved preprocessed dataset to {OUTPUT_PATH}")
