# COS720-Project

## 🛡️ Phishing Email Detection System (AI-Powered)

This project uses a pre-trained Hugging Face model to detect phishing emails based on content, sender, and subject lines. It supports GPU acceleration for fast and efficient training/inference.

---

## ⚙️ Environment Setup (Local GPU)

### ✅ Requirements
- Python 3.8+
- NVIDIA GPU with CUDA support
- VS Code (recommended)
- CUDA Toolkit + cuDNN installed (for PyTorch GPU support)

---

### 🧰 1. Clone the Repository
```bash
git clone https://github.com/your-username/phishing-detector.git
cd phishing-detector

### 2. Set up virtual env
 Create virtual environment
python -m venv venv

# Activate virtual environment
Windows
venv\Scripts\activate
```

If you are able to load in you will see a (venv) to the far left of your terminal

### 3. Install dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets scikit-learn pandas tqdm
```

### 4. Check GPU access (for faster training)
```
python scripts/check_gpu.py
```

On success it will indicate CUDA support and what GPU you are currently running.

### 5. Load base model
```
python scripts/load_model.py
```
The model will download and a message will appear indicating success upon loading.




