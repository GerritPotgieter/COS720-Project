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

### 3. Install dependancies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets scikit-learn pandas tqdm


### 4. Check GPU access (for faster training)
python scripts/check_gpu.py


### 5. Load base model
python scripts/load_model.py





