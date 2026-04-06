# 🩸 AI-Based Anemia Detection System
### Major Project | Deep Learning + NLP + Streamlit

---

## 📁 Project Structure

```
anemia_project/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── model/
│   └── anemia_vgg16_model.h5      # Trained VGG16 model (generated after training)
├── notebook/
│   └── anemia_detection.ipynb     # Complete Jupyter Notebook
├── docs/
│   ├── generate_report.py         # PDF report generator script
│   └── anemia_detection_report.pdf # Complete academic report PDF
└── assets/                        # Generated plots & sample outputs
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit App
```bash
streamlit run app.py
```

### 3. Run Jupyter Notebook
```bash
jupyter notebook notebook/anemia_detection.ipynb
```

### 4. Generate PDF Report
```bash
python docs/generate_report.py
```

---

## 📦 Dataset

Download the Blood Cell dataset from Kaggle:
- **URL:** https://www.kaggle.com/datasets/paultimothymooney/blood-cells
- Place in `dataset/blood_cells/train/` and `dataset/blood_cells/test/`
- The notebook also auto-creates a **synthetic dataset** for testing

---

## 🌐 Deploy on Streamlit Cloud

1. Push this folder to GitHub
2. Visit https://share.streamlit.io
3. Connect GitHub → Select repo → Set `app.py` as main file
4. Click Deploy

---

## ⚠️ Note
- The Streamlit app runs in **simulation mode** if no trained model is present
- Train the model using the Jupyter Notebook first, then place `anemia_vgg16_model.h5` in `model/`
- Modify `app.py` to load the actual model by uncommenting the TensorFlow prediction section

---

## 📋 Technology Stack
| Component | Technology |
|-----------|-----------|
| Deep Learning | TensorFlow / Keras |
| Pre-trained Model | VGG16 (ImageNet) |
| Web App | Streamlit |
| Image Processing | OpenCV, Pillow |
| Report Generation | Template NLP |
| Data Science | NumPy, Pandas, Matplotlib |

---
**Major Project | 2025-2026**
