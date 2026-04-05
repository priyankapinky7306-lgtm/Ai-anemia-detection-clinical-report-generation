# 🚀 Deployment Guide — AnemiaAI on Streamlit Cloud

---

## Prerequisites
- GitHub account (free)
- Streamlit Cloud account (free): https://share.streamlit.io

---

## Step 1: Train the Model (Optional)
Run the Jupyter notebook to train and save the model:
```bash
jupyter notebook notebook/anemia_detection.ipynb
```
This saves `model/anemia_vgg16_model.h5`

> **Note:** The app runs in simulation mode without the model file.
> For demo/academic purposes, simulation mode works perfectly.

---

## Step 2: Push to GitHub

```bash
# Initialize repo
git init
git add .
git commit -m "Initial commit: AnemiaAI - Academic Project"

# Create repo on GitHub.com, then:
git remote add origin https://github.com/YOURUSERNAME/anemiaai.git
git branch -M main
git push -u origin main
```

**If model file is >100MB, use Git LFS:**
```bash
git lfs install
git lfs track "*.h5"
git add .gitattributes
git add model/anemia_vgg16_model.h5
git commit -m "Add model with LFS"
git push
```

---

## Step 3: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click **"New app"**
3. Select **"From existing repo"**
4. Choose your GitHub repository
5. Set **Main file path** to: `app.py`
6. Click **"Deploy!"**
7. Wait 2-5 minutes for the build
8. Your app is live at: `https://yourusername-anemiaai.streamlit.app`

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: tensorflow` | Missing dependency | Check requirements.txt has `tensorflow==2.15.0` |
| `MemoryError` | Free tier has 1GB RAM | Use `tensorflow-cpu` or reduce model |
| `FileNotFoundError: model/...h5` | Model not uploaded | App auto-falls back to simulation mode |
| `Pillow version mismatch` | Wrong Pillow version | Use `Pillow==10.2.0` in requirements.txt |
| Build timeout | Heavy packages | Add `tensorflow-cpu` only; remove GPU packages |
| Image upload error | Browser issue | Try Chrome; max upload is 200MB by default |

---

## Updating requirements.txt for Cloud

For **Streamlit Cloud free tier** (1 GB RAM), use lighter dependencies:
```
streamlit==1.32.0
tensorflow-cpu==2.15.0
numpy==1.26.4
Pillow==10.2.0
scikit-learn==1.4.1
pandas==2.2.1
opencv-python-headless==4.9.0.80
```

---

## Local Docker Deployment (Optional)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t anemiaai .
docker run -p 8501:8501 anemiaai
# Visit: http://localhost:8501
```
