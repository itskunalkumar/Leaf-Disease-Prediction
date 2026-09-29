<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:1b8a4b,100:7ed957&height=240&section=header&text=PlantAI&fontSize=84&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Plant%20Leaf%20Disease%20Prediction%20%E2%80%A2%20InceptionV3%20%2B%20Flask&descSize=20&descAlignY=60" width="100%"/>

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&duration=3200&pause=900&color=1B8A4B&center=true&vCenter=true&width=720&lines=Upload+a+leaf.+Get+a+diagnosis.;Deep+Learning+meets+Precision+Agriculture.;InceptionV3+Transfer+Learning+%F0%9F%A7%A0;Flask+Web+App+with+Treatment+Advice+%F0%9F%8C%BF" alt="Typing SVG" />
</a>

<br/>

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Transfer%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask&logoColor=white)
![Render](https://img.shields.io/badge/Deploy-Render-46E3B7?style=for-the-badge&logo=render&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)
[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://leaf-disease-prediction-gt.streamlit.app/)

**🍅 An end-to-end deep learning system that looks at a plant leaf and tells you what's wrong with it — and how to fix it.**

[✨ Features](#-features) • [🏗️ Architecture](#%EF%B8%8F-system-architecture) • [🧠 Model](#-model-architecture) • [🚀 Quick Start](#-quick-start) • [☁️ Deploy](#%EF%B8%8F-deployment) • [🗺️ Roadmap](#%EF%B8%8F-roadmap)

</div>

---

## 🌟 Overview

Crop diseases quietly destroy yields, and farmers often can't get an expert diagnosis in time. **PlantAI** closes that gap: a farmer uploads a photo of a tomato, potato or pepper leaf, a fine-tuned **InceptionV3** network classifies it in seconds, and the app returns the **disease name, confidence, symptoms, and recommended treatment**.

<div align="center">

| 📸 Input | 🧠 Brain | 📋 Output |
|:---:|:---:|:---:|
| Leaf image (JPG / PNG) | InceptionV3 (transfer learning) | Disease + confidence + treatment |

</div>

---

## ✨ Features

- 🔬 **Deep-learning diagnosis**: InceptionV3 backbone pre-trained on ImageNet, fine-tuned on tomato, potato and pepper leaf images
- ⚡ **Fast inference**: single forward pass, results in about a second on CPU
- 🌿 **Actionable advice**: every prediction is mapped to symptoms and treatment via `disease_info.json`
- 🖥️ **Clean web UI**: drag-and-drop upload, instant preview, readable result card
- ☁️ **Render-ready**: `gunicorn` + `requirements.txt` for one-click cloud deploys
- 🧩 **Modular**: swap the model, the disease knowledge base, or the UI independently

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    U([👨‍🌾 User]) -->|uploads leaf image| F[🌐 Flask Web App<br/>app.py]
    F --> P[🧪 Preprocessing<br/>resize 299×299 • scale • batch]
    P --> M[[🧠 InceptionV3<br/>model_inception.h5]]
    M -->|softmax probabilities| A{🎯 Argmax +<br/>Confidence}
    A --> K[(📚 disease_info.json<br/>symptoms • treatment)]
    K --> R[📋 Result Page<br/>disease • confidence • advice]
    R --> U

    style U fill:#1b8a4b,stroke:#0f2027,color:#fff
    style F fill:#0f2027,stroke:#7ed957,color:#fff
    style M fill:#ff6f00,stroke:#0f2027,color:#fff
    style K fill:#3776ab,stroke:#0f2027,color:#fff
    style R fill:#7ed957,stroke:#0f2027,color:#0f2027
```

### 🧱 Layered View

```text
                    ┌──────────────────────────────────────────┐
   PRESENTATION     │   HTML • CSS • JS   (templates/static)   │
                    └───────────────────┬──────────────────────┘
                                        │  HTTP (multipart/form-data)
                    ┌───────────────────▼──────────────────────┐
   APPLICATION      │        Flask  •  routes  •  gunicorn      │
                    └───────────────────┬──────────────────────┘
                                        │  numpy tensor (1,299,299,3)
                    ┌───────────────────▼──────────────────────┐
   INTELLIGENCE     │   InceptionV3  →  GAP  →  Dense  →  Softmax │
                    └───────────────────┬──────────────────────┘
                                        │  class index
                    ┌───────────────────▼──────────────────────┐
   KNOWLEDGE        │        disease_info.json  (advice)        │
                    └──────────────────────────────────────────┘
```

---

## 🧠 Model Architecture

```mermaid
flowchart TB
    IN[🖼️ Input 299×299×3] --> BASE

    subgraph BASE [❄️ InceptionV3 Base — ImageNet weights]
        direction TB
        S[Stem convs] --> IA[Inception A ×3]
        IA --> IB[Inception B ×5]
        IB --> IC[Inception C ×2]
    end

    BASE --> GAP[Global Average Pooling]
    GAP --> D1[Dense + ReLU]
    D1 --> DO[Dropout]
    DO --> OUT[🎯 Dense + Softmax<br/>disease classes]

    style BASE fill:#0f2027,stroke:#7ed957,color:#fff
    style OUT fill:#1b8a4b,stroke:#0f2027,color:#fff
```

**Why InceptionV3?** Its parallel multi-scale filters (1×1, 3×3, 5×5) capture both fine lesion texture and large-scale leaf patterns, which is exactly what disease spotting needs. Transfer learning means strong accuracy without training from scratch.

### 🔁 Training Pipeline

```mermaid
flowchart LR
    A[📁 Dataset] --> B[🧹 Clean + Split<br/>train / val / test]
    B --> C[🔀 Augmentation<br/>flip • rotate • zoom • shift]
    C --> D[❄️ Freeze base<br/>train new head]
    D --> E[🔥 Fine-tune<br/>top Inception blocks]
    E --> F[📈 Evaluate<br/>accuracy • confusion matrix]
    F --> G[💾 Save model_inception.h5]
```

---

## 📊 Results

The PlantAI model is built for real-time plant leaf disease classification using
InceptionV3 transfer learning.

| Metric | Result |
|---|---:|
| 🧠 Model | InceptionV3 |
| 🎯 Disease Classes | **10** |
| 🖼️ Input Image Size | **299 × 299 × 3** |
| 📦 Model Size | **~88 MB** |
| ⚡ Inference | **Single forward pass** |
| 🌿 Supported Crops | **Tomato • Potato • Pepper** |

### 🔬 Supported Disease Classes

The model can classify **10 disease/health categories**:

1. Pepper Bacterial Spot
2. Pepper Healthy
3. Potato Early Blight
4. Potato Healthy
5. Potato Late Blight
6. Tomato Mosaic Virus
7. Tomato Yellow Leaf Curl Virus
8. Tomato Bacterial Spot
9. Tomato Early Blight
10. Tomato Late Blight

### 🚀 Try PlantAI Live

**🌐 Live Demo:**
👉 https://leaf-disease-prediction-gt.streamlit.app/ and upload a leaf image. 🍅🥔🌶️

Upload a leaf image and get an AI-powered disease prediction with
confidence and treatment recommendations.

<div align="center">

---

## 🗂️ Project Structure

```text
Leaf-Disease-Prediction/
├── 📓 notebook.ipynb          # Data prep, training, evaluation
├── 🌐 app.py                  # Flask application
├── 📚 disease_info.json       # Symptoms + treatment per class
├── 🧠 model_inception.h5      # Trained InceptionV3 weights (~88 MB)
├── 📄 requirements.txt        # Pinned dependencies
├── 🎨 templates/              # HTML pages (upload, result)
├── 🖌️ static/                 # CSS, JS, sample images
├── 🖼️ assets/                 # README images & training plots
└── 📘 README.md
```

---

## 🚀 Quick Start

```bash
# 1️⃣ Clone
git clone https://github.com/itskunalkumar/Leaf-Disease-Prediction.git
cd Leaf-Disease-Prediction

# 2️⃣ Create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the app
python app.py

---

## 🔌 How a Prediction Works

```python
img   = load_img(path, target_size=(299, 299))     # 1. resize
x     = img_to_array(img) / 255.0                  # 2. normalize
x     = np.expand_dims(x, axis=0)                  # 3. add batch dim
probs = model.predict(x)[0]                        # 4. inference
idx   = int(np.argmax(probs))                      # 5. best class
info  = disease_info[class_names[idx]]             # 6. look up advice
```

---

## ☁️ Deployment

Ready for **Render** (or any WSGI host):

```bash
gunicorn app:app --timeout 120 --workers 1 --preload
```

<details>
<summary><b>💡 Deployment tips for the free tier</b></summary>

- The ~88 MB model plus TensorFlow is heavy on a 512 MB instance. Use **1 worker** and `--preload` so the model loads once.
- Raise `--timeout` (e.g. 120) so the first cold-start request isn't killed as a `WORKER TIMEOUT`.
- Load the model **once at startup**, never inside the request handler.
- Consider converting to **TensorFlow Lite** for a much smaller memory footprint.

</details>

---

## 🛠️ Tech Stack

<div align="center">

<img src="https://skillicons.dev/icons?i=python,tensorflow,flask,html,css,js,git,github&theme=dark" />

</div>

| Layer | Technology |
|:--|:--|
| Deep learning | TensorFlow / Keras, InceptionV3 |
| Data & viz | NumPy, Pandas, Matplotlib |
| Backend | Flask, Gunicorn |
| Frontend | HTML5, CSS3, JavaScript |
| Hosting | Render |

---

## 🗺️ Roadmap

- [x] InceptionV3 transfer-learning classifier
- [x] Flask web app with treatment recommendations
- [x] Render-ready deployment
- [ ] 🔍 Grad-CAM heatmaps to show *where* the model looked
- [ ] 📱 Mobile-friendly camera capture
- [ ] 🌾 Support for more crops (apple, grape, corn…)
- [ ] ⚡ TensorFlow Lite / ONNX for faster, lighter inference
- [ ] 🌐 Multilingual advice (Hindi, Bengali, …)
- [ ] 🐳 Docker image + CI/CD

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome!

1. 🍴 Fork the repo
2. 🌱 Create a branch: `git checkout -b feature/amazing-idea`
3. 💾 Commit: `git commit -m "Add amazing idea"`
4. 🚀 Push and open a Pull Request

---

## 👨‍💻 Author

<div align="center">

**Kunal Kumar**

[![GitHub](https://img.shields.io/badge/GitHub-itskunalkumar-181717?style=for-the-badge&logo=github)](https://github.com/itskunalkumar)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/kunalbtech2024/)

*If this project helped you, drop a ⭐ — it means a lot!*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:7ed957,50:1b8a4b,100:0f2027&height=140&section=footer" width="100%"/>

</div>
