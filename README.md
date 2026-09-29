# PlantAI — Tomato / Plant Leaf Disease Detection

Streamlit deployment of a 10-class InceptionV3 plant leaf disease classifier with:

- Single-pass 299×299 inference
- Cached TensorFlow model loading
- Image-quality checks
- Top-3 predictions and confidence margin
- Conservative local treatment/IPM guidance
- Optional weather context using Open-Meteo
- Optional Groq AI recommendations

## Streamlit Community Cloud

Entrypoint: `streamlit_app.py`

Python: 3.11

Secrets (optional):

```toml
GROQ_API_KEY = "your-key"
GROQ_MODEL = "qwen/qwen3.8-27b"
APP_COUNTRY = "India"
```

Select the recommendation engine in the app sidebar. Groq provides optional AI recommendations without Google Search grounding.

Do not commit `.streamlit/secrets.toml`.

## Local run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Model

Place `model_inception.h5` in the same directory as `streamlit_app.py`.

The model must have exactly 10 outputs matching the class list in `streamlit_app.py`.
