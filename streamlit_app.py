import io
import os
import json
from pathlib import Path

# TensorFlow CPU/thread settings must be set before TensorFlow work starts.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input

from image_quality import assess_image_bytes
from treatment_engine import get_disease_profile
from ai_recommendation import google_grounded_recommendation, local_fallback_recommendation
from weather_engine import get_weather

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_inception.h5"
MAX_UPLOAD_MB = 8

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
]

DISPLAY_NAMES = {
    "Pepper__bell___Bacterial_spot": "Pepper Bacterial Spot",
    "Pepper__bell___healthy": "Pepper Healthy",
    "Potato___Early_blight": "Potato Early Blight",
    "Potato___healthy": "Potato Healthy",
    "Potato___Late_blight": "Potato Late Blight",
    "Tomato__Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "Tomato_Bacterial_spot": "Tomato Bacterial Spot",
    "Tomato_Early_blight": "Tomato Early Blight",
    "Tomato_Late_blight": "Tomato Late Blight",
}

CROP_NAMES = {
    key: (
        "Bell Pepper" if key.startswith("Pepper")
        else "Potato" if key.startswith("Potato")
        else "Tomato"
    )
    for key in CLASS_NAMES
}


def configure_tensorflow() -> None:
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        # TensorFlow may already have initialized its runtime in some environments.
        pass


configure_tensorflow()


@st.cache_resource(show_spinner=False)
def load_prediction_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = load_model(MODEL_PATH, compile=False)
    outputs = int(model.output_shape[-1])
    if outputs != len(CLASS_NAMES):
        raise RuntimeError(
            f"Model has {outputs} output classes; expected {len(CLASS_NAMES)}."
        )
    return model


def normalize_predictions(predictions: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.ndim != 1:
        predictions = predictions.reshape(-1)

    total = float(np.sum(predictions))
    if np.any(predictions < 0) or not np.isclose(total, 1.0, atol=0.05):
        shifted = predictions - np.max(predictions)
        exp = np.exp(np.clip(shifted, -50, 50))
        predictions = exp / np.sum(exp)
    else:
        predictions = predictions / max(total, 1e-8)
    return predictions


def confidence_band(confidence: float, margin: float, entropy: float):
    if confidence >= 85 and margin >= 10 and entropy < 1.1:
        return "high", "High model confidence"
    if confidence >= 65 and margin >= 5:
        return "medium", "Moderate model confidence"
    return "low", "Uncertain prediction — confirm before treatment"


def predict_image(model, image: Image.Image):
    image = image.convert("RGB").resize((299, 299), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32)
    array = preprocess_input(array)
    array = np.expand_dims(array, axis=0)

    # One inference only: avoids the double-memory/CPU cost of test-time augmentation.
    predictions = normalize_predictions(model.predict(array, verbose=0)[0])
    order = np.argsort(predictions)[::-1]

    entropy = float(
        -np.sum(
            np.clip(predictions, 1e-9, 1.0)
            * np.log(np.clip(predictions, 1e-9, 1.0))
        )
    )

    top_predictions = []
    for index in order[:3]:
        key = CLASS_NAMES[int(index)]
        top_predictions.append(
            {
                "key": key,
                "name": DISPLAY_NAMES[key],
                "confidence": round(float(predictions[index]) * 100, 2),
                "crop": CROP_NAMES[key],
            }
        )

    best = top_predictions[0]
    margin = (
        best["confidence"] - top_predictions[1]["confidence"]
        if len(top_predictions) > 1
        else best["confidence"]
    )
    band, label = confidence_band(best["confidence"], margin, entropy)

    return {
        "raw_disease": best["key"],
        "disease": best["name"],
        "crop": best["crop"],
        "confidence": best["confidence"],
        "margin": round(margin, 2),
        "entropy": round(entropy, 3),
        "confidence_band": band,
        "confidence_label": label,
        "top_predictions": top_predictions,
    }


def get_secret(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        value = st.secrets.get(name, default)
        return str(value) if value else default
    except Exception:
        return default


def render_list(title: str, items):
    if not items:
        return
    st.markdown(f"**{title}**")
    for item in items:
        st.markdown(f"- {item}")


def render_recommendation(result: dict):
    if not result:
        return

    if result.get("status") == "ok":
        data = result.get("data", {})
        mode = result.get("mode", "AI recommendation")
        st.success(f"{mode}")

        st.markdown(f"### {data.get('summary', 'Recommendation')}")
        if data.get("cause"):
            st.write(data["cause"])

        cols = st.columns(2)
        with cols[0]:
            render_list("Symptoms", data.get("symptoms", []))
            render_list("Immediate actions", data.get("immediate_actions", []))
            render_list("Treatment options", data.get("treatment_options", []))
            render_list("Active ingredients", data.get("active_ingredients", []))
        with cols[1]:
            render_list("Application requirements", data.get("application_requirements", []))
            render_list("Safety requirements", data.get("safety_requirements", []))
            render_list("Resistance management", data.get("resistance_management", []))
            render_list("Prevention", data.get("prevention", []))

        if data.get("weather_note"):
            st.info(data["weather_note"])
        render_list("When to seek expert help", data.get("when_to_seek_expert", []))

        if data.get("dosage_status"):
            st.caption(f"Dosage status: {data['dosage_status']}")
        render_list("Dosage notes", data.get("dosage_details", []))

        if data.get("confidence_note"):
            st.caption(data["confidence_note"])

        sources = result.get("sources", [])
        if sources:
            st.markdown("### Sources")
            for source in sources:
                title = source.get("title") or source.get("url")
                url = source.get("url")
                if url:
                    st.markdown(f"- [{title}]({url})")

    elif result.get("status") == "not_configured":
        st.info("Gemini is not configured. The app is using its local evidence-oriented recommendation engine.")
    else:
        st.warning(
            "Grounded AI was unavailable, so the app is showing the local recommendation instead."
        )


def main():
    st.set_page_config(
        page_title="PlantAI — Leaf Disease Detection",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;600;700;800&display=swap');
        :root {
            --forest: #07130f;
            --forest-deep: #030907;
            --mint: #a5f3c7;
            --green: #39e58c;
            --cyan: #72e5e0;
            --amber: #f4c96b;
            --ink: #eaf8ef;
            --muted: #91aaa0;
            --line: rgba(165, 243, 199, .16);
            --glass: rgba(10, 31, 24, .7);
        }
        html, body, [class*='css'] {font-family: 'Manrope', sans-serif;}
        .stApp {
            color: var(--ink);
            background:
                linear-gradient(135deg, rgba(57, 229, 140, .05), transparent 38%),
                radial-gradient(circle at 82% 8%, rgba(114, 229, 224, .09), transparent 25%),
                var(--forest-deep);
        }
        .stApp::before {
            content: '';
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: .3;
            background-image: linear-gradient(rgba(165,243,199,.045) 1px, transparent 1px), linear-gradient(90deg, rgba(165,243,199,.045) 1px, transparent 1px);
            background-size: 42px 42px;
            mask-image: linear-gradient(to bottom, black, transparent 78%);
        }
        .block-container {max-width: 1380px; padding: 2.2rem 3rem 5rem; position: relative;}
        [data-testid='stSidebar'] {background: rgba(3, 15, 11, .94); border-right: 1px solid var(--line);}
        [data-testid='stSidebar'] > div:first-child {padding-top: 2rem;}
        [data-testid='stSidebar'] h2, [data-testid='stSidebar'] label {color: var(--mint);}
        [data-testid='stSidebar'] .stCaption {color: var(--muted);}
        .hero {
            min-height: 230px;
            padding: 1.6rem 2rem 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid var(--line);
            border-radius: 26px;
            position: relative;
            overflow: hidden;
            transform-style: preserve-3d;
            background: linear-gradient(110deg, rgba(14, 48, 34, .88), rgba(7, 24, 19, .55));
            box-shadow: 0 26px 80px rgba(0, 0, 0, .35), inset 0 1px 0 rgba(255,255,255,.08);
            animation: hero-in .8s cubic-bezier(.22, 1, .36, 1) both;
        }
        .hero::before {
            content: '';
            position: absolute;
            width: 520px;
            height: 180px;
            right: -120px;
            top: -80px;
            border: 1px solid rgba(114,229,224,.24);
            border-radius: 50%;
            transform: rotate(-18deg) translateZ(30px);
            box-shadow: 0 0 80px rgba(57,229,140,.12), inset 0 0 30px rgba(114,229,224,.06);
        }
        .hero::after {
            content: 'LIVE / VISION ENGINE';
            position: absolute;
            right: 1.8rem;
            bottom: 1.3rem;
            color: rgba(165,243,199,.58);
            font: 500 .66rem 'DM Mono', monospace;
            letter-spacing: .18em;
        }
        .hero h1 {font-size: clamp(2.4rem, 5vw, 4.8rem); line-height: .95; letter-spacing: -.07em; margin: 0 0 .8rem; color: var(--ink); text-shadow: 0 12px 34px rgba(0,0,0,.3);}
        .hero h1 span {color: var(--green);}
        .hero p {max-width: 580px; font-size: 1rem; line-height: 1.7; color: var(--muted); margin: 0;}
        .hero-tag {color: var(--amber); font: 500 .7rem 'DM Mono', monospace; letter-spacing: .18em; text-transform: uppercase; margin-bottom: 1.2rem;}
        .upload-label {color: var(--mint); font: 500 .7rem 'DM Mono', monospace; letter-spacing: .16em; text-transform: uppercase; margin: .4rem 0 .65rem;}
        [data-testid='stFileUploader'] {border: 1px solid var(--line); border-radius: 20px; padding: .5rem; background: rgba(10,31,24,.58); box-shadow: 0 18px 50px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.06);}
        [data-testid='stFileUploaderDropzone'] {border: 1px dashed rgba(57,229,140,.42); background: rgba(57,229,140,.035); border-radius: 15px; transition: .3s ease;}
        [data-testid='stFileUploaderDropzone']:hover {border-color: var(--green); background: rgba(57,229,140,.1); transform: translateY(-2px);}
        [data-testid='stImage'] img {border-radius: 18px; border: 1px solid var(--line); box-shadow: 0 24px 54px rgba(0,0,0,.4);}
        .stButton > button {border: 1px solid rgba(57,229,140,.52); border-radius: 12px; background: linear-gradient(135deg, #35d783, #0c9361); color: #03130b; font-weight: 800; box-shadow: 0 12px 28px rgba(57,229,140,.16); transition: transform .25s ease, box-shadow .25s ease;}
        .stButton > button:hover {transform: translateY(-3px); box-shadow: 0 17px 34px rgba(57,229,140,.26); border-color: var(--mint);}
        [data-testid='stMetric'] {padding: 1.1rem 1.25rem; border: 1px solid var(--line); border-radius: 16px; background: linear-gradient(145deg, rgba(17,52,39,.8), rgba(7,24,18,.62)); box-shadow: 10px 14px 0 rgba(3,9,7,.34), 0 16px 35px rgba(0,0,0,.22); transform: perspective(700px) rotateX(2deg);}
        [data-testid='stMetricLabel'] {color: var(--muted); font: 500 .68rem 'DM Mono', monospace; letter-spacing: .1em; text-transform: uppercase;}
        [data-testid='stMetricValue'] {color: var(--mint); font-weight: 800; letter-spacing: -.04em;}
        [data-testid='stProgressBar'] > div > div {background: linear-gradient(90deg, var(--green), var(--cyan));}
        [data-testid='stExpander'] {border: 1px solid var(--line); border-radius: 15px; background: rgba(9, 29, 22, .52);}
        .stAlert {border-radius: 14px; border: 1px solid var(--line); background: rgba(12, 40, 29, .64);}
        hr {border-color: var(--line); margin: 2rem 0;}
        h2, h3 {color: var(--mint); letter-spacing: -.035em;}
        code, .stCaption {font-family: 'DM Mono', monospace;}
        .empty-state {padding: 2.6rem 2rem; margin-top: 1rem; border: 1px solid var(--line); border-radius: 22px; background: rgba(9, 29, 22, .52); box-shadow: 0 20px 60px rgba(0,0,0,.18);}
        .empty-state h2 {margin: 0 0 .5rem; color: var(--ink);}
        .empty-state p {color: var(--muted); max-width: 560px; line-height: 1.7;}
        .signal-row {display: flex; gap: .6rem; flex-wrap: wrap; margin-top: 1.3rem;}
        .signal {border: 1px solid var(--line); border-radius: 999px; padding: .4rem .75rem; color: var(--mint); font: 500 .68rem 'DM Mono', monospace; letter-spacing: .05em; background: rgba(57,229,140,.05);}
        @keyframes hero-in {from {opacity: 0; transform: translateY(18px) perspective(900px) rotateX(3deg);} to {opacity: 1; transform: translateY(0) perspective(900px) rotateX(0);}}
        @media (max-width: 800px) {.block-container {padding: 1rem 1rem 3rem;} .hero {min-height: 260px; padding: 1.3rem;} .hero::after {right: 1.3rem; bottom: .9rem;} [data-testid='stMetric'] {margin-bottom: .6rem;}}
        </style>
        <div class="hero">
            <div class="hero-tag">Plant health intelligence / 01</div>
            <h1>Plant<span>AI</span></h1>
            <p>See the signal in every leaf. Upload a field image and turn visual symptoms into a clear, evidence-oriented next move.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Settings")
        country = st.text_input("Country / region", value=get_secret("APP_COUNTRY", "India"))
        city = st.text_input("City (optional)", value="")
        use_grounded_ai = st.checkbox(
            "Use Gemini + Google Search grounding",
            value=False,
            help="Enable this only after the disease prediction is complete. It makes a separate web-grounded AI request.",
        )
        st.divider()
        st.caption("Model: InceptionV3")
        st.caption("Input: 299 × 299 RGB")
        st.caption("Classes: 10")
        st.caption("Inference: single pass")

    st.markdown("<div class='upload-label'>Input surface / leaf image</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a clear leaf image",
        type=["jpg", "jpeg", "png"],
        max_upload_size=MAX_UPLOAD_MB,
        help="Use a close, well-lit photo of the affected leaf. Maximum 8 MB.",
    )

    if uploaded is None:
        st.markdown(
            """
            <div class="empty-state">
                <h2>Bring a leaf into focus</h2>
                <p>Drop a clear, well-lit image above. The vision engine will score the crop, surface the top three conditions, and assemble a conservative management path.</p>
                <div class="signal-row">
                    <span class="signal">299 × 299 RGB</span>
                    <span class="signal">10 conditions</span>
                    <span class="signal">single-pass inference</span>
                    <span class="signal">IPM guidance</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    image_bytes = uploaded.getvalue()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    left, right = st.columns([1, 1.3], gap="large")
    with left:
        st.image(image, caption=uploaded.name, use_container_width=True)
        quality = assess_image_bytes(image_bytes)
        if quality["quality"] == "poor":
            st.warning("Image quality is poor. A clearer photo may improve prediction reliability.")
        elif quality["warnings"]:
            st.info("Image quality notes: " + " ".join(quality["warnings"]))

    with right:
        if st.button("🔍 Analyze disease", type="primary", use_container_width=True):
            try:
                with st.spinner("Loading model and analyzing the leaf…"):
                    model = load_prediction_model()
                    prediction = predict_image(model, image)
                    profile = get_disease_profile(prediction["raw_disease"])

                    weather = get_weather(city) if city.strip() else {"status": "not_requested"}
                    allow_treatment = (
                        prediction["confidence_band"] != "low"
                        and quality["quality"] != "poor"
                    )
                    local = local_fallback_recommendation(
                        prediction["raw_disease"],
                        prediction["disease"],
                        prediction["crop"],
                        prediction["confidence"],
                        prediction["margin"],
                        allow_treatment,
                        weather,
                    )

                    st.session_state["analysis"] = {
                        "prediction": prediction,
                        "quality": quality,
                        "profile": profile,
                        "weather": weather,
                        "local_recommendation": local,
                        "ai_recommendation": None,
                        "country": country.strip() or "India",
                        "city": city.strip(),
                    }
                st.success("Analysis complete.")
            except Exception as exc:
                st.error("The image could not be analyzed.")
                st.exception(exc)

    analysis = st.session_state.get("analysis")
    if not analysis:
        return

    prediction = analysis["prediction"]
    st.divider()
    st.subheader("Prediction")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Condition", prediction["disease"])
    metric_cols[1].metric("Crop", prediction["crop"])
    metric_cols[2].metric("Confidence", f"{prediction['confidence']:.2f}%")
    metric_cols[3].metric("Top-2 margin", f"{prediction['margin']:.2f} pts")

    if prediction["confidence_band"] == "high":
        st.success(prediction["confidence_label"])
    elif prediction["confidence_band"] == "medium":
        st.warning(prediction["confidence_label"])
    else:
        st.error(prediction["confidence_label"])

    st.markdown("### Top predictions")
    for item in prediction["top_predictions"]:
        st.progress(
            min(item["confidence"] / 100.0, 1.0),
            text=f"{item['name']} — {item['confidence']:.2f}%",
        )

    with st.expander("Image quality", expanded=False):
        st.json(analysis["quality"])

    with st.expander("Disease profile", expanded=False):
        profile = analysis["profile"]
        st.write(f"**Category:** {profile.get('category', 'Plant health condition')}")
        st.write(f"**Cause:** {profile.get('cause', '')}")
        render_list("Symptoms", profile.get("symptoms", []))
        render_list("Integrated management", profile.get("ipm", []))
        render_list("Active ingredients / options", profile.get("active_ingredients", []))
        if profile.get("chemical_note"):
            st.info(profile["chemical_note"])
        render_list("Prevention", profile.get("prevention", []))
        if profile.get("expert"):
            st.warning(profile["expert"])

    weather = analysis["weather"]
    if weather.get("status") == "ok":
        with st.expander("Current weather context", expanded=False):
            st.write(
                f"{weather.get('city', city)}: {weather.get('temperature_c')} °C, "
                f"{weather.get('humidity_pct')}% humidity, "
                f"{weather.get('precipitation_mm')} mm precipitation. "
                f"Risk context: **{weather.get('risk', 'normal')}**."
            )
            if weather.get("reasons"):
                st.write("; ".join(weather["reasons"]).capitalize() + ".")
            st.caption("Weather source: Open-Meteo. Weather is contextual and does not diagnose disease.")

    st.subheader("Treatment & management guidance")
    render_recommendation(analysis["local_recommendation"])

    if use_grounded_ai:
        api_key = get_secret("GEMINI_API_KEY")
        if not api_key:
            st.warning("Add GEMINI_API_KEY in Streamlit Secrets to enable Google-grounded recommendations.")
        elif st.button("🌐 Generate Google-grounded recommendation", use_container_width=True):
            with st.spinner("Searching current agricultural sources and generating guidance…"):
                # The function reads GEMINI_API_KEY from the environment, so expose the
                # Streamlit secret only for this process without storing it in the repo.
                os.environ["GEMINI_API_KEY"] = api_key
                os.environ["GEMINI_MODEL"] = get_secret("GEMINI_MODEL", "gemini-3.8-flash")
                ai_result = google_grounded_recommendation(
                    prediction["raw_disease"],
                    prediction["crop"],
                    prediction["confidence"],
                    prediction["margin"],
                    analysis["country"],
                    prediction["confidence_band"] != "low" and analysis["quality"]["quality"] != "poor",
                    analysis["weather"],
                )
                analysis["ai_recommendation"] = ai_result
                st.session_state["analysis"] = analysis

        if analysis.get("ai_recommendation"):
            st.divider()
            st.subheader("Google-grounded recommendation")
            render_recommendation(analysis["ai_recommendation"])

    st.caption(
        "Important: This is decision-support software, not a laboratory diagnosis. "
        "Verify the diagnosis and follow current local agricultural extension advice and product labels before applying pesticides."
    )


if __name__ == "__main__":
    main()
