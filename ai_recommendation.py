import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests

from treatment_engine import get_disease_profile


def _get(obj: Any, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _source_list(response) -> List[Dict[str, str]]:
    sources: List[Dict[str, str]] = []
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        metadata = _get(candidate, "grounding_metadata") or _get(candidate, "groundingMetadata")
        chunks = _get(metadata, "grounding_chunks", []) or _get(metadata, "groundingChunks", []) or []
        for chunk in chunks:
            web = _get(chunk, "web")
            url = _get(web, "uri") or _get(web, "url")
            title = _get(web, "title") or url
            if url and not any(s["url"] == url for s in sources):
                sources.append({"title": title, "url": url, "authority": _authority(url)})
        # Some SDK responses expose citations as annotations on text parts.
        content = _get(candidate, "content")
        parts = _get(content, "parts", []) or []
        for part in parts:
            annotations = _get(part, "annotations", []) or []
            for ann in annotations:
                url = _get(ann, "url")
                title = _get(ann, "title") or url
                if url and not any(s["url"] == url for s in sources):
                    sources.append({"title": title, "url": url, "authority": _authority(url)})
    sources.sort(key=lambda s: ({"high": 0, "medium": 1, "low": 2}.get(s["authority"], 2), s["title"] or ""))
    return sources[:12]


def _authority(url: str) -> str:
    u = (url or "").lower()
    high_domains = ("icar.gov.in", "ppqs.gov.in", "agricoop.gov.in", "gov.in", "ac.in", "edu", "fao.org", "cabi.org", "ipm.ucanr.edu", "extension.", "usda.gov")
    if any(d in u for d in high_domains):
        return "high"
    if any(d in u for d in (".edu", ".org")):
        return "medium"
    return "low"


def _schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "cause": {"type": "string"},
            "symptoms": {"type": "array", "items": {"type": "string"}},
            "immediate_actions": {"type": "array", "items": {"type": "string"}},
            "treatment_options": {"type": "array", "items": {"type": "string"}},
            "active_ingredients": {"type": "array", "items": {"type": "string"}},
            "application_requirements": {"type": "array", "items": {"type": "string"}},
            "safety_requirements": {"type": "array", "items": {"type": "string"}},
            "resistance_management": {"type": "array", "items": {"type": "string"}},
            "prevention": {"type": "array", "items": {"type": "string"}},
            "weather_note": {"type": "string"},
            "when_to_seek_expert": {"type": "array", "items": {"type": "string"}},
            "confidence_note": {"type": "string"},
            "dosage_status": {"type": "string"},
            "dosage_details": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "cause", "symptoms", "immediate_actions", "treatment_options", "active_ingredients", "application_requirements", "safety_requirements", "resistance_management", "prevention", "weather_note", "when_to_seek_expert", "confidence_note", "dosage_status", "dosage_details"],
    }


def google_grounded_recommendation(disease: str, crop: str, confidence: float, margin: float, country: str = "India", allow_treatment: bool = True, weather: Dict[str, Any] | None = None) -> Dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"status": "not_configured"}
    try:
        from google import genai
        from google.genai import types

        profile = get_disease_profile(disease)
        weather = weather or {"status": "not_requested"}
        treatment_rule = (
            "The classifier is uncertain. Do not recommend pesticide products, active ingredients, doses or spray schedules. "
            "Give confirmation steps and non-chemical management only."
            if not allow_treatment else
            "You may identify evidence-supported active ingredients/options, but never invent a dose, formulation, interval, "
            "registration status, PHI or REI. Exact numeric application instructions may only be included if a current authoritative "
            "source explicitly supports the exact crop + disease + country/product context. Otherwise say to follow the current label."
        )
        weather_context = json.dumps(weather, ensure_ascii=False)
        prompt = f"""
You are PlantAI, an evidence-grounded agricultural decision-support assistant.

MODEL
Crop: {crop}
Predicted condition: {disease}
Confidence: {confidence:.2f}%
Top-1/top-2 margin: {margin:.2f} percentage points
Region: {country}
Date: {datetime.now(timezone.utc).date().isoformat()}
Optional weather: {weather_context}

BASE KNOWLEDGE (use only as a starting point; current sources take precedence):
{json.dumps(profile, ensure_ascii=False)}

Use Google Search grounding. Prioritize current authoritative sources: for India, ICAR, Indian agricultural universities, government agriculture/plant-protection sources and official pesticide/label information where available; also use CABI, FAO, UC IPM, USDA and Cooperative Extension when useful. Avoid retailer pages as primary evidence.

{treatment_rule}

Return structured JSON matching the requested schema.
Rules:
- This is plant disease management, not human medicine.
- Start with integrated pest management and non-chemical controls.
- Do not treat model confidence as a laboratory diagnosis.
- For viral disease, do not claim fungicides cure the virus.
- For healthy predictions, do not recommend pesticide treatment.
- Separate active ingredient from trade brand. Prefer active ingredients.
- Never invent a product registration, dose, dilution, spray interval, PHI or REI.
- If an exact dose is not verified, dosage_status must be "not_verified" and dosage_details must say to follow the current local product label.
- Include PPE and other safety requirements whenever chemical treatment is discussed.
- Mention resistance management for fungicides/insecticides when relevant.
- Use the weather only as contextual risk information, not as a diagnosis.
- If evidence conflicts, say so and prefer the most authoritative/current source.
"""
        client = genai.Client(api_key=api_key)
        search_tool = types.Tool(google_search=types.GoogleSearch())
        config_kwargs = {
            "tools": [search_tool],
            "temperature": 0.1,
        }
        # Gemini 3 supports structured output together with Google Search.
        try:
            config_kwargs.update({"response_mime_type": "application/json", "response_schema": _schema()})
        except Exception:
            pass
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        raw = (getattr(response, "text", None) or "").strip()
        if not raw:
            return {"status": "empty_response"}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Graceful compatibility with SDK/model combinations that return plain text.
            data = {
                "summary": raw,
                "cause": "See grounded response.",
                "symptoms": [], "immediate_actions": [], "treatment_options": [], "active_ingredients": [],
                "application_requirements": [], "safety_requirements": [], "resistance_management": [],
                "prevention": [], "weather_note": "", "when_to_seek_expert": [],
                "confidence_note": "AI-generated grounded guidance; verify locally.",
                "dosage_status": "not_verified", "dosage_details": ["Follow the current local product label."],
            }
        sources = _source_list(response)
        return {"status": "ok", "mode": "AI + Google Search grounding", "data": data, "sources": sources, "searched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    except Exception as exc:
        return {"status": "error", "error": str(exc)[:400]}


def groq_recommendation(disease: str, crop: str, confidence: float, margin: float, country: str = "India", allow_treatment: bool = True, weather: Dict[str, Any] | None = None) -> Dict[str, Any]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"status": "not_configured", "provider": "Groq"}

    profile = get_disease_profile(disease)
    weather = weather or {"status": "not_requested"}
    treatment_rule = (
        "The classifier is uncertain. Do not recommend pesticide products, active ingredients, doses or spray schedules. "
        "Give confirmation steps and non-chemical management only."
        if not allow_treatment else
        "You may identify evidence-supported active ingredients/options, but never invent a dose, formulation, interval, "
        "registration status, PHI or REI. Say to follow the current local product label when exact evidence is unavailable."
    )
    prompt = f"""
You are PlantAI, an agricultural decision-support assistant for tomato leaf disease management.

Predicted condition: {disease}
Confidence: {confidence:.2f}%
Top-1/top-2 margin: {margin:.2f} percentage points
Region: {country}
Date: {datetime.now(timezone.utc).date().isoformat()}
Optional weather: {json.dumps(weather, ensure_ascii=False)}
Base disease profile: {json.dumps(profile, ensure_ascii=False)}

{treatment_rule}

Return only valid JSON with exactly these keys:
summary, cause, symptoms, immediate_actions, treatment_options, active_ingredients,
application_requirements, safety_requirements, resistance_management, prevention,
weather_note, when_to_seek_expert, confidence_note, dosage_status, dosage_details.
Array values must be arrays of strings. This is plant disease management, not human medicine.
Do not claim fungicides cure viral disease. Do not recommend pesticide treatment for a healthy prediction.
Never invent doses, product registrations, PHI or REI. Mention PPE when chemical treatment is discussed.
"""
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            },
            timeout=45,
        )
        response.raise_for_status()
        payload = response.json()
        raw = payload["choices"][0]["message"]["content"].strip()
        data = json.loads(raw)
        if weather.get("status") != "ok":
            data["weather_note"] = ""
        return {
            "status": "ok",
            "mode": "Groq AI recommendation",
            "data": data,
            "sources": [],
            "searched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }
    except requests.HTTPError as exc:
        detail = exc.response.text[:350] if exc.response is not None else str(exc)
        return {"status": "error", "provider": "Groq", "error": f"Groq API error: {detail}"}
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "error", "provider": "Groq", "error": f"Invalid Groq response: {exc}"}
    except requests.RequestException as exc:
        return {"status": "error", "provider": "Groq", "error": f"Groq connection error: {exc}"}
    except Exception as exc:
        return {"status": "error", "provider": "Groq", "error": str(exc)[:400]}


def local_fallback_recommendation(disease: str, display_name: str, crop: str, confidence: float, margin: float, allow_treatment: bool, weather: Dict[str, Any] | None = None) -> Dict[str, Any]:
    profile = get_disease_profile(disease)
    weather = weather or {"status": "not_requested"}
    if not allow_treatment:
        data = {
            "summary": "The image prediction is not sufficiently separated from alternatives for a chemical-treatment recommendation.",
            "cause": profile["cause"], "symptoms": profile["symptoms"],
            "immediate_actions": ["Retake a clear image in natural light", "Inspect nearby plants", "Do not apply a pesticide based only on this uncertain prediction"],
            "treatment_options": ["Use confirmation and non-chemical management first"], "active_ingredients": [],
            "application_requirements": ["No chemical application recommended until diagnosis is confirmed"],
            "safety_requirements": ["AI confidence is not a laboratory diagnosis"], "resistance_management": [],
            "prevention": profile["prevention"], "weather_note": _weather_note(weather),
            "when_to_seek_expert": [profile["expert"]], "confidence_note": f"Confidence {confidence:.2f}% with margin {margin:.2f} points.",
            "dosage_status": "not_verified", "dosage_details": ["No dose or spray schedule is provided without current label evidence."],
        }
    elif disease.endswith("healthy"):
        data = {
            "summary": "The model predicts a healthy plant; no disease treatment is indicated from this prediction.",
            "cause": profile["cause"], "symptoms": [], "immediate_actions": profile["ipm"], "treatment_options": [profile["chemical_note"]],
            "active_ingredients": [], "application_requirements": [], "safety_requirements": ["Avoid unnecessary pesticide use"],
            "resistance_management": [], "prevention": profile["prevention"], "weather_note": _weather_note(weather),
            "when_to_seek_expert": [profile["expert"]], "confidence_note": f"Image screening confidence {confidence:.2f}%.",
            "dosage_status": "not_applicable", "dosage_details": [],
        }
    else:
        data = {
            "summary": f"{display_name} requires integrated management and local confirmation where treatment decisions are consequential.",
            "cause": profile["cause"], "symptoms": profile["symptoms"], "immediate_actions": profile["ipm"],
            "treatment_options": [profile["chemical_note"]], "active_ingredients": profile["active_ingredients"],
            "application_requirements": ["Verify crop + disease registration", "Verify current formulation and label", "Follow label application method", "Verify PHI/REI where applicable"],
            "safety_requirements": ["Use label-specified PPE", "Avoid spray drift and exposure", "Keep products away from children/animals", "Never mix products unless the label permits it"],
            "resistance_management": ["Rotate modes of action where current local guidance recommends it", "Do not repeat the same mode of action unnecessarily"],
            "prevention": profile["prevention"], "weather_note": _weather_note(weather), "when_to_seek_expert": [profile["expert"]],
            "confidence_note": f"Image screening confidence {confidence:.2f}%; not a laboratory diagnosis.",
            "dosage_status": "not_verified", "dosage_details": ["No universal dose is provided. Verify the current local product label or official agricultural recommendation for the exact crop, disease, formulation and region."],
        }
    return {"status": "fallback", "mode": "Curated safety fallback", "data": data, "sources": [], "searched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}


def _weather_note(weather: Dict[str, Any]) -> str:
    if weather.get("status") != "ok":
        return ""
    return f"{weather.get('city')}: {weather.get('temperature_c')}°C, humidity {weather.get('humidity_pct')}%, precipitation {weather.get('precipitation_mm')} mm. Risk context: {weather.get('risk')}." + (" Reasons: " + ", ".join(weather.get('reasons', [])) + "." if weather.get('reasons') else "")
