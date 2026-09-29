import os
from typing import Dict, Any


def get_weather(city: str) -> Dict[str, Any]:
    """Optional no-key Open-Meteo lookup. Failure is non-fatal."""
    city = (city or "").strip()
    if not city:
        return {"status": "not_requested"}
    try:
        import requests
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=8,
        )
        geo.raise_for_status()
        results = geo.json().get("results") or []
        if not results:
            return {"status": "not_found", "city": city}
        place = results[0]
        forecast = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
                "timezone": "auto",
            },
            timeout=8,
        )
        forecast.raise_for_status()
        current = forecast.json().get("current", {})
        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        precipitation = current.get("precipitation")
        risk = "normal"
        reasons = []
        if humidity is not None and humidity >= 80:
            risk = "elevated"
            reasons.append("high humidity")
        if precipitation is not None and precipitation > 0:
            risk = "elevated"
            reasons.append("recent/current precipitation")
        if temp is not None and 18 <= temp <= 30 and humidity is not None and humidity >= 80:
            risk = "high"
            reasons.append("warm and humid conditions")
        return {
            "status": "ok",
            "city": place.get("name", city),
            "country": place.get("country"),
            "temperature_c": temp,
            "humidity_pct": humidity,
            "precipitation_mm": precipitation,
            "risk": risk,
            "reasons": reasons,
            "source": "Open-Meteo",
        }
    except Exception as exc:
        return {"status": "error", "city": city, "error": str(exc)[:180]}
