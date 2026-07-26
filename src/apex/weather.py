"""Tier 2 — race-day weather from Open-Meteo.

No API key, no account, free for non-commercial use. Fetched at build time and cached
into the payload, so the dashboard never calls it.

Why it matters more than it looks: rain compresses the field and shifts weight from the
car to the driver, which is a different regime rather than a nuisance parameter. van
Kesteren & Bergkamp fit explicit wet-race random slopes for exactly this reason. Even in
the dry, track temperature moved across a 37 °C range this season, and the degradation
slopes in Layer 0 are pooled across all of it.

This module supplies the forecast and a rain probability. It does **not** yet re-fit the
model in the wet — with one wet race in 2026 there is not enough in-regulation evidence
to estimate a wet regime, so the honest use is to widen the forecast and say so, which is
what `uncertainty_multiplier` is for.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import requests

ENDPOINT = "https://api.open-meteo.com/v1/forecast"

# Circuit coordinates for the 2026 calendar. Only what the schedule actually visits.
CIRCUIT_COORDS = {
    "Australian Grand Prix": (-37.8497, 144.9680),
    "Chinese Grand Prix": (31.3389, 121.2200),
    "Japanese Grand Prix": (34.8431, 136.5407),
    "Miami Grand Prix": (25.9581, -80.2389),
    "Canadian Grand Prix": (45.5000, -73.5228),
    "Monaco Grand Prix": (43.7347, 7.4206),
    "Barcelona Grand Prix": (41.5700, 2.2611),
    "Spanish Grand Prix": (41.5700, 2.2611),
    "Austrian Grand Prix": (47.2197, 14.7647),
    "British Grand Prix": (52.0786, -1.0169),
    "Belgian Grand Prix": (50.4372, 5.9714),
    "Hungarian Grand Prix": (47.5789, 19.2486),
    "Dutch Grand Prix": (52.3888, 4.5409),
    "Italian Grand Prix": (45.6156, 9.2811),
    "Azerbaijan Grand Prix": (40.3725, 49.8533),
    "Singapore Grand Prix": (1.2914, 103.8640),
    "United States Grand Prix": (30.1328, -97.6411),
    "Mexico City Grand Prix": (19.4042, -99.0907),
    "São Paulo Grand Prix": (-23.7036, -46.6997),
    "Las Vegas Grand Prix": (36.1147, -115.1728),
    "Qatar Grand Prix": (25.4900, 51.4542),
    "Abu Dhabi Grand Prix": (24.4672, 54.6031),
}


@dataclass
class RaceWeather:
    event: str
    date: str
    available: bool
    rain_probability: float | None = None      # 0-1, max over the race window
    precipitation_mm: float | None = None
    temperature_c: float | None = None
    wind_kph: float | None = None
    note: str = ""

    @property
    def is_wet_risk(self) -> bool:
        return (self.rain_probability or 0) >= 0.30


def fetch(event: str, date: str, hour_from: int = 12, hour_to: int = 17
          ) -> RaceWeather:
    """Forecast over the race window. Returns `available=False` rather than raising —
    a missing forecast should degrade the page, not break the build."""
    coords = CIRCUIT_COORDS.get(event)
    if coords is None:
        return RaceWeather(event, date, False, note="no coordinates for this circuit")

    lat, lon = coords
    try:
        r = requests.get(ENDPOINT, params={
            "latitude": lat, "longitude": lon, "start_date": date, "end_date": date,
            "hourly": "precipitation_probability,precipitation,temperature_2m,wind_speed_10m",
            "timezone": "UTC",
        }, timeout=20)
        r.raise_for_status()
        h = r.json()["hourly"]
    except (requests.RequestException, KeyError, ValueError) as exc:
        return RaceWeather(event, date, False,
                           note=f"forecast unavailable ({type(exc).__name__})")

    sl = slice(hour_from, hour_to + 1)
    def _max(key):
        vals = [v for v in (h.get(key) or [])[sl] if v is not None]
        return max(vals) if vals else None

    prob = _max("precipitation_probability")
    return RaceWeather(
        event=event, date=date, available=True,
        rain_probability=None if prob is None else round(prob / 100.0, 3),
        precipitation_mm=_max("precipitation"),
        temperature_c=_max("temperature_2m"),
        wind_kph=_max("wind_speed_10m"),
    )


def uncertainty_multiplier(w: RaceWeather) -> float:
    """How much to widen the forecast for rain risk.

    Deliberately crude and deliberately conservative. With one wet race in the current
    regulations there is no basis for a fitted wet regime, so this is a stated assumption
    rather than an estimate: a realistic chance of rain raises the entropy of the
    finishing order, and pretending otherwise would be the more confident error.
    """
    p = w.rain_probability or 0.0
    if not w.available or p < 0.15:
        return 1.0
    return round(1.0 + 0.8 * min(p, 1.0), 3)


def to_dict(w: RaceWeather) -> dict:
    d = asdict(w)
    d["uncertainty_multiplier"] = uncertainty_multiplier(w)
    d["wet_risk"] = w.is_wet_risk
    return d
