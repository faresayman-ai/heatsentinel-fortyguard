"""
HeatSentinel — Hugging Face Spaces deployment
Wraps the UnifiedOrchestrator from the notebook with a Gradio chat UI.
Secrets come from HF Space secrets (set in Settings → Variables and secrets):
  FORTYGUARD_API_KEY, GROQ_API_KEY, GITHUB_TOKEN
"""
import os
import sys
import subprocess

def ensure_requirements():
    packages = [
        "gradio>=4.44.0",
        "huggingface_hub<0.26.0",
        "langchain-core>=0.3.0",
        "langchain-groq>=0.2.0",
        "pydantic>=2.7.0",
        "shapely>=2.0.0",
        "requests>=2.31.0",
        "starlette<1.0.0",
    ]
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", *packages],
        capture_output=True, text=True
    )
    print(result.stdout[-3000:], flush=True)
    if result.returncode != 0:
        print(result.stderr[-3000:], flush=True)
        raise RuntimeError("Failed to install dependencies at runtime")

ensure_requirements()

import spaces

@spaces.GPU
def _gpu_warmup():
    return True
# ── 1. Install the private FortyGuard library from GitHub ─────────────────────
def install_fortyguard():
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN secret is missing. "
            "Add it in your HF Space → Settings → Variables and secrets."
        )
    repo_url = f"https://oauth2:{token}@github.com/faresayman-ai/heatsentinel-fortyguard.git"
    clone_dir = "/tmp/heatsentinel-fortyguard"
    if not os.path.exists(clone_dir):
        result = subprocess.run(
            ["git", "clone", repo_url, clone_dir],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to clone heatsentinel-fortyguard repo.\n"
                f"stderr: {result.stderr}\n"
                "Check that GITHUB_TOKEN has read access to the repo."
            )
    if clone_dir not in sys.path:
        sys.path.insert(0, clone_dir)

install_fortyguard()

# ── 2. Set environment variables from HF secrets ──────────────────────────────
os.environ.setdefault("FORTYGUARD_BASE_URL", "https://api.fortyguard.com")
# FORTYGUARD_API_KEY and GROQ_API_KEY are already in env from HF secrets.
# Validate them early so the error is obvious at startup.
for key in ("FORTYGUARD_API_KEY", "GROQ_API_KEY"):
    if not os.environ.get(key):
        raise RuntimeError(
            f"{key} secret is missing. "
            "Add it in your HF Space → Settings → Variables and secrets."
        )

# ── 3. Import everything exactly as in the notebook ───────────────────────────
import threading
import time
import math
import json
import inspect
import requests
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from shapely.strtree import STRtree
from shapely.geometry import Point

from fortyguard import FortyGuardClient
from fortyguard.exceptions import FortyGuardError

# ── Thread-local FortyGuard client (matches notebook) ─────────────────────────
_thread_local = threading.local()

def get_client():
    if not hasattr(_thread_local, "client"):
        _thread_local.client = FortyGuardClient()
    return _thread_local.client

# ── Retry helpers ──────────────────────────────────────────────────────────────
def call_with_retry(func, *args, retries=3, **kwargs):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except (requests.exceptions.RequestException, TimeoutError) as e:
            last_exc = e
            wait = min(2 ** attempt, 5)
            time.sleep(wait)
        except FortyGuardError as e:
            msg = str(e)
            if any(code in msg for code in (" 429", " 500", " 502", " 503", " 504")):
                last_exc = e
                wait = min(2 ** attempt, 5)
                time.sleep(wait)
            else:
                raise
    raise last_exc

def call_llm_with_retry(func, *args, retries=2, **kwargs):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            wait = min(2 ** attempt, 4)
            time.sleep(wait)
    raise last_exc

# ── Temperature annotation ─────────────────────────────────────────────────────
_TEMP_LIKE_KEY_SUBSTR = ("temp", "heat_index", "apparent", "wet_bulb", "feels_like")

def annotate_temperature_units(data, source_unit: str = "celsius"):
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            out[k] = annotate_temperature_units(v, source_unit)
            key_lower = k.lower()
            if (
                isinstance(v, (int, float))
                and any(sub in key_lower for sub in _TEMP_LIKE_KEY_SUBSTR)
                and not key_lower.endswith(("_f", "_fahrenheit"))
            ):
                f_val = v * 9 / 5 + 32 if source_unit == "celsius" else v
                out[f"{k}_f"] = round(f_val, 1)
        return out
    elif isinstance(data, list):
        return [annotate_temperature_units(item, source_unit) for item in data]
    else:
        return data

# ── QueryInfo model ────────────────────────────────────────────────────────────
class QueryInfo(BaseModel):
    city: Optional[str] = Field(default=None, description="The primary city or location to analyze.")
    city_b: Optional[str] = Field(default=None, description="A second city to compare against.")
    start_date: Optional[str] = Field(default=None, description="Start date YYYY-MM-DD.")
    end_date: Optional[str] = Field(default=None, description="End date YYYY-MM-DD.")
    start_time: Optional[str] = Field(default=None, description="Start time HH:MM 24h.")
    end_time: Optional[str] = Field(default=None, description="End time HH:MM 24h.")
    filter_type: Optional[int] = Field(default=1, description="1=single hour, 2=hour range, 3=single day, 4=day range.")
    granularity: Optional[int] = Field(default=100, description="Tile resolution in meters.")
    latitude: Optional[float] = Field(default=None)
    longitude: Optional[float] = Field(default=None)
    full_city_boundary: Optional[bool] = Field(default=False)
    temperature: Optional[float] = Field(default=None, description="Ambient air temp °C.")
    hours_above_threshold: Optional[float] = Field(default=None)

# ── LLM setup ─────────────────────────────────────────────────────────────────
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1)
structured_llm = llm.with_structured_output(QueryInfo, method="json_mode")

FEW_SHOT_EXAMPLES = """
## Extraction Rules (read before examples)
- filter_type 1 = current snapshot (no date/time range)
- filter_type 2 = intra-day hour range (same calendar date, explicit start + end hour)
- filter_type 3 = single full-day aggregate (a whole named day, no specific hour)
- filter_type 4 = multi-day range (spans >1 calendar date, may involve 2 cities)
- full_city_boundary = true only when the user explicitly wants area-wide / citywide data
  (keywords: "whole city", "heat map", "city-wide", "across the city", "district-level")
- city_b is populated only when two cities are explicitly named for comparison
- Relative references ("now", "today", "yesterday", "last Tuesday") are resolved
  against today_date (UTC). Never invent or assume a date.

## Examples

### 1 — Live snapshot, single city
User: "Is it safe to run outside in Houston right now?"
Output: {"city": "Houston", "filter_type": 1, "full_city_boundary": false}

### 2 — Intra-day hour range
User: "What were temperatures like in San Antonio between 11am and 3pm today?"
Output: {"city": "San Antonio", "filter_type": 2, "start_time": "11:00", "end_time": "15:00", "full_city_boundary": false}

### 3 — Single-day aggregate
User: "How bad was the heat in Las Vegas last Friday?"
Output: {"city": "Las Vegas", "filter_type": 3, "full_city_boundary": false}

### 4 — Multi-day range
User: "Show me heat trends in Atlanta over the past 5 days"
Output: {"city": "Atlanta", "filter_type": 4, "full_city_boundary": false}

### 5 — Two-city comparison
User: "Compare heat exposure in Phoenix and Tucson this week"
Output: {"city": "Phoenix", "city_b": "Tucson", "filter_type": 4, "full_city_boundary": false}

### 6 — Citywide heat map
User: "I need a district-level heat breakdown of Dallas for yesterday"
Output: {"city": "Dallas", "filter_type": 3, "full_city_boundary": true}

### 7 — Coordinates
User: "What's the heat index at 29.7604° N, 95.3698° W right now?"
Output: {"latitude": 29.7604, "longitude": -95.3698, "filter_type": 1, "full_city_boundary": false}
"""

prompt = PromptTemplate(
    template=(
        "You are a parameter extractor. Extract heatmap query parameters "
        "from the user input and respond with a single valid JSON object "
        "matching the required schema — no prose, no markdown, JSON only.\n"
        "Today's date is {today_date} (UTC). Resolve any relative date or time "
        "reference against this date.\n"
        "{few_shot}\n"
        "User input: {user_input}"
    ),
    input_variables=["user_input"],
    partial_variables={
        "few_shot": FEW_SHOT_EXAMPLES,
        "today_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    },
)
chain = prompt | structured_llm

location_retry_prompt = PromptTemplate(
    template=(
        "Extract ONLY the location from the user input below — either a city "
        "name, or explicit latitude/longitude if given. Ignore dates, times, "
        "and everything else. Leave all other fields null. Respond with a "
        "single valid JSON object — no prose, no markdown, JSON only.\n"
        "User input: {user_input}"
    ),
    input_variables=["user_input"],
)
location_retry_chain = location_retry_prompt | structured_llm


def _fill_defaults(params):
    now = datetime.now(timezone.utc)
    if not params.filter_type:
        params.filter_type = 1
    if not params.start_date:
        params.start_date = now.strftime("%Y-%m-%d")
    if params.filter_type in (1, 2) and not params.start_time:
        params.start_time = now.strftime("%H:00")
    if params.filter_type == 2 and not params.end_time:
        params.end_time = f"{(now.hour + 1) % 24:02d}:00"
    if params.filter_type == 4 and not params.end_date:
        params.end_date = params.start_date
    return params


def _validate(params):
    if params.filter_type == 2 and params.start_time and not params.end_time:
        params.filter_type = 1
    if params.filter_type == 4 and params.start_date and params.end_date:
        if params.end_date < params.start_date:
            params.start_date, params.end_date = params.end_date, params.start_date
    if params.filter_type == 4 and params.start_date and params.end_date:
        span = (datetime.fromisoformat(params.end_date) - datetime.fromisoformat(params.start_date)).days
        if span > 31:
            from datetime import timedelta
            params.end_date = (datetime.fromisoformat(params.start_date) + timedelta(days=31)).strftime("%Y-%m-%d")
    if params.city_b and not params.city:
        params.city_b = None
    return params


def extract_params(user_input: str) -> QueryInfo:
    try:
        params = call_llm_with_retry(chain.invoke, {"user_input": user_input})
        if not isinstance(params, QueryInfo):
            params = QueryInfo(**params)
    except Exception:
        params = QueryInfo()

    if not params.city and not (params.latitude and params.longitude):
        try:
            retry_result = call_llm_with_retry(location_retry_chain.invoke, {"user_input": user_input})
            if not isinstance(retry_result, QueryInfo):
                retry_result = QueryInfo(**retry_result)
            if retry_result.city or (retry_result.latitude and retry_result.longitude):
                params.city = retry_result.city or params.city
                params.latitude = retry_result.latitude or params.latitude
                params.longitude = retry_result.longitude or params.longitude
        except Exception:
            pass

    params = _fill_defaults(params)
    params = _validate(params)
    return params

# ── Geocoding ──────────────────────────────────────────────────────────────────
class OutsideUSError(Exception):
    def __init__(self, city_name):
        self.city_name = city_name
        super().__init__(f"'{city_name}' was not found in the United States.")

_GEOCODE_CACHE = {}
_GEOCODE_LOCK = threading.Lock()

def _bbox_polygon(lat, lon, km=5.0):
    lat_offset = km / 111.0
    lon_offset = km / (111.0 * max(math.cos(math.radians(lat)), 0.01))
    coords = [[[lon - lon_offset, lat - lat_offset],
               [lon + lon_offset, lat - lat_offset],
               [lon + lon_offset, lat + lat_offset],
               [lon - lon_offset, lat + lat_offset],
               [lon - lon_offset, lat - lat_offset]]]
    return {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
            "geometry": {"type": "Polygon", "coordinates": coords}}]}

def _geocode_city(city_name, need_geojson):
    cached = _GEOCODE_CACHE.get(city_name)
    if cached and (not need_geojson or cached.get("geojson")):
        return cached
    with _GEOCODE_LOCK:
        cached = _GEOCODE_CACHE.get(city_name)
        if cached and (not need_geojson or cached.get("geojson")):
            return cached
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": city_name, "format": "json",
                  "polygon_geojson": 1 if need_geojson else 0,
                  "limit": 1, "countrycodes": "us"}
        headers = {"User-Agent": "heatsentinel-fortyguard"}
        resp = requests.get(url, params=params, headers=headers).json()
        if not resp:
            raise OutsideUSError(city_name)
        entry = {"lat": float(resp[0]["lat"]), "lon": float(resp[0]["lon"]),
                 "geojson": resp[0].get("geojson")}
        _GEOCODE_CACHE[city_name] = entry
        return entry

def resolve_point(params):
    if params.city:
        entry = _geocode_city(params.city, need_geojson=False)
        params.latitude = entry["lat"]
        params.longitude = entry["lon"]
        return params
    if params.latitude and params.longitude:
        return params
    raise ValueError("Could not determine latitude and longitude from your input.")

def get_city_polygon(city_name, full_boundary=False, bbox_km=5):
    entry = _geocode_city(city_name, need_geojson=full_boundary)
    if not full_boundary:
        return _bbox_polygon(entry["lat"], entry["lon"], km=bbox_km)
    geojson = entry["geojson"]
    if geojson["type"] == "Polygon":
        coords = geojson["coordinates"][0]
    elif geojson["type"] == "MultiPolygon":
        coords = max(geojson["coordinates"], key=lambda p: len(p[0]))[0]
    else:
        raise ValueError(f"Unexpected geometry type: {geojson['type']}")
    return {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [coords]}}]}

def get_polygon(params):
    if params.city:
        entry = _geocode_city(params.city, need_geojson=bool(params.full_city_boundary))
        params.latitude = entry["lat"]
        params.longitude = entry["lon"]
        return get_city_polygon(params.city, full_boundary=bool(params.full_city_boundary), bbox_km=5)
    elif params.latitude and params.longitude:
        return _bbox_polygon(params.latitude, params.longitude, km=1.0)
    else:
        raise ValueError("Please provide either a city name or latitude/longitude.")

# ── Heat risk classification ───────────────────────────────────────────────────
def classify_heat_risk(peak_c, heat_index_c=None):
    reference = heat_index_c if heat_index_c is not None else peak_c
    if reference < 27:
        tier, color, advice = "Safe", "green", "Normal activity is fine."
    elif reference < 32:
        tier, color, advice = "Caution", "yellow", (
            "Fatigue possible with prolonged outdoor exposure. Stay hydrated and take breaks in the shade.")
    elif reference < 39:
        tier, color, advice = "Extreme Caution", "orange", (
            "Heat cramps and heat exhaustion are possible. Limit strenuous outdoor activity.")
    elif reference < 46:
        tier, color, advice = "Danger", "red", (
            "Heat cramps and exhaustion are likely. Heat stroke is possible. Suspend non-essential outdoor work.")
    else:
        tier, color, advice = "Extreme Danger", "darkred", (
            "Heat stroke is imminent without immediate cooling. All outdoor activity should stop.")
    return {"tier": tier, "color": color, "advice": advice,
            "reference_temp_c": round(reference, 1), "using_heat_index": heat_index_c is not None}

def extract_safe_windows(env_result):
    try:
        locations = env_result.get("locations", [])
        if not locations:
            return {}
        params = locations[0].get("parameters", {})
        hi_series = params.get("heat_index_celsius", [])
        if not hi_series:
            return {}
        safe_hours = [h for h, val in enumerate(hi_series) if val is not None and val < 32]
        if not safe_hours:
            summary = "No safe outdoor window today — heat index exceeds Caution level all day."
        else:
            ranges = []
            start = safe_hours[0]
            prev = safe_hours[0]
            for h in safe_hours[1:]:
                if h != prev + 1:
                    ranges.append((start, prev))
                    start = h
                prev = h
            ranges.append((start, prev))
            def fmt(h):
                suffix = "AM" if h < 12 else "PM"
                display = h if h <= 12 else h - 12
                display = 12 if display == 0 else display
                return f"{display}:00 {suffix}"
            parts = [f"{fmt(s)}–{fmt(e+1)}" for s, e in ranges]
            summary = "Safe outdoor windows: " + ", ".join(parts)
        return {"safe_hours_count": len(safe_hours), "safe_windows_summary": summary,
                "peak_heat_index_c": round(max(v for v in hi_series if v is not None), 1)}
    except Exception:
        return {}

# ── Tile index ─────────────────────────────────────────────────────────────────
_tile_polys = None
_tile_index = None
_tile_lock = threading.Lock()

def _build_tile_index(tiles_list):
    global _tile_polys, _tile_index
    _tile_polys = [t[0] for t in tiles_list]
    _tile_index = STRtree(_tile_polys)

def tile_for(tiles_list, lat, lon):
    global _tile_polys, _tile_index
    if _tile_index is None or _tile_polys is None:
        with _tile_lock:
            if _tile_index is None or _tile_polys is None:
                _build_tile_index(tiles_list)
    p = Point(lon, lat)
    candidates = list(_tile_index.query(p))
    for idx in candidates:
        if _tile_polys[idx].contains(p):
            return tiles_list[idx]
    pool = [tiles_list[i] for i in candidates] if candidates else tiles_list
    return min(pool, key=lambda t: t[0].centroid.distance(p))

def _filtered_kwargs(func, **kwargs):
    accepted = set(inspect.signature(func).parameters)
    return {k: v for k, v in kwargs.items() if k in accepted}

# ── Tool functions ─────────────────────────────────────────────────────────────
def temp_stats(user_input_or_params):
    params = user_input_or_params if isinstance(user_input_or_params, QueryInfo) else extract_params(user_input_or_params)
    if not params.city and (not params.latitude or not params.longitude):
        raise ValueError("temp_stats: no location — need a city or latitude/longitude.")
    if params.granularity is None:
        params.granularity = 100
    exceedance_stats = None
    if params.hours_above_threshold is not None and params.filter_type in (2, 4):
        try:
            exc_kwargs = _filtered_kwargs(
                get_client().create_heatmap,
                polygon_aoi=get_polygon(params),
                start_date=params.start_date, start_time=params.start_time,
                end_date=params.end_date, end_time=params.end_time,
                filter_type=params.filter_type, granularity=params.granularity,
                analytic_type="exceedance", threshold=params.hours_above_threshold,
                direction="above", verbose=False, timeout=180, poll_interval=3,
            )
            exc_response = call_with_retry(get_client().create_heatmap, **exc_kwargs)
            exceedance_stats = exc_response['result'].get('stats_data', {})
        except Exception:
            pass

    _EARLIEST_DATE = datetime(2021, 1, 1).date()
    if params.start_date:
        start = datetime.strptime(params.start_date, "%Y-%m-%d").date()
        if start < _EARLIEST_DATE:
            raise ValueError(f"FortyGuard's catalog starts 2021-01-01 — '{params.start_date}' has no data.")
        if start > datetime.now().date():
            raise ValueError(f"'{params.start_date}' is in the future — no data for it yet.")

    polygon = get_polygon(params)
    kwargs = _filtered_kwargs(
        get_client().create_heatmap,
        polygon_aoi=polygon, start_date=params.start_date, start_time=params.start_time,
        end_date=params.end_date, end_time=params.end_time, filter_type=params.filter_type,
        granularity=params.granularity, verbose=False, timeout=180, poll_interval=3,
    )
    response = call_with_retry(get_client().create_heatmap, **kwargs)
    result = response['result']
    stats = result.get('stats_data', {})
    raw_stats = None
    for key, value in stats.items():
        normalized = key.lower().replace(" ", "_")
        if "temp" in normalized and "stat" in normalized:
            raw_stats = value
            break
    if raw_stats is None:
        raw_stats = {}
    result = annotate_temperature_units(raw_stats, source_unit="celsius")
    if exceedance_stats is not None:
        result["hours_above_threshold_c"] = params.hours_above_threshold
        result["hours_above_threshold_stats"] = exceedance_stats
    return result


def environmental_parameters(user_input_or_params, peak_temp=None):
    params = user_input_or_params if isinstance(user_input_or_params, QueryInfo) else extract_params(user_input_or_params)
    resolve_point(params)
    if peak_temp is None and params.temperature is None:
        raise ValueError("environmental_parameters: temperature is required.")
    kwargs = _filtered_kwargs(
        get_client().environmental_parameters,
        latitude=params.latitude, longitude=params.longitude,
        temperature=float(peak_temp or params.temperature),
        start_date=params.start_date, start_time=params.start_time,
        end_date=params.end_date, end_time=params.end_time,
        filter_type=params.filter_type, verbose=False, timeout=60,
    )
    response = call_with_retry(get_client().environmental_parameters, **kwargs)
    return response['result']


def satellite_segmentation(user_input_or_params):
    params = user_input_or_params if isinstance(user_input_or_params, QueryInfo) else extract_params(user_input_or_params)
    if params.granularity is None:
        params.granularity = 100
    polygon = get_polygon(params)
    kwargs = _filtered_kwargs(
        get_client().satellite_segmentation,
        polygon_aoi=polygon, latitude=params.latitude, longitude=params.longitude,
        start_date=params.start_date, start_time=params.start_time,
        filter_type=params.filter_type, granularity=params.granularity,
        verbose=False, timeout=180,
    )
    response = call_with_retry(get_client().satellite_segmentation, **kwargs)
    return response.get('result', response)


def street_view_segmentation(user_input_or_params):
    params = user_input_or_params if isinstance(user_input_or_params, QueryInfo) else extract_params(user_input_or_params)
    if params.granularity is None:
        params.granularity = 100
    polygon = get_polygon(params)
    kwargs = _filtered_kwargs(
        get_client().street_view_segmentation,
        polygon_aoi=polygon, latitude=params.latitude, longitude=params.longitude,
        start_date=params.start_date, start_time=params.start_time,
        filter_type=params.filter_type, granularity=params.granularity,
        verbose=False, timeout=60,
    )
    response = call_with_retry(get_client().street_view_segmentation, **kwargs)
    return response.get('result', response)


def weather_forecast(user_input_or_params):
    params = user_input_or_params if isinstance(user_input_or_params, QueryInfo) else extract_params(user_input_or_params)
    resolve_point(params)
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": params.latitude, "longitude": params.longitude,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode",
                "temperature_unit": "celsius", "forecast_days": 7, "timezone": "auto"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    daily = data.get("daily") or {}
    return {"source": "Open-Meteo (independent forecast — not FortyGuard, not a live measurement)",
            "dates": daily.get("time", []), "max_temp_c": daily.get("temperature_2m_max", []),
            "min_temp_c": daily.get("temperature_2m_min", []),
            "precipitation_probability_pct": daily.get("precipitation_probability_max", []),
            "weathercode": daily.get("weathercode", [])}

# ── Router ─────────────────────────────────────────────────────────────────────
VALID_FUNCTIONS = frozenset({
    "temp_stats", "environmental_parameters", "satellite_segmentation",
    "street_view_segmentation", "heat_briefing", "weather_forecast",
})

class RouterInfo(BaseModel):
    functions: list[str] = Field(description="Ordered list of functions to call.")
    reasoning: str = Field(description="One sentence explaining function selection.")

    @field_validator("functions")
    @classmethod
    def validate_functions(cls, v):
        invalid = [f for f in v if f not in VALID_FUNCTIONS]
        if invalid:
            raise ValueError(f"Unknown function(s): {invalid}. Must be subset of: {sorted(VALID_FUNCTIONS)}")
        return v

router_parser = JsonOutputParser(pydantic_object=RouterInfo)

ROUTER_FEW_SHOT = """
## Routing Rules
- Return functions in execution order (dependencies first).
- 'environmental_parameters' always needs 'temp_stats' first.
- 'heat_briefing' is self-contained — never pair with other functions.
- 'satellite_segmentation' = aerial/overhead. 'street_view_segmentation' = ground level.
- 'weather_forecast' = future conditions only.

## Examples
"how hot is it in Austin right now" → {"functions": ["temp_stats"], "reasoning": "Current temperature reading."}
"heat index and humidity in Phoenix" → {"functions": ["temp_stats", "environmental_parameters"], "reasoning": "Heat index needs env_params which needs temp first."}
"why is downtown Dallas hotter" → {"functions": ["satellite_segmentation"], "reasoning": "Surface composition question."}
"full heat risk briefing for Houston" → {"functions": ["heat_briefing"], "reasoning": "Full analysis requested."}
"what will weather be like next week" → {"functions": ["weather_forecast"], "reasoning": "Future conditions."}
"""

router_prompt = PromptTemplate(
    template=(
        "You are a routing agent for HeatSentinel. Select function(s) to call.\n\n"
        "{format_instructions}\n\n"
        "{few_shot}\n"
        "Recent conversation (last 2 turns):\n{history_text}\n\n"
        "User query: {user_input}"
    ),
    input_variables=["user_input", "history_text"],
    partial_variables={"format_instructions": router_parser.get_format_instructions(),
                       "few_shot": ROUTER_FEW_SHOT},
)
router_chain = router_prompt | llm | router_parser

def heat_briefing(user_input_or_params):
    params = user_input_or_params if isinstance(user_input_or_params, QueryInfo) else extract_params(user_input_or_params)
    results = {}
    try:
        results["temp_stats"] = temp_stats(params)
    except Exception as e:
        results["temp_stats"] = {"error": str(e)}
    peak_temp = None
    try:
        ts = results["temp_stats"]
        peak_temp = ts.get("max") or ts.get("Max") or ts.get("maximum") or ts.get("Maximum")
    except Exception:
        pass
    for fn, fn_func in [("environmental_parameters", lambda: environmental_parameters(params, peak_temp=peak_temp)),
                         ("satellite_segmentation", lambda: satellite_segmentation(params)),
                         ("street_view_segmentation", lambda: street_view_segmentation(params)),
                         ("weather_forecast", lambda: weather_forecast(params))]:
        try:
            results[fn] = fn_func()
        except Exception as e:
            results[fn] = {"error": str(e)}
    return results

FUNCTION_MAP = {
    "temp_stats": temp_stats,
    "environmental_parameters": environmental_parameters,
    "satellite_segmentation": satellite_segmentation,
    "street_view_segmentation": street_view_segmentation,
    "weather_forecast": weather_forecast,
    "heat_briefing": heat_briefing,
}

def route(user_input, history_text=""):
    result = router_chain.invoke({"user_input": user_input, "history_text": history_text})
    result = RouterInfo(**result)
    if "heat_briefing" in result.functions and len(result.functions) > 1:
        result.functions = ["heat_briefing"]
    return result

# ── Conversation memory ────────────────────────────────────────────────────────
class ConversationMemory:
    def __init__(self, max_turns=10):
        self.messages = []
        self.max_turns = max_turns

    def add_user(self, text):
        self.messages.append(HumanMessage(content=text))
        self._trim()

    def add_ai(self, text):
        self.messages.append(AIMessage(content=text))
        self._trim()

    def _trim(self):
        limit = self.max_turns * 2
        if len(self.messages) > limit:
            self.messages = self.messages[-limit:]

    def as_list(self):
        return self.messages

    def clear(self):
        self.messages = []

# ── Explainer LLM ──────────────────────────────────────────────────────────────
explainer_llm = llm.bind(temperature=0.4)

EXPLAINER_SYSTEM = """You are HeatSentinel, an urban heat-risk assistant powered by FortyGuard sensor data.
Your job is to translate raw JSON heat readings into clear, actionable guidance for non-technical users.

LOCATION ANCHOR: The 'location' field in the current turn's JSON is authoritative. Use it verbatim.
TIME PERIOD: The JSON's 'reporting_period' field is the authoritative window.
DATA AVAILABILITY: If neither 'temp_stats' nor 'heat_risk' is present, no temperature data was requested.
  Skip section ① and report what IS present (satellite, street_view, forecast).
ERROR HANDLING:
  {{"error": "outside_us"}} → "HeatSentinel currently covers US cities only."
  {{"error": "location_not_found"}} → Ask for a US city name or coordinates.
  {{"error": "processing_timeout"}} → Suggest retrying in ~1 minute.

OUTPUT STRUCTURE (in order, skip ① only when no temp data):
① RISK VERDICT (1 sentence) — state heat-risk tier and plain-English headline
② KEY NUMBERS (2–4 bullets) — peak temperature, heat index, etc.
③ WHAT IT FEELS LIKE (1 short paragraph) — vivid physical description
④ SAFE WINDOWS — state safe hours or say there are none
⑤ RECOMMENDED ACTIONS (2–3 specific items)

TEMPERATURE UNITS:
• Fields ending in '_f' are Fahrenheit — display as °F
• All other temperature fields are Celsius — display as °C
• Never perform your own unit conversions

TONE: Conversational and direct — like a knowledgeable local friend."""

explainer_prompt = ChatPromptTemplate.from_messages([
    ("system", EXPLAINER_SYSTEM),
    MessagesPlaceholder("history"),
    ("human", "User question: {user_input}\n\nFortyGuard live data (JSON):\n{fortyguard_json}\n\n"
               "Follow the ①–⑤ structure. If an error key is present, respond with only the error message."),
])
explainer_chain = explainer_prompt | explainer_llm

# ── UnifiedOrchestrator (identical to notebook) ────────────────────────────────
class UnifiedOrchestrator:
    MAX_JSON_LENGTH = 8000
    AUTO_ENV_THRESHOLD = 35.0
    AUTO_SAT_THRESHOLD = 37.0
    RESULT_CACHE_TTL = 600

    def __init__(self, verbose=False):
        self.memory = ConversationMemory()
        self.cached_params = None
        self.temp_cache = {}
        self.last_known_temp = None
        self.last_known_temp_loc = None
        self._result_cache = {}
        self.verbose = verbose

    def _loc_key(self, lat, lon):
        if lat is None or lon is None:
            return None
        return (round(float(lat), 4), round(float(lon), 4))

    def _cache_temperature(self, lat, lon, temp):
        if temp is None:
            return
        key = self._loc_key(lat, lon)
        temp = float(temp)
        if key is not None:
            self.temp_cache[key] = temp
        self.last_known_temp = temp
        self.last_known_temp_loc = key

    def _lookup_temperature(self, lat, lon):
        key = self._loc_key(lat, lon)
        if key is not None and key in self.temp_cache:
            return self.temp_cache[key]
        if key is not None and key == self.last_known_temp_loc:
            return self.last_known_temp
        return None

    def _temp_stats_cache_key(self, params):
        loc = (params.city or "").strip().lower() or self._loc_key(params.latitude, params.longitude)
        return (loc, params.start_date, params.start_time, params.end_date,
                params.end_time, params.filter_type, params.granularity)

    def _get_cached_result(self, cache_key):
        entry = self._result_cache.get(cache_key)
        if entry is None:
            return None
        ts, value = entry
        if time.monotonic() - ts > self.RESULT_CACHE_TTL:
            return None
        return value

    def _set_cached_result(self, cache_key, value):
        self._result_cache[cache_key] = (time.monotonic(), value)

    def _history_text(self):
        msgs = self.memory.as_list()[-4:]
        if not msgs:
            return "(none)"
        lines = []
        for m in msgs:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            lines.append(f"{role}: {m.content[:200]}")
        return "\n".join(lines)

    def _extract_peak(self, temp_stats_result):
        if not isinstance(temp_stats_result, dict):
            return None
        direct = (temp_stats_result.get("max") or temp_stats_result.get("Max")
                  or temp_stats_result.get("maximum") or temp_stats_result.get("Maximum"))
        if direct is not None:
            return direct
        for key, value in temp_stats_result.items():
            if isinstance(value, (int, float)):
                k = key.lower()
                if "max" in k and "temp" in k:
                    return value
        return None

    def _inject_risk_tier(self, results, temp_stats_key="temp_stats", env_key="environmental_parameters"):
        ts = results.get(temp_stats_key, {})
        peak = self._extract_peak(ts)
        if peak is None:
            return
        hi = None
        try:
            env = results.get(env_key, {})
            locs = env.get("locations", [])
            if locs:
                hi_series = locs[0].get("parameters", {}).get("heat_index_celsius", [])
                if hi_series:
                    hi = max(v for v in hi_series if v is not None)
        except Exception:
            pass
        results["heat_risk"] = classify_heat_risk(float(peak), hi)

    def _annotate_env_params(self, result):
        if not isinstance(result, dict):
            return
        result["_note_heat_index_artifact"] = (
            "heat_index_celsius tracks overnight humidity, not real diurnal cycle — "
            "use apparent_temperature_celsius peak hour instead.")
        result["_note_grid_resolution"] = (
            "This endpoint resolves on a weather grid coarser than typical parcel spacing.")

    def _run_single_tool(self, fn_name, current_params, cached_temp_stats, results):
        if fn_name not in FUNCTION_MAP:
            return cached_temp_stats
        try:
            if fn_name == "temp_stats":
                try:
                    cache_key = self._temp_stats_cache_key(current_params)
                    cached_hit = self._get_cached_result(cache_key)
                    if cached_hit is not None:
                        cached_temp_stats = cached_hit
                    else:
                        cached_temp_stats = temp_stats(current_params)
                        self._set_cached_result(cache_key, cached_temp_stats)
                    results["temp_stats"] = cached_temp_stats
                    peak = self._extract_peak(cached_temp_stats)
                    self._cache_temperature(current_params.latitude, current_params.longitude, peak)
                except ValueError:
                    results["temp_stats"] = {"error": "location_not_found"}
                    return cached_temp_stats

            elif fn_name == "environmental_parameters":
                try:
                    resolve_point(current_params)
                except ValueError:
                    results["environmental_parameters"] = {"error": "location_not_found"}
                    return cached_temp_stats
                temperature = current_params.temperature
                if temperature is None and cached_temp_stats:
                    temperature = self._extract_peak(cached_temp_stats)
                if temperature is None:
                    temperature = self._lookup_temperature(current_params.latitude, current_params.longitude)
                if temperature is None:
                    return cached_temp_stats
                kwargs = _filtered_kwargs(
                    get_client().environmental_parameters,
                    latitude=current_params.latitude, longitude=current_params.longitude,
                    temperature=float(temperature),
                    start_date=current_params.start_date, start_time=current_params.start_time,
                    end_date=current_params.end_date, end_time=current_params.end_time,
                    filter_type=current_params.filter_type, verbose=False, timeout=60,
                )
                response = call_with_retry(get_client().environmental_parameters, **kwargs)
                env_result = annotate_temperature_units(response["result"], source_unit="celsius")
                self._annotate_env_params(env_result)
                results["environmental_parameters"] = env_result
                self._cache_temperature(current_params.latitude, current_params.longitude, temperature)
                safe = extract_safe_windows(response["result"])
                if safe:
                    results["safe_windows"] = safe

            elif fn_name == "satellite_segmentation":
                try:
                    resolve_point(current_params)
                except ValueError:
                    results["satellite_segmentation"] = {"error": "location_not_found"}
                    return cached_temp_stats
                results["satellite_segmentation"] = satellite_segmentation(current_params)

            elif fn_name == "street_view_segmentation":
                try:
                    resolve_point(current_params)
                except ValueError:
                    results["street_view_segmentation"] = {"error": "location_not_found"}
                    return cached_temp_stats
                results["street_view_segmentation"] = street_view_segmentation(current_params)

            elif fn_name == "weather_forecast":
                try:
                    resolve_point(current_params)
                except ValueError:
                    results["weather_forecast"] = {"error": "location_not_found"}
                    return cached_temp_stats
                results["weather_forecast"] = weather_forecast(current_params)

        except OutsideUSError as e:
            results[fn_name] = {"error": "outside_us", "city": e.city_name}
        except Exception as e:
            msg = str(e)
            if "processing" in msg.lower() and "still" in msg.lower():
                results[fn_name] = {"error": "processing_timeout"}
            else:
                results[fn_name] = {"error": msg}
        return cached_temp_stats

    def _run_comparison(self, current_params, results):
        city_a = current_params.city
        city_b = current_params.city_b
        params_a = current_params.model_copy()
        params_b = current_params.model_copy()
        params_b.city = city_b
        params_b.latitude = None
        params_b.longitude = None
        results_a, results_b = {}, {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            fa = executor.submit(self._run_single_tool, "temp_stats", params_a, None, results_a)
            fb = executor.submit(self._run_single_tool, "temp_stats", params_b, None, results_b)
            fa.result(); fb.result()
        temp_a, temp_b = results_a.get("temp_stats", {}), results_b.get("temp_stats", {})
        if isinstance(temp_a, dict) and temp_a.get("error"):
            results["comparison"] = temp_a; return
        if isinstance(temp_b, dict) and temp_b.get("error"):
            results["comparison"] = temp_b; return
        peak_a = self._extract_peak(temp_a)
        risk_a = classify_heat_risk(float(peak_a)) if peak_a is not None else {}
        peak_b = self._extract_peak(temp_b)
        risk_b = classify_heat_risk(float(peak_b)) if peak_b is not None else {}
        hotter = None; delta = None
        if peak_a is not None and peak_b is not None:
            delta = round(abs(float(peak_a) - float(peak_b)), 1)
            hotter = city_a if float(peak_a) >= float(peak_b) else city_b
        results["comparison"] = {city_a: {"temp_stats": temp_a, "heat_risk": risk_a},
                                  city_b: {"temp_stats": temp_b, "heat_risk": risk_b},
                                  "hotter_city": hotter, "delta_c": delta}
        if hotter == city_a and risk_a:
            results["heat_risk"] = risk_a
        elif hotter == city_b and risk_b:
            results["heat_risk"] = risk_b

    def _run_briefing(self, current_params, results):
        params_for_temp = current_params.model_copy(deep=True)
        params_for_sat  = current_params.model_copy(deep=True)
        params_for_sv   = current_params.model_copy(deep=True)
        params_for_wx   = current_params.model_copy(deep=True)
        results_temp, results_sat, results_sv, results_wx = {}, {}, {}, {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            tf = executor.submit(self._run_single_tool, "temp_stats", params_for_temp, None, results_temp)
            sf = executor.submit(self._run_single_tool, "satellite_segmentation", params_for_sat, None, results_sat)
            vf = executor.submit(self._run_single_tool, "street_view_segmentation", params_for_sv, None, results_sv)
            wf = executor.submit(self._run_single_tool, "weather_forecast", params_for_wx, None, results_wx)
            cached_temp_stats = tf.result(); sf.result(); vf.result(); wf.result()
        results.update(results_temp); results.update(results_sat)
        results.update(results_sv); results.update(results_wx)
        if params_for_temp.latitude and params_for_temp.longitude:
            current_params.latitude  = params_for_temp.latitude
            current_params.longitude = params_for_temp.longitude
        self._run_single_tool("environmental_parameters", current_params, cached_temp_stats, results)
        self._inject_risk_tier(results)

    def _run_tools(self, user_input):
        history_text = self._history_text()
        with ThreadPoolExecutor(max_workers=2) as executor:
            router_future  = executor.submit(call_llm_with_retry, router_chain.invoke,
                                             {"user_input": user_input, "history_text": history_text})
            params_future  = executor.submit(extract_params, user_input)
            try:
                decision = RouterInfo(**router_future.result())
            except Exception:
                params_future.result()
                return {}
            current_params = params_future.result()

        if "heat_briefing" in decision.functions and len(decision.functions) > 1:
            decision.functions = ["heat_briefing"]

        results = {}
        if not current_params.city and not (current_params.latitude and current_params.longitude):
            if self.cached_params and (self.cached_params.latitude or self.cached_params.longitude):
                current_params.latitude  = current_params.latitude  or self.cached_params.latitude
                current_params.longitude = current_params.longitude or self.cached_params.longitude
                current_params.start_date = current_params.start_date or self.cached_params.start_date
                current_params.start_time = current_params.start_time or self.cached_params.start_time
                current_params.city = current_params.city or self.cached_params.city
        self.cached_params = current_params

        results["location"] = (current_params.city or
            (f"{current_params.latitude:.4f}, {current_params.longitude:.4f}"
             if current_params.latitude and current_params.longitude else "unknown"))

        if current_params.city and current_params.city_b:
            try:
                resolve_point(current_params)
            except OutsideUSError as e:
                results["comparison"] = {"error": "outside_us", "city": e.city_name}
                return results
            except ValueError:
                results["comparison"] = {"error": "location_not_found"}
                return results
            self._run_comparison(current_params, results)
            return results

        if "heat_briefing" in decision.functions:
            try:
                resolve_point(current_params)
            except OutsideUSError as e:
                results["temp_stats"] = {"error": "outside_us", "city": e.city_name}
                return results
            except ValueError:
                results["temp_stats"] = {"error": "location_not_found"}
                return results
            self._run_briefing(current_params, results)
            return results

        if "environmental_parameters" in decision.functions and "temp_stats" not in decision.functions:
            decision.functions.insert(0, "temp_stats")

        cached_temp_stats = None
        for fn_name in decision.functions:
            try:
                cached_temp_stats = self._run_single_tool(fn_name, current_params, cached_temp_stats, results)
            except OutsideUSError as e:
                results[fn_name] = {"error": "outside_us", "city": e.city_name}
            except Exception as e:
                results[fn_name] = {"error": str(e)}

        need_env = (
            "temp_stats" in results and "environmental_parameters" not in results
            and self._extract_peak(results.get("temp_stats", {})) is not None
            and float(self._extract_peak(results.get("temp_stats", {}))) >= self.AUTO_ENV_THRESHOLD
            and current_params.filter_type in (1, 2, 3)
        )
        need_sat = (
            "temp_stats" in results and "satellite_segmentation" not in results
            and self._extract_peak(results.get("temp_stats", {})) is not None
            and float(self._extract_peak(results.get("temp_stats", {}))) >= self.AUTO_SAT_THRESHOLD
        )
        if need_env or need_sat:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                if need_env:
                    futures.append(executor.submit(self._run_single_tool, "environmental_parameters",
                                                   current_params, cached_temp_stats, results))
                if need_sat:
                    futures.append(executor.submit(self._run_single_tool, "satellite_segmentation",
                                                   current_params, cached_temp_stats, results))
                for f in futures:
                    f.result()

        if "temp_stats" in results:
            self._inject_risk_tier(results)
        return results

    _IMAGE_LIKE_KEY_SUBSTR = ("image", "img", "photo", "picture")

    @staticmethod
    def _strip_binary_fields(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if (isinstance(v, str) and len(v) > 200
                        and any(sub in k.lower() for sub in UnifiedOrchestrator._IMAGE_LIKE_KEY_SUBSTR)):
                    out[k] = f"<{len(v)}-char image payload omitted from LLM context>"
                else:
                    out[k] = UnifiedOrchestrator._strip_binary_fields(v)
            return out
        elif isinstance(obj, list):
            return [UnifiedOrchestrator._strip_binary_fields(item) for item in obj]
        else:
            return obj

    @staticmethod
    def _trim_result(obj, max_chars=8000):
        obj = UnifiedOrchestrator._strip_binary_fields(obj)
        s = json.dumps(obj, indent=2, default=str)
        if len(s) <= max_chars:
            return s
        trimmed = dict(obj)
        dropped = []
        protected = {"location", "heat_risk"}
        sized_keys = sorted((k for k in trimmed if k not in protected),
                            key=lambda k: len(json.dumps(trimmed[k], default=str)), reverse=True)
        for k in sized_keys:
            if len(json.dumps(trimmed, indent=2, default=str)) <= max_chars:
                break
            del trimmed[k]
            dropped.append(k)
        if dropped:
            trimmed["_truncated_fields"] = dropped
        s = json.dumps(trimmed, indent=2, default=str)
        if len(s) > max_chars:
            s = json.dumps({"location": obj.get("location", "unknown"),
                             "_truncated_fields": list(obj.keys())}, indent=2)
        return s

    def ask(self, user_input):
        try:
            fortyguard_results = self._run_tools(user_input)
            fortyguard_json = self._trim_result(fortyguard_results, self.MAX_JSON_LENGTH)
            response = call_llm_with_retry(
                explainer_chain.invoke,
                {"user_input": user_input, "fortyguard_json": fortyguard_json,
                 "history": self.memory.as_list()},
            )
            answer = response.content
        except Exception as e:
            answer = (
                "Sorry — I ran into a problem pulling that data and couldn't finish "
                "the analysis. Mind rephrasing, or trying again in a moment? "
                f"(Error: {e})"
            )
        self.memory.add_user(user_input)
        self.memory.add_ai(answer)
        return answer

# ── Gradio UI ──────────────────────────────────────────────────────────────────
import gradio as gr

import gradio_client.utils as _gc_utils

_original_json_schema_to_python_type = _gc_utils._json_schema_to_python_type

def _patched_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _original_json_schema_to_python_type(schema, defs)

_gc_utils._json_schema_to_python_type = _patched_json_schema_to_python_type

_original_get_type = _gc_utils.get_type

def _patched_get_type(schema):
    if isinstance(schema, bool):
        return "Any"
    return _original_get_type(schema)

_gc_utils.get_type = _patched_get_type

heat_theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="red",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_950",
    body_background_fill_dark="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_width="1px",
    block_border_color="*neutral_800",
    block_radius="16px",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_500",
    button_primary_text_color="white",
    input_background_fill="*neutral_800",
)

CUSTOM_CSS = """
#header-row {padding: 8px 4px 4px 4px;}
#header-row h1 {font-size: 1.85rem; margin-bottom: 0.15rem;}
#header-row p {opacity: 0.75; margin-top: 0;}
#chatbot {border-radius: 16px;}
#footer-note {opacity: 0.55; font-size: 0.8rem; text-align: center; margin-top: 6px;}
footer {display: none !important;}
"""

EXAMPLE_PROMPTS = [
    "How hot is it in Phoenix right now?",
    "Give me a full heat briefing for Houston today",
    "Compare heat in Dallas vs Miami",
    "What's the safest time to run outside in Austin today?",
]

def create_orchestrator():
    return UnifiedOrchestrator(verbose=False)

def user_submit(message, history):
    if not message.strip():
        return gr.update(), history
    return "", history + [(message, None)]

def bot_respond(history, orchestrator_state):
    if not history:
        return history, orchestrator_state
    message = history[-1][0]
    if orchestrator_state is None:
        orchestrator_state = create_orchestrator()
    answer = orchestrator_state.ask(message)
    history[-1] = (message, answer)
    return history, orchestrator_state

def clear_chat():
    return [], None

with gr.Blocks(title="HeatSentinel", theme=heat_theme, css=CUSTOM_CSS, fill_height=True) as demo:
    orchestrator_state = gr.State(None)

    with gr.Row(elem_id="header-row"):
        with gr.Column():
            gr.Markdown(
                "# 🌡️ HeatSentinel\n"
                "**Urban heat risk assistant**, powered by live FortyGuard sensor data — "
                "risk tiers, safe outdoor windows, and plain-language briefings for any US city."
            )

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        height=520,
        show_label=False,
        bubble_full_width=False,
        avatar_images=(None, "🌡️"),
        render_markdown=True,
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask about heat conditions in any US city…",
            show_label=False,
            lines=1,
            scale=8,
            container=False,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    gr.Examples(
        examples=EXAMPLE_PROMPTS,
        inputs=msg,
        label="Try asking",
    )

    clear_btn = gr.Button("🗑️  Clear conversation", variant="secondary", size="sm")

    gr.Markdown(
        "HeatSentinel currently covers US locations only. "
        "Data is provided for informational purposes and is not a substitute for official heat advisories.",
        elem_id="footer-note",
    )

    msg.submit(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_respond, [chatbot, orchestrator_state], [chatbot, orchestrator_state]
    )
    send_btn.click(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_respond, [chatbot, orchestrator_state], [chatbot, orchestrator_state]
    )

if __name__ == "__main__":
    demo.launch()
