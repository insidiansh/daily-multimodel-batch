"""
DAILY MULTI-MODEL HEALTH & INFERENCE BATCH

What this script does:
- Runs GPT-OSS-120B (text), Whisper (audio), DeepSeek-OCR (image)
- Uses RANDOM open-source input for each model every day
- Records which model WORKED and which FAILED
- Outputs ONE consolidated JSON report

Previous errors (now FIXED):
1. Whisper failures:
   - Cause: Audio URLs from www2.cs.uic.edu often fail DNS resolution in Colab/VMs
   - Fix: Switched to Wikimedia Commons audio (globally cached, stable)

2. DeepSeek OCR failures:
   - Cause: OCR backend could not fetch some image URLs (403 Forbidden)
   - Fix: Download image locally and send BASE64 instead of image_url

3. Partial pipeline failures:
   - Cause: One model failure stopped the flow
   - Fix: Each model runs in isolated try/except blocks

This version is production-safe.
"""

# =====================================================
# 📦 IMPORTS
# =====================================================
import os
import json
import datetime
import random
import requests
import base64
from openai import OpenAI

# =====================================================
# 🔑 CONFIG (API KEY VIA ENV VAR)
# =====================================================
SIMPLISMART_API_KEY = os.getenv("SIMPLISMART_API_KEY")
if not SIMPLISMART_API_KEY:
    raise RuntimeError("SIMPLISMART_API_KEY not set")

SIMPLISMART_BASE_URL = "https://api.simplismart.live"
WHISPER_URL = "https://http.whisper.proxy.prod.s9t.link/model/infer/whisper"

BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# 📅 METADATA
# =====================================================
today = datetime.date.today().isoformat()
run_time = datetime.datetime.now().strftime("%H:%M")

report = {
    "date": today,
    "run_time": run_time,
    "results": {}
}

# =====================================================
# 🧠 1. GPT-OSS-120B (TEXT MODEL)
# =====================================================
try:
    # Random public-domain text every day
    text_sources = [
        "https://www.gutenberg.org/files/84/84-0.txt",    # Frankenstein
        "https://www.gutenberg.org/files/1342/1342-0.txt",# Pride & Prejudice
        "https://www.gutenberg.org/files/11/11-0.txt"     # Alice in Wonderland
    ]
    text_url = random.choice(text_sources)
    text = requests.get(text_url, timeout=20).text[:2000]

    client = OpenAI(
        api_key=SIMPLISMART_API_KEY,
        base_url=SIMPLISMART_BASE_URL,
        default_headers={
            "id": "524436ef-5d4c-4d55-9351-71d67036b92b"  # GPT tenant ID
        }
    )

    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": f"Summarize this text:\n{text}"}],
        max_tokens=300,
        temperature=0
    )

    report["results"]["gpt_oss_120b"] = {
        "status": "success",
        "input": text_url,
        "output_preview": resp.choices[0].message.content[:300]
    }

except Exception as e:
    report["results"]["gpt_oss_120b"] = {
        "status": "failure",
        "error": str(e)
    }

# =====================================================
# 🎧 2. WHISPER (AUDIO MODEL)
# =====================================================
try:
    # FIXED: Switched from unstable university URLs to Wikimedia Commons
    audio_sources = [
        "https://upload.wikimedia.org/wikipedia/commons/4/45/En-us-hello.ogg",
        "https://upload.wikimedia.org/wikipedia/commons/3/3c/En-us-weather.ogg"
    ]
    audio_url = random.choice(audio_sources)
    audio_path = os.path.join(BASE_DIR, "sample_audio.ogg")

    with requests.get(audio_url, stream=True, timeout=30) as r:
        with open(audio_path, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "audio_file": audio_b64,
        "task": "transcribe",
        "language": "en",
        "without_timestamps": True
    }

    headers = {
        "Authorization": f"Bearer {SIMPLISMART_API_KEY}",
        "Content-Type": "application/json"
    }

    resp = requests.post(
        WHISPER_URL,
        json=payload,
        headers=headers,
        timeout=60
    ).json()

    report["results"]["whisper_large_v2"] = {
        "status": "success",
        "input": audio_url,
        "output_preview": str(resp)[:300]
    }

    os.remove(audio_path)

except Exception as e:
    report["results"]["whisper_large_v2"] = {
        "status": "failure",
        "error": str(e)
    }

# =====================================================
# 🖼️ 3. DEEPSEEK OCR (IMAGE MODEL)
# =====================================================
try:
    # FIXED: OCR now uses BASE64 image upload (not image_url)
    image_sources = [
        "https://upload.wikimedia.org/wikipedia/commons/4/4b/ReceiptSwiss.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/3/3f/Fax2.png"
    ]
    image_url = random.choice(image_sources)
    image_bytes = requests.get(image_url, timeout=20).content
    image_b64 = base64.b64encode(image_bytes).decode()

    client = OpenAI(
        api_key=SIMPLISMART_API_KEY,
        base_url=SIMPLISMART_BASE_URL,
        default_headers={
            "id": "81095ce8-515a-442a-8514-d4424ec84ce2"  # OCR tenant ID
        }
    )

    resp = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-OCR",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract all readable text from this image"},
                {"type": "image_base64", "image_base64": image_b64}
            ]
        }],
        max_tokens=500,
        temperature=0
    )

    report["results"]["deepseek_ocr"] = {
        "status": "success",
        "input": image_url,
        "output_preview": resp.choices[0].message.content[:300]
    }

except Exception as e:
    report["results"]["deepseek_ocr"] = {
        "status": "failure",
        "error": str(e)
    }

# =====================================================
# 💾 SAVE & PRINT JSON REPORT
# =====================================================
output_path = os.path.join(
    OUTPUT_DIR,
    f"daily_model_health_{today}.json"
)

with open(output_path, "w") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
