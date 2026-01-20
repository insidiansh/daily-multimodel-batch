import os
import json
import datetime
import random
import requests
import base64
from openai import OpenAI

# CONFIG

os.environ["SIMPLISMART_API_KEY"] = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1dWlkIjoiMjcxMDc0ZDctYTM0NS00MDkzLWE0ZWEtOGNiNzRhODk2MWVlIiwiZXhwIjoxNzY4OTg3MzcwLCJvcmdfdXVpZCI6IjgwNzcwZGZjLTc3ZTAtNDE2MC05M2NhLTY3OGRlMmRkMTY1MSJ9.VO-IYZUTFkCKg_0FWC65PVqGkKw_65s_Aa_9MDZHa5s"
)
SIMPLISMART_API_KEY = os.getenv("SIMPLISMART_API_KEY")
if not SIMPLISMART_API_KEY:
    raise RuntimeError("SIMPLISMART_API_KEY not set")

SIMPLISMART_BASE_URL = "https://api.simplismart.live"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# METADATA

today = datetime.date.today().isoformat()
run_time = datetime.datetime.now().strftime("%H:%M")

report = {
    "date": today,
    "run_time": run_time,
    "results": {}
}

# 1. GPT-OSS-120B (TEXT)

try:
    text_sources = [
        "https://www.gutenberg.org/files/84/84-0.txt",
        "https://www.gutenberg.org/files/1342/1342-0.txt",
        "https://www.gutenberg.org/files/11/11-0.txt"
    ]

    text_url = random.choice(text_sources)
    text = requests.get(text_url, timeout=20).text[:1500]

    client = OpenAI(
        api_key=SIMPLISMART_API_KEY,
        base_url=SIMPLISMART_BASE_URL,
        default_headers={"id": "524436ef-5d4c-4d55-9351-71d67036b92b"}
    )

    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": f"Summarize this text:\n{text}"}],
        max_tokens=200,
        temperature=0
    )

    report["results"]["openai/gpt-oss-120b"] = {
        "status": 200
    }

except Exception as e:
    report["results"]["openai/gpt-oss-120b"] = {
        "status": 500
    }
# 2. DEEPSEEK OCR (IMAGE)
try:
    image_sources = [
        "https://upload.wikimedia.org/wikipedia/commons/4/4b/ReceiptSwiss.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/3/3f/Fax2.png"
    ]

    image_url = random.choice(image_sources)

    image_bytes = requests.get(
        image_url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30
    ).content

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    client = OpenAI(
        api_key=SIMPLISMART_API_KEY,
        base_url=SIMPLISMART_BASE_URL,
        default_headers={"id": "81095ce8-515a-442a-8514-d4424ec84ce2"}
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
        max_tokens=300,
        temperature=0
    )

    report["results"]["deepseek-ai/DeepSeek-OCR"] = {
        "status": 200
    }

except Exception as e:
    report["results"]["deepseek-ai/DeepSeek-OCR"] = {
        "status": 500
    }

output_path = os.path.join(
    OUTPUT_DIR,
    f"daily_model_health_{today}.json"
)

with open(output_path, "w") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
