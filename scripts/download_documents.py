import os
import json
import requests
from urllib.parse import urlparse
from tqdm import tqdm

INPUT_FILE = "outputs/cleaned_json/documents.json"
OUTPUT_DIR = "data/docs"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_filename_from_url(url):
    return os.path.basename(urlparse(url).path)

def download_file(url, out_path):
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False

def download_all():
    ensure_dir(OUTPUT_DIR)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"🔽 Starting download of {len(documents)} files...")
    for doc in tqdm(documents):
        url = doc["url"]
        fname = get_filename_from_url(url)
        out_path = os.path.join(OUTPUT_DIR, fname)

        # Skip if already downloaded
        if os.path.exists(out_path):
            continue

        success = download_file(url, out_path)
        if not success:
            continue

    print(f"✅ Downloaded documents saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    download_all()
