import requests
from bs4 import BeautifulSoup
import json
import os
from urllib.parse import urljoin

PAGES_TO_CRAWL = [
  "https://www.mosdac.gov.in/insat-3d-references",
  "https://www.mosdac.gov.in/insat-3dr-references",
  "https://www.mosdac.gov.in/scatsat-1-references",
  "https://www.mosdac.gov.in/oceansat-2-references",
  "https://www.mosdac.gov.in/oceansat3-references",
  "https://www.mosdac.gov.in/megha-tropiques-references",
  "https://www.mosdac.gov.in/saral-references",
  "https://mosdac.gov.in/thredds/catalog.html",
  "https://www.mosdac.gov.in/atlases",
  "https://www.mosdac.gov.in/tools",
  "https://www.mosdac.gov.in/indian-mainland-coastal-product",
  "https://www.mosdac.gov.in/insat-3s-references",
]




OUTPUT_FILE = "outputs/cleaned_json/documents.json"
FILE_EXTENSIONS = [".pdf", ".docx", ".xlsx", ".zip", ".rar", ".tar.gz"]

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def is_document_link(href):
    return any(href.lower().endswith(ext) for ext in FILE_EXTENSIONS)

def crawl_documents():
    documents = []

    for page_url in PAGES_TO_CRAWL:
        try:
            print(f"🔍 Crawling: {page_url}")
            resp = requests.get(page_url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            for a in soup.find_all("a", href=True):
                href = a["href"]
                if is_document_link(href):
                    full_url = urljoin(page_url, href)
                    title = a.get_text(strip=True) or os.path.basename(href)
                    documents.append({
                        "title": title,
                        "url": full_url,
                        "source_page": page_url
                    })
        except Exception as e:
            print(f"⚠️ Error scraping {page_url}: {e}")

    ensure_dir(OUTPUT_FILE)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    print(f"✅ Found and saved {len(documents)} document links to {OUTPUT_FILE}")

if __name__ == "__main__":
    crawl_documents()
