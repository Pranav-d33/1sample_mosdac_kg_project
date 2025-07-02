import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Dataset URLs from open_data_crawler
DATASET_URLS = [
    "https://www.mosdac.gov.in/bayesian-based-mt-saphir-rainfall",
    "https://www.mosdac.gov.in/gps-derived-integrated-water-vapour",
    "https://www.mosdac.gov.in/gsmap-isro-rain",
    "https://www.mosdac.gov.in/meteosat8-cloud-properties",
    "https://www.mosdac.gov.in/3d-volumetric-terls-dwrproduct",
    "https://www.mosdac.gov.in/inland-water-height",
    "https://www.mosdac.gov.in/river-discharge",
    "https://www.mosdac.gov.in/soil-moisture",
    "https://www.mosdac.gov.in/global-ocean-surface-current",
    "https://www.mosdac.gov.in/high-resolution-sea-surface-salinity",
    "https://www.mosdac.gov.in/indian-mainland-coastal-product",
    "https://www.mosdac.gov.in/ocean-subsurface",
    "https://www.mosdac.gov.in/oceanic-eddies-detection",
    "https://www.mosdac.gov.in/sea-ice-occurrence-probability",
    "https://www.mosdac.gov.in/wave-based-renewable-energy"
]

# Site pages from mosdac_site_crawler
SITE_PAGES = [
    "https://www.mosdac.gov.in/insat-3d",
    "https://www.mosdac.gov.in/insat-3dr",
    "https://www.mosdac.gov.in/scatsat-1",
    "https://www.mosdac.gov.in/kalpana-1",
    "https://www.mosdac.gov.in/saral-altika",
    "https://www.mosdac.gov.in/oceansat-2",
    "https://www.mosdac.gov.in/insat-3a",
    "https://www.mosdac.gov.in/oceansat-3",
    "https://www.mosdac.gov.in/megha-tropiques",
    "https://www.mosdac.gov.in/insat-3ds",
    "https://www.mosdac.gov.in/insat-3dr-payloads",
    "https://www.mosdac.gov.in/insat-3dr-spacecraft",
    "https://www.mosdac.gov.in/insat-3dr-objectives",
    "https://www.mosdac.gov.in/insat-3d-objectives",
    "https://www.mosdac.gov.in/insat-3d-spacecraft",
    "https://www.mosdac.gov.in/insat-3d-payloads",
    "https://www.mosdac.gov.in/kalpana-1-objectives",
    "https://www.mosdac.gov.in/kalpana-1-spacecraft",
    "https://www.mosdac.gov.in/kalpana-1-payloads",
    "https://www.mosdac.gov.in/insat-3a-objectives",
    "https://www.mosdac.gov.in/insat-3a-spacecraft",
    "https://www.mosdac.gov.in/insat-3a-payloads",
    "https://www.mosdac.gov.in/megha-tropiques-objectives",
    "https://www.mosdac.gov.in/megha-tropiques-spacecraft",
    "https://www.mosdac.gov.in/megha-tropiques-payloads",
    "https://www.mosdac.gov.in/saral-altika-objectives",
    "https://www.mosdac.gov.in/saral-altika-payloads",
    "https://www.mosdac.gov.in/saral-altika-spacecraft",
    "https://www.mosdac.gov.in/oceansat-2-objectives",
    "https://www.mosdac.gov.in/oceansat-2-spacecraft",
    "https://www.mosdac.gov.in/oceansat-2-payloads",
    "https://www.mosdac.gov.in/oceansat-3-objectives",
    "https://www.mosdac.gov.in/oceansat-3-spacecraft",
    "https://www.mosdac.gov.in/oceansat-3-payloads",
    "https://www.mosdac.gov.in/insat-3s-objectives",
    "https://www.mosdac.gov.in/insat-3s-spacecraft",
    "https://www.mosdac.gov.in/insat-3s-payloads",
    "https://www.mosdac.gov.in/scatsat-1-objectives",
    "https://www.mosdac.gov.in/scatsat-1-spacecraft",
    "https://www.mosdac.gov.in/scatsat-1-payloads"
]

OUTPUT_FILE = "outputs/cleaned_json/site_data.json"

# Extensions for downloadable data
FILE_EXTS = (".hdf", ".nc", ".tif", ".tiff", ".zip", ".gz")

# Ensure output directory exists
def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# Use Selenium to fetch dynamic page source
def get_dynamic_html(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    driver.get(url)
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "table")))
    except:
        pass
    html = driver.page_source
    driver.quit()
    return html

# Crawl dataset page (from open_data_crawler)
def crawl_dataset_page(url):
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract title
    title_tag = soup.find(["h1", "h2"])
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Extract all visible paragraph text
    paragraphs = [
        p.get_text(strip=True)
        for p in soup.find_all("p")
        if p.get_text(strip=True)
    ]

    # Extract all tables (as lists of rows)
    tables = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)

    # Extract downloadable links
    download_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if any(href.lower().endswith(ext) for ext in FILE_EXTS):
            download_links.append(urljoin(url, href))

    # Extract image links (optional)
    image_links = [
        urljoin(url, img["src"])
        for img in soup.find_all("img", src=True)
    ]

    return {
        "url": url,
        "type": "dataset",
        "title": title,
        "paragraphs": paragraphs,
        "tables": tables,
        "download_links": download_links,
        "image_links": image_links
    }

# Parse site page (from mosdac_site_crawler)
def parse_site_page(page_url):
    try:
        if "catalog" in page_url:
            html = get_dynamic_html(page_url)
        else:
            html = requests.get(page_url, timeout=15).text
    except Exception as e:
        print(f"❌ Failed to load {page_url}: {e}")
        return None

    soup = BeautifulSoup(html, "html.parser")

    # Skip if page contains FAQ-like patterns
    if "faq" in page_url.lower() or soup.find(string=lambda text: text and "Frequently Asked Questions" in text):
        print(f"🚫 Skipping FAQ page: {page_url}")
        return None

    # 1. Meta tags
    metas = []
    for m in soup.find_all("meta", attrs={"content": True}):
        key = m.get("name") or m.get("property")
        if key:
            metas.append({"key": key, "content": m["content"]})

    # 2. Tables
    tables = []
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if cells:
                rows.append(cells)
        tables.append({"headers": headers, "rows": rows})

    # 3. ARIA labels
    aria_labels = []
    for el in soup.find_all(attrs={"aria-label": True}):
        aria_labels.append({
            "tag": el.name,
            "aria-label": el.get("aria-label"),
            "text": el.get_text(strip=True)
        })

    # 4. Mission overview / payload details
    mission_section = soup.select_one(".mission-overview, .payload-table, #content")
    mission_details = mission_section.get_text(separator="\n", strip=True) if mission_section else ""

    # 5. Product catalog extraction
    product_catalog = []
    for prod_table in soup.select("table"):
        for tr in prod_table.find_all("tr"):
            cols = [td.get_text(strip=True) for td in tr.find_all("td")]
            if cols:
                product_catalog.append(cols)

    return {
        "url": page_url,
        "type": "site_page",
        "meta": metas,
        "tables": tables,
        "aria_labels": aria_labels,
        "mission_details": mission_details,
        "product_catalog": product_catalog
    }

def main():
    all_data = []
    
    # Crawl dataset pages
    print("🔍 Crawling dataset pages...")
    for url in DATASET_URLS:
        print(f"🔍 Scraping dataset: {url}")
        try:
            data = crawl_dataset_page(url)
            all_data.append(data)
        except Exception as e:
            print(f"❌ Error with dataset {url}: {e}")
            all_data.append({"url": url, "type": "dataset", "error": str(e)})

    # Crawl site pages
    print("\n🔍 Crawling site structure pages...")
    for url in SITE_PAGES:
        print(f"🔍 Scraping site page: {url}")
        try:
            page_data = parse_site_page(url)
            if page_data:
                all_data.append(page_data)
        except Exception as e:
            print(f"❌ Error with site page {url}: {e}")
            all_data.append({"url": url, "type": "site_page", "error": str(e)})

    # Save merged output
    ensure_dir(OUTPUT_FILE)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    dataset_count = len([d for d in all_data if d.get("type") == "dataset"])
    site_count = len([d for d in all_data if d.get("type") == "site_page"])
    
    print(f"\n✅ Merged crawling complete!")
    print(f"📊 Dataset pages: {dataset_count}")
    print(f"🏗️ Site structure pages: {site_count}")
    print(f"💾 Total pages crawled: {len(all_data)}")
    print(f"📁 Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()