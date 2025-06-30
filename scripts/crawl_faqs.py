import requests
from bs4 import BeautifulSoup
import json
import os
import re

FAQ_URL = "https://www.mosdac.gov.in/faq-page"
OUTPUT_PATH = "outputs/cleaned_json/faqs.json"

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def clean_text(text):
    """Clean and normalize text content"""
    if not text:
        return ""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove any remaining HTML entities
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    return text

def debug_html_structure(soup):
    """Debug the HTML structure to understand the FAQ layout"""
    print("\n🔍 DEBUGGING HTML STRUCTURE:")
    
    # Look for common FAQ-related classes
    print("\n📋 Looking for common FAQ-related elements:")
    
    # Check for various div classes
    common_selectors = [
        "div[class*='faq']",
        "div[class*='question']", 
        "div[class*='answer']",
        "div[class*='accordion']",
        "div[class*='collapse']",
        "div[class*='view']",
        "div[class*='field']",
        "div[class*='content']"
    ]
    
    for selector in common_selectors:
        elements = soup.select(selector)
        if elements:
            print(f"✅ Found {len(elements)} elements with selector: {selector}")
            for i, elem in enumerate(elements[:2]):  # Show first 2
                classes = elem.get('class', [])
                print(f"   Element {i+1}: classes = {classes}")
    
    # Check for questions (elements containing question marks)
    print(f"\n❓ Looking for question patterns:")
    question_patterns = soup.find_all(text=re.compile(r'\?'))
    print(f"✅ Found {len(question_patterns)} text nodes containing '?'")
    
    # Show first few question candidates
    questions_found = []
    for text_node in question_patterns[:10]:
        parent = text_node.parent
        if parent and parent.name:
            text = clean_text(str(text_node))
            if len(text) > 10 and len(text) < 300:
                questions_found.append({
                    'text': text,
                    'parent_tag': parent.name,
                    'parent_classes': parent.get('class', [])
                })
    
    print(f"\n📝 Sample question candidates:")
    for i, q in enumerate(questions_found[:5]):
        print(f"{i+1}. Text: {q['text'][:80]}...")
        print(f"   Parent: <{q['parent_tag']}> with classes: {q['parent_classes']}")
    
    # Look for main content areas
    print(f"\n🏠 Looking for main content containers:")
    main_selectors = [
        "div#content",
        "div.content", 
        "main",
        "div[class*='main']",
        "div[class*='region']",
        "div[class*='block']"
    ]
    
    for selector in main_selectors:
        elements = soup.select(selector)
        if elements:
            print(f"✅ Found main container: {selector} ({len(elements)} elements)")

def crawl_faqs():
    print(f"🔍 Crawling FAQs from: {FAQ_URL}")
    
    try:
        # Add headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        resp = requests.get(FAQ_URL, headers=headers, timeout=15)
        resp.raise_for_status()
        print(f"✅ Successfully fetched FAQ page ({len(resp.text)} characters)")
    except Exception as e:
        print(f"❌ Failed to fetch FAQ page: {e}")
        return

    soup = BeautifulSoup(resp.text, 'html.parser')
    
    # Debug the HTML structure first
    debug_html_structure(soup)
    
    # Target the specific MOSDAC FAQ structure
    print("\n🔍 Starting FAQ extraction...")
    
    faqs = []
    
    # Strategy 1: Look for any elements containing questions
    print("\n📋 Strategy 1: Broad question search...")
    all_elements = soup.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'strong', 'b'])
    
    question_candidates = []
    for elem in all_elements:
        text = clean_text(elem.get_text())
        if "?" in text and 10 < len(text) < 400:
            question_candidates.append({
                'element': elem,
                'text': text,
                'tag': elem.name,
                'classes': elem.get('class', [])
            })
    
    print(f"✅ Found {len(question_candidates)} potential questions")
    
    # Process question candidates
    for i, candidate in enumerate(question_candidates[:20]):  # Process first 20
        question = candidate['text']
        elem = candidate['element']
        
        # Try to find answer near this question
        answer = ""
        
        # Strategy A: Answer in same element (like <p><strong>Q?</strong> Answer</p>)
        if elem.find(['strong', 'b']):
            strong_elem = elem.find(['strong', 'b'])
            if strong_elem and "?" in strong_elem.get_text():
                question = clean_text(strong_elem.get_text())
                full_text = clean_text(elem.get_text())
                answer = full_text.replace(question, "").strip()
        
        # Strategy B: Answer in next sibling elements
        if not answer or len(answer) < 10:
            answer_parts = []
            for sibling in elem.find_next_siblings(['p', 'div'])[:3]:
                sibling_text = clean_text(sibling.get_text())
                if sibling_text and len(sibling_text) > 5:
                    answer_parts.append(sibling_text)
            if answer_parts:
                answer = ' '.join(answer_parts)
        
        # Strategy C: Answer in parent container
        if not answer or len(answer) < 10:
            parent = elem.parent
            if parent:
                parent_text = clean_text(parent.get_text())
                if len(parent_text) > len(question) + 20:
                    answer = parent_text.replace(question, "").strip()
        
        if question and len(question) > 10:
            faqs.append({
                "question": question,
                "answer": answer[:1000] if answer else "",  # Limit answer length
                "source_tag": candidate['tag'],
                "source_classes": candidate['classes']
            })
            print(f"   ✅ Extracted FAQ {i+1}: {question[:50]}... (Answer: {len(answer)} chars)")
    
    # Remove duplicates
    unique_faqs = []
    seen_questions = set()
    
    for faq in faqs:
        question = faq["question"].strip()
        if question and question not in seen_questions:
            unique_faqs.append({
                "question": question,
                "answer": faq["answer"]
            })
            seen_questions.add(question)
    
    faqs = unique_faqs

    # Save results
    ensure_dir(OUTPUT_PATH)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(faqs, f, indent=2, ensure_ascii=False)

    print(f"\n📊 Final Results:")
    print(f"✅ Extracted {len(faqs)} unique FAQs to {OUTPUT_PATH}")
    
    # Show summary
    complete_faqs = sum(1 for faq in faqs if faq["answer"].strip())
    question_only = len(faqs) - complete_faqs
    print(f"   📄 Complete FAQs (with answers): {complete_faqs}")
    print(f"   ❔ Question-only FAQs: {question_only}")
    
    # Show examples
    print(f"\n📋 Sample FAQs:")
    for i, faq in enumerate(faqs[:3]):
        answer_preview = faq["answer"][:150] + "..." if len(faq["answer"]) > 150 else faq["answer"]
        print(f"\n{i+1}. Q: {faq['question']}")
        print(f"   A: {answer_preview or '[No answer found]'}")
    
    # Also save the HTML for manual inspection if needed
    with open("debug_faq_page.html", "w", encoding="utf-8") as f:
        f.write(soup.prettify())
    print(f"\n🔧 Saved HTML structure to debug_faq_page.html for manual inspection")

if __name__ == "__main__":
    crawl_faqs()