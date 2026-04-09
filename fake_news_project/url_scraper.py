import requests
from bs4 import BeautifulSoup
import re

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   'AppleWebKit/537.36 Chrome/120.0 Safari/537.36'
}

def scrape_article(url: str) -> dict:
    """
    Fetch a news article from a URL.
    Returns: {'title': str, 'text': str, 'error': str or None}
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return {'title': '', 'text': '', 'error': f"Could not fetch URL: {e}"}

    soup = BeautifulSoup(resp.text, 'lxml')

    # Remove noise: scripts, styles, nav, ads, footers
    for tag in soup.find_all(['script','style','nav','footer',
                               'header','aside','form','iframe']):
        tag.decompose()

    # Extract title
    title = ''
    if soup.find('h1'):
        title = soup.find('h1').get_text(strip=True)
    elif soup.find('title'):
        title = soup.find('title').get_text(strip=True)

    # Extract article paragraphs — most content lives in p tags
    paragraphs = soup.find_all('p')
    text = ' '.join(p.get_text(strip=True) for p in paragraphs)

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    if len(text) < 100:
        return {
            'title': title, 'text': text,
            'error': 'Article text too short — site may block scraping.'
        }

    return {'title': title, 'text': title + ' ' + text, 'error': None}
