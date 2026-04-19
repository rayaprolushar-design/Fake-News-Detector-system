import requests
import re
import time
from bs4 import BeautifulSoup
import random

# Rotate user agents so sites don't recognise us as a bot
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 '
    '(KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
]

def _get_headers():
    return {
        'User-Agent'     : random.choice(USER_AGENTS),
        'Accept'         : 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9,hi;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection'     : 'keep-alive',
        'Referer'        : 'https://www.google.com/',
    }

def _strategy_beautifulsoup(url: str) -> dict:
    """Strategy 1: BeautifulSoup — works on most standard news sites."""
    resp = requests.get(url, headers=_get_headers(), timeout=12)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, 'lxml')

    # Remove noise tags
    for tag in soup.find_all([
        'script', 'style', 'nav', 'footer', 'header',
        'aside', 'form', 'iframe', 'noscript', 'ads'
    ]):
        tag.decompose()

    # Try article tag first, then main, then body
    container = (
        soup.find('article') or
        soup.find('main')    or
        soup.find('div', class_=re.compile(r'article|content|story|body', re.I)) or
        soup.find('body')
    )

    title = ''
    if soup.find('h1'):
        title = soup.find('h1').get_text(strip=True)
    elif soup.find('title'):
        title = soup.find('title').get_text(strip=True)

    paragraphs = container.find_all('p') if container else []
    text = ' '.join(p.get_text(strip=True) for p in paragraphs if len(p.get_text()) > 40)
    text = re.sub(r'\s+', ' ', text).strip()

    if len(text) < 150:
        raise ValueError("Not enough content extracted")

    return {'title': title, 'text': title + ' ' + text,
            'strategy': 'BeautifulSoup', 'error': None}


def _strategy_newspaper(url: str) -> dict:
    """Strategy 2: newspaper3k — handles many edge cases BS4 misses."""
    from newspaper import Article
    article = Article(url)
    article.download()
    article.parse()

    if len(article.text) < 100:
        raise ValueError("Newspaper extracted too little text")

    return {
        'title'   : article.title,
        'text'    : article.title + ' ' + article.text,
        'strategy': 'newspaper3k',
        'error'   : None
    }


def _strategy_raw_text(url: str) -> dict:
    """Strategy 3: Raw text dump — last resort, works on almost anything."""
    resp = requests.get(url, headers=_get_headers(), timeout=15)
    soup = BeautifulSoup(resp.text, 'lxml')

    # Just grab ALL text, strip HTML — messy but gets something
    raw  = soup.get_text(separator=' ', strip=True)
    text = re.sub(r'\s+', ' ', raw)[:3000]  # cap at 3000 chars

    title = ''
    if soup.find('title'):
        title = soup.find('title').get_text(strip=True)

    if len(text) < 80:
        raise ValueError("Raw text too short — site may be JS-only")

    return {'title': title, 'text': title + ' ' + text,
            'strategy': 'raw text', 'error': None}


def scrape_article(url: str, retries: int = 2) -> dict:
    """
    Main scraping function.
    Tries 3 strategies in order with retry logic.
    Returns: {'title', 'text', 'strategy', 'error'}
    """
    url = url.strip()
    if not url.startswith('http'):
        url = 'https://' + url

    strategies = [
        ('BeautifulSoup', _strategy_beautifulsoup),
        ('newspaper3k',   _strategy_newspaper),
        ('raw text',      _strategy_raw_text),
    ]

    last_error = ''
    for attempt in range(retries + 1):
        for name, strategy in strategies:
            try:
                result = strategy(url)
                result['attempts'] = attempt + 1
                return result
            except Exception as e:
                last_error = f"{name}: {str(e)[:80]}"
                continue

        if attempt < retries:
            time.sleep(1.5)  # wait before retry

    return {
        'title'   : '',
        'text'    : '',
        'strategy': 'none',
        'error'   : f"All strategies failed. Last error: {last_error}"
    }


# Test
if __name__ == '__main__':
    test_urls = [
        'https://www.bbc.com/news',
        'https://ndtv.com',
        'https://timesofindia.indiatimes.com',
    ]
    for url in test_urls:
        r = scrape_article(url)
        status = 'OK' if not r['error'] else 'FAIL'
        print(f"[{status}] {url[:40]:40s} strategy={r.get('strategy','?')}")
        if not r['error']:
            print(f"  Title: {r['title'][:60]}")
            print(f"  Text : {r['text'][:80]}...")
