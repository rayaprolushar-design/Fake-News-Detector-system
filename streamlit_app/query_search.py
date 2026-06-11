import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote

def search_google_news(query: str, num_results: int = 5) -> list:
    """
    Search Google News RSS feed for a given query.
    Returns a list of dicts with title, link, and pubDate.
    """
    if not query:
        return []
        
    url = f"https://news.google.com/rss/search?q={quote(query)}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        
        root = ET.fromstring(resp.text)
        items = root.findall('.//item')
        
        results = []
        for item in items[:num_results]:
            title = item.find('title').text if item.find('title') is not None else 'No Title'
            link = item.find('link').text if item.find('link') is not None else ''
            pubDate = item.find('pubDate').text if item.find('pubDate') is not None else ''
            
            results.append({
                'title': title,
                'link': link,
                'published': pubDate
            })
            
        return results
    except Exception as e:
        print(f"Error fetching Google News for query '{query}': {e}")
        return []

if __name__ == "__main__":
    import sys
    search_query = "India AI deepfake" if len(sys.argv) == 1 else sys.argv[1]
    res = search_google_news(search_query)
    for r in res:
        print(f"- {r['title']}")
        print(f"  {r['link']}")
        print()
