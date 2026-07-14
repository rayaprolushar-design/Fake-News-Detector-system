# pib_checker.py
# Integrates PIB Fact Check and official Indian govt sources

import requests
from urllib.parse import quote_plus

# Official Indian government news sources — highest credibility
GOVT_SOURCES = {
    'pib.gov.in'       : (10.0, '🇮🇳 PIB — Press Information Bureau'),
    'mygov.in'          : (9.8,  '🇮🇳 MyGov — Official Govt of India'),
    'india.gov.in'      : (9.8,  '🇮🇳 National Portal of India'),
    'mohfw.gov.in'      : (9.7,  '🇮🇳 Ministry of Health'),
    'meity.gov.in'      : (9.7,  '🇮🇳 MeitY'),
    'rbi.org.in'        : (9.9,  '🇮🇳 Reserve Bank of India'),
    'eci.gov.in'        : (9.9,  '🇮🇳 Election Commission of India'),
    'supremecourt.gov.in': (9.9, '🇮🇳 Supreme Court of India'),
    'isro.gov.in'       : (9.8,  '🇮🇳 ISRO'),
    'icmr.gov.in'       : (9.7,  '🇮🇳 ICMR'),
}

def check_pib(query: str) -> list:
    """
    Searches PIB Fact Check RSS feed for matching fact-checks.
    Returns list of matching PIB verdicts.
    """
    try:
        # PIB Fact Check has a public RSS feed
        rss_url = f"https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3"
        resp = requests.get(rss_url, timeout=5)
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.content)

        results = []
        query_words = set(query.lower().split())

        for item in root.findall('.//item')[:20]:
            title = item.find('title')
            link  = item.find('link')
            if title is None: continue

            title_text = title.text or ''
            title_words = set(title_text.lower().split())

            # Check overlap between query and PIB fact-check title
            overlap = query_words & title_words - {
                'the','is','a','of','in','and','to','that'
            }
            if len(overlap) >= 2:
                results.append({
                    'title'  : title_text[:100],
                    'url'    : link.text if link is not None else '',
                    'source' : 'PIB Fact Check',
                    'rating' : 'GOVERNMENT VERIFIED'
                })
        return results[:2]

    except:
        return []


def get_govt_credibility(url: str):
    """Check if a URL is from an official government source."""
    import re
    try:
        domain = re.search(r'(?:https?://)?(?:www\.)?([^/]+)', url).group(1)
        for gov_domain, (score, name) in GOVT_SOURCES.items():
            if gov_domain in domain:
                return score, name
    except: pass
    return None, None


def fmt_pib_result(pib_results: list) -> str:
    """Format PIB fact-check results for WhatsApp."""
    if not pib_results: return ""
    lines = ["\n🇮🇳 *PIB Fact Check also reviewed this:*"]
    for r in pib_results:
        lines.append(f"  ✓ {r['title']}")
        lines.append(f"  _Source: {r['source']}_")
    return "\n".join(lines)
