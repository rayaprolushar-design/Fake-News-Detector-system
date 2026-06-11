# factcheck_api.py
# Checks Google's Fact Check Tools API for existing human fact-checks
# Free API key: https://developers.google.com/fact-check/tools/api/reference/rest
# Get key at: https://console.cloud.google.com → Enable "Fact Check Tools API"

import requests, os
from urllib.parse import quote_plus

API_KEY = os.getenv('GOOGLE_FACTCHECK_API_KEY', '')
BASE_URL = 'https://factchecktools.googleapis.com/v1alpha1/claims:search'

def check_google_factcheck(query: str, max_results: int = 3) -> list:
    """
    Search Google's Fact Check database for a claim.
    Returns list of existing fact-checks or empty list if none found.
    """
    if not API_KEY:
        return []  # silently skip if no key configured

    try:
        resp = requests.get(BASE_URL, params={
            'key'          : API_KEY,
            'query'        : query[:200],
            'languageCode' : 'en',
            'pageSize'     : max_results
        }, timeout=5)

        data = resp.json()
        if 'claims' not in data: return []

        results = []
        for claim in data['claims'][:3]:
            reviews = claim.get('claimReview', [])
            if not reviews: continue
            review = reviews[0]
            results.append({
                'claim'      : claim.get('text', '')[:120],
                'rating'     : review.get('textualRating', 'Unknown'),
                'source'     : review.get('publisher', {}).get('name', ''),
                'url'        : review.get('url', ''),
                'review_date': review.get('reviewDate', '')[:10]
            })
        return results

    except Exception:
        return []  # never crash the main app for an API failure


def render_factcheck_results(results: list):
    """Renders fact-check results in Streamlit."""
    import streamlit as st
    if not results: return

    st.divider()
    st.markdown("**🔎 Human fact-checkers also reviewed this claim:**")

    for r in results:
        rating = r['rating'].lower()
        if   any(w in rating for w in ['false','fake','mislead','wrong']):
            icon = '❌'
        elif any(w in rating for w in ['true','correct','accurate']):
            icon = '✅'
        else:
            icon = '⚠️'

        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.05);
                    border-radius:8px;padding:12px 14px;margin-bottom:8px;">
          <div style="font-size:13px;color:#E8F0FE;margin-bottom:4px;">
            {icon} <strong>{r['rating']}</strong> — {r['source']}
          </div>
          <div style="font-size:12px;color:#cbd5e1;">{r['claim']}</div>
          <div style="font-size:11px;color:#4A6070;margin-top:4px;">
            <a href="{r['url']}" target="_blank"
               style="color:#2B7FD4;">Read full fact-check →</a>
             ·  {r['review_date']}
          </div>
        </div>
        """, unsafe_allow_html=True)
