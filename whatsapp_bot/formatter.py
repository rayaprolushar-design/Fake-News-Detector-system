import re

APP_URL  = "https://srikar-verifyai.streamlit.app"   # your real URL
BOT_WA   = "14155238886"                              # your Twilio number, digits only

def fmt_share_message() -> str:
    """
    Returns a ready-to-forward message that users can long-press
    and tap Forward on, straight to their groups and contacts.
    """
    return f"""🔍 *Spread the word!*
━━━━━━━━━━━━━━━━━━━━
Forward this message to friends & family groups:

_"Hey! Found this WhatsApp bot that checks if news is fake in 5 seconds. Works in Hindi, Telugu & English too. Just forward any suspicious message to it:_
_wa.me/{BOT_WA}_

_Built by a student — pretty cool!"_

🌐 Web app: {APP_URL}
━━━━━━━━━━━━━━━━━━━━
_Tap and hold the message above → Forward_"""

def clean_markdown_for_whatsapp(text: str) -> str:
    """
    Cleans general markdown to make it WhatsApp-compatible:
    1. Replaces markdown headers (# Header) with *Header*.
    2. Replaces bullet list markers (- Item) with unicode bullet points (• Item).
    3. Trims consecutive newlines.
    """
    # Remove HTML tags if any
    text = re.sub(r'<[^>]*>', '', text)
    
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Match headers: e.g. "### Hello" -> "*Hello*"
        header_match = re.match(r'^#+\s+(.*)$', line.strip())
        if header_match:
            cleaned_lines.append(f"*{header_match.group(1).strip()}*")
            continue
            
        # Match bullet points: e.g. "- Item" -> "• Item"
        bullet_match = re.match(r'^\s*[-*+]\s+(.*)$', line)
        if bullet_match:
            cleaned_lines.append(f"• {bullet_match.group(1).strip()}")
            continue
            
        cleaned_lines.append(line)
        
    return '\n'.join(cleaned_lines)

DIVIDER = "━━━━━━━━━━━━━━━━━━━━"

def format_divider() -> str:
    return DIVIDER

def _footer(extra_tip: str = "") -> list:
    """Shared footer used by every reply — keeps branding consistent."""
    lines = ["", DIVIDER]
    if extra_tip:
        lines.append(extra_tip)
    lines += [
        f"_VerifyAI · {APP_URL}_",
        "_Type /share to spread the word_"
    ]
    return lines

def format_footer(lang: str = 'english') -> str:
    footers = {
        'english': f'\n━━━━━━━━━━━━━━━━━━━━\n🔍 *VerifyAI Bot*\nAlso try the web app:\n_{APP_URL}_',
        'hindi': f'\n━━━━━━━━━━━━━━━━━━━━\n🔍 *VerifyAI बॉट*\nवेब ऐप भी आज़माएं:\n_{APP_URL}_',
        'telugu': f'\n━━━━━━━━━━━━━━━━━━━━\n🔍 *VerifyAI బాట్*\nవెబ్ యాప్‌ని కూడా ప్రయత్నించండి:\n_{APP_URL}_'
    }
    return footers.get(lang, footers['english'])

HELP_MESSAGES = {
    'english': f"""🔍 *Welcome to VerifyAI!*
{DIVIDER}
I help you verify the authenticity of news and media.

*What I can analyze:*
📝 Send a *claim or message* to check for fake news.
🔗 Send a *URL link* to check source credibility & analyze the article.
🖼️ Send an *image* to check if it's AI-generated.
🤖 Type `/ai <text>` to verify if a text is AI-written.

{DIVIDER}
_VerifyAI · Always verify with trusted sources_""",

    'hindi': f"""🔍 *VerifyAI में आपका स्वागत है!*
{DIVIDER}
मैं आपको खबरों की सत्यता जांचने में मदद करता हूं।

*मैं क्या जांच सकता हूं:*
📝 कोई भी *खबर या संदेश* भेजें
🔗 कोई *URL* भेजें — मैं पूरा आर्टिकल जांचूंगा
🖼️ कोई *फोटो* भेजें — AI जेनरेटेड है या असली?
🤖 */ai <टेक्स्ट>* — AI ने लिखा है या इंसान ने?

{DIVIDER}
_VerifyAI · हमेशा विश्वसनीय स्रोतों से जांचें_""",

    'telugu': f"""🔍 *VerifyAI కి స్వాగతం!*
{DIVIDER}
నేను వార్తల నిజాయితీని తనిఖీ చేయడంలో మీకు సహాయం చేస్తాను।

*నేను ఏమి తనిఖీ చేయగలను:*
📝 ఏదైనా *వార్త లేదా సందేశం* పంపండి
🔗 *URL* పంపండి — నేను పూర్తి వ్యాసాన్ని తనిఖీ చేస్తాను
🖼️ *ఫోటో* పంపండి — AI జెనరేటెడ్ అా లేదా నిజమైనదా?
🤖 */ai <టెక్స్ట్>* — AI రాసిందా లేదా మానవుడు రాశాడా?

{DIVIDER}
_VerifyAI · నమ్మదగిన మూలాలతో ఎల్లప్పుడూ ధృవీకరించండి_"""
}

def fmt_help(lang: str = 'english') -> str:
    """Returns welcome/help guides based on user language."""
    if lang in HELP_MESSAGES:
        return HELP_MESSAGES[lang]
    return HELP_MESSAGES['english']

def fmt_news_result(headline: str, label_native: str, confidence: float, note: str, lang: str = 'english') -> str:
    """Formats news prediction output for WhatsApp in native languages."""
    headers = {
        'english': '🚨 *FAKE NEWS ANALYSIS*',
        'hindi': '🚨 *फेक न्यूज़ विश्लेषण*',
        'telugu': '🚨 *నకిలీ వార్తల విశ్లేషణ*'
    }
    
    claims = {
        'english': 'Claim',
        'hindi': 'दावा',
        'telugu': 'దావా'
    }
    
    verdicts = {
        'english': 'Verdict',
        'hindi': 'निष्कर्ष',
        'telugu': 'తీర్పు'
    }
    
    confidences = {
        'english': 'Confidence',
        'hindi': 'विश्वास प्रतिशत',
        'telugu': 'నమ్మకం'
    }
    
    analyses = {
        'english': 'Analysis',
        'hindi': 'विश्लेषण',
        'telugu': 'విశ్లేషణ'
    }
    
    header = headers.get(lang, headers['english'])
    claim_lbl = claims.get(lang, claims['english'])
    verdict_lbl = verdicts.get(lang, verdicts['english'])
    conf_lbl = confidences.get(lang, confidences['english'])
    analysis_lbl = analyses.get(lang, analyses['english'])
    
    lines = [
        header,
        DIVIDER,
        f"*{claim_lbl}:* \"_{headline.strip()}_\"",
        "",
        f"*{verdict_lbl}:* *{label_native}*",
        f"*{conf_lbl}:* *{confidence}%*",
        "",
        f"*{analysis_lbl}:* {note}"
    ]
    lines += _footer()
    return "\n".join(lines)

def fmt_ai_text_result(text: str, result: dict) -> str:
    """Formats AI vs Human text output for WhatsApp."""
    verdict_emoji = "🤖" if "AI" in result['verdict'] else "✅"
    verdict = f"{verdict_emoji} *{result['verdict']}*"
    
    signals = result['signals']
    signals_str = (
        f"• *Perplexity:* {signals['perplexity']}%\n"
        f"• *Burstiness:* {signals['burstiness']}%\n"
        f"• *AI Vocab:* {signals['ai_vocab']}%\n"
        f"• *Repetition:* {signals['repetition']}%\n"
        f"• *Readability:* {signals['readability']}%"
    )
    
    flagged = ", ".join([f"`{w}`" for w in result['flagged_words']]) if result['flagged_words'] else "_None_"
    excerpt = text.strip()[:100] + ("..." if len(text.strip()) > 100 else "")
    
    lines = [
        "🤖 *AI TEXT ANALYSIS*",
        DIVIDER,
        f"*Text Excerpt:* \"_{excerpt}_\"",
        "",
        f"*Verdict:* {verdict}",
        f"*AI Probability:* *{result['ai_score']}%*",
        "",
        f"*Linguistic Signals:*",
        signals_str,
        "",
        f"*AI-Overused Words:* {flagged}"
    ]
    lines += _footer()
    return "\n".join(lines)

def fmt_image_result(result: dict) -> str:
    """Formats AI Image detection output for WhatsApp."""
    verdict_emoji = "🤖" if "AI" in result['verdict'] else "📷"
    verdict = f"{verdict_emoji} *{result['verdict']}*"
    
    signals = result['signals']
    signals_str = (
        f"• *Noise Pattern:* {signals['noise_pattern']}%\n"
        f"• *DCT Frequency:* {signals['dct_frequency']}%\n"
        f"• *Colour Smoothness:* {signals['colour_smoothness']}%\n"
        f"• *Edge Uniformity:* {signals['edge_uniformity']}%"
    )
    
    meta = result['metadata']
    if meta['has_exif']:
        meta_str = f"• *Camera:* {meta['camera'] or '_Unknown_'}\n• *Software:* {meta['software'] or '_None_'}"
    else:
        meta_str = "• *Camera EXIF:* _Not Found_ (typical of AI/web images)"
        
    lines = [
        "🖼️ *AI IMAGE ANALYSIS*",
        DIVIDER,
        f"*Verdict:* {verdict}",
        f"*AI Probability:* *{result['ai_score']}%*",
        "",
        f"*Visual Signals:*",
        signals_str,
        "",
        f"*Metadata:*",
        meta_str
    ]
    lines += _footer()
    return "\n".join(lines)

def fmt_url_result(url: str, title: str, domain: str, credibility: dict, label: str, confidence: float, note: str) -> str:
    """Formats combined URL credibility and content analysis."""
    cred_rating = credibility.get('rating', 'UNKNOWN')
    cred_emoji = "🟢" if cred_rating == "TRUSTED" else "🔴" if cred_rating == "QUESTIONABLE" else "🟡"
    cred_score = credibility.get('score', 'N/A')
    
    cred_str = (
        f"• *Domain:* `{domain}`\n"
        f"• *Trust Level:* {cred_emoji} *{cred_rating}*\n"
        f"• *Score:* *{cred_score}/100*"
    )
    
    content_emoji = "✅" if "REAL" in label else "❌" if "FAKE" in label else "⚠️"
    content_verdict = f"{content_emoji} *{label}*"
    
    lines = [
        "🌐 *URL & ARTICLE ANALYSIS*",
        DIVIDER,
        f"*Article Title:* \"_{title.strip()}_\"",
        "",
        f"*Source Credibility:*",
        cred_str,
        "",
        f"*Content Analysis:*",
        f"• *Verdict:* {content_verdict}",
        f"• *Confidence:* *{confidence}%*",
        f"• *Detail:* {note}"
    ]
    lines += _footer()
    return "\n".join(lines)

# Legacy aliases for backward compatibility
def format_fake_news(*args, **kwargs) -> str:
    return fmt_news_result(*args, **kwargs)

def format_ai_text(*args, **kwargs) -> str:
    return fmt_ai_text_result(*args, **kwargs)

def format_ai_image(*args, **kwargs) -> str:
    return fmt_image_result(*args, **kwargs)

def format_url_analysis(*args, **kwargs) -> str:
    return fmt_url_result(*args, **kwargs)
