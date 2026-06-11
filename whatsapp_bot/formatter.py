import re

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

def format_divider() -> str:
    return "━━━━━━━━━━━━━━━━━━━━"

def format_footer(lang: str = 'english') -> str:
    footers = {
        'english': '\n━━━━━━━━━━━━━━━━━━━━\n🔍 *VerifyAI Bot*\nVerify before you share!',
        'hindi': '\n━━━━━━━━━━━━━━━━━━━━\n🔍 *VerifyAI बॉट*\nसाझा करने से पहले जांचें!',
        'telugu': '\n━━━━━━━━━━━━━━━━━━━━\n🔍 *VerifyAI బాట్*\nషేర్ చేయడానికి ముందు ధృవీకరించండి!'
    }
    return footers.get(lang, footers['english'])

HELP_MESSAGES = {
    'english': """🔍 *Welcome to VerifyAI!*
━━━━━━━━━━━━━━━━━━━━
I help you verify the authenticity of news and media.

*What I can analyze:*
📝 Send a *claim or message* to check for fake news.
🔗 Send a *URL link* to check source credibility & analyze the article.
🖼️ Send an *image* to check if it's AI-generated.
🤖 Type `/ai <text>` to verify if a text is AI-written.

━━━━━━━━━━━━━━━━━━━━
_VerifyAI · Always verify with trusted sources_""",

    'hindi': """🔍 *VerifyAI में आपका स्वागत है!*
━━━━━━━━━━━━━━━━━━━━
मैं आपको खबरों की सत्यता जांचने में मदद करता हूं।

*मैं क्या जांच सकता हूं:*
📝 कोई भी *खबर या संदेश* भेजें
🔗 कोई *URL* भेजें — मैं पूरा आर्टिकल जांचूंगा
🖼️ कोई *फोटो* भेजें — AI जेनरेटेड है या असली?
🤖 */ai <टेक्स्ट>* — AI ने लिखा है या इंसान ने?

━━━━━━━━━━━━━━━━━━━━
_VerifyAI · हमेशा विश्वसनीय स्रोतों से जांचें_""",

    'telugu': """🔍 *VerifyAI కి స్వాగతం!*
━━━━━━━━━━━━━━━━━━━━
నేను వార్తల నిజాయితీని తనిఖీ చేయడంలో మీకు సహాయం చేస్తాను।

*నేను ఏమి తనిఖీ చేయగలను:*
📝 ఏదైనా *వార్త లేదా సందేశం* పంపండి
🔗 *URL* పంపండి — నేను పూర్తి వ్యాసాన్ని తనిఖీ చేస్తాను
🖼️ *ఫోటో* పంపండి — AI జెనరేటెడ్ అా లేదా నిజమైనదా?
🤖 */ai <టెక్స్ట్>* — AI రాసిందా లేదా మానవుడు రాశాడా?

━━━━━━━━━━━━━━━━━━━━
_VerifyAI · నమ్మదగిన మూలాలతో ఎల్లప్పుడూ ధృవీకరించండి_"""
}

def fmt_help(lang: str = 'english') -> str:
    """Returns welcome/help guides based on user language."""
    if lang in HELP_MESSAGES:
        return HELP_MESSAGES[lang]
    return HELP_MESSAGES['english']

def format_fake_news(headline: str, label_native: str, confidence: float, note: str, lang: str = 'english') -> str:
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
    
    body = (
        f"{header}\n"
        f"{format_divider()}\n"
        f"*{claim_lbl}:* \"_{headline.strip()}_\"\n\n"
        f"*{verdict_lbl}:* *{label_native}*\n"
        f"*{conf_lbl}:* *{confidence}%*\n\n"
        f"*{analysis_lbl}:* {note}"
    )
    return body + format_footer(lang)

def format_ai_text(text: str, result: dict) -> str:
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
    
    body = (
        f"🤖 *AI TEXT ANALYSIS*\n"
        f"{format_divider()}\n"
        f"*Text Excerpt:* \"_{excerpt}_\"\n\n"
        f"*Verdict:* {verdict}\n"
        f"*AI Probability:* *{result['ai_score']}%*\n\n"
        f"*Linguistic Signals:*\n{signals_str}\n\n"
        f"*AI-Overused Words:* {flagged}"
    )
    return body + format_footer('english')

def format_ai_image(result: dict) -> str:
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
        
    body = (
        f"🖼️ *AI IMAGE ANALYSIS*\n"
        f"{format_divider()}\n"
        f"*Verdict:* {verdict}\n"
        f"*AI Probability:* *{result['ai_score']}%*\n\n"
        f"*Visual Signals:*\n{signals_str}\n\n"
        f"*Metadata:*\n{meta_str}"
    )
    return body + format_footer('english')

def format_url_analysis(url: str, title: str, domain: str, credibility: dict, label: str, confidence: float, note: str) -> str:
    """Formats combined URL credibility and content analysis."""
    # Source Credibility
    cred_rating = credibility.get('rating', 'UNKNOWN')
    cred_emoji = "🟢" if cred_rating == "TRUSTED" else "🔴" if cred_rating == "QUESTIONABLE" else "🟡"
    cred_score = credibility.get('score', 'N/A')
    
    cred_str = (
        f"• *Domain:* `{domain}`\n"
        f"• *Trust Level:* {cred_emoji} *{cred_rating}*\n"
        f"• *Score:* *{cred_score}/100*"
    )
    
    # Content analysis
    content_emoji = "✅" if "REAL" in label else "❌" if "FAKE" in label else "⚠️"
    content_verdict = f"{content_emoji} *{label}*"
    
    body = (
        f"🌐 *URL & ARTICLE ANALYSIS*\n"
        f"{format_divider()}\n"
        f"*Article Title:* \"_{title.strip()}_\"\n\n"
        f"*Source Credibility:*\n{cred_str}\n\n"
        f"*Content Analysis:*\n"
        f"• *Verdict:* {content_verdict}\n"
        f"• *Confidence:* *{confidence}%*\n"
        f"• *Detail:* {note}"
    )
    return body + format_footer('english')
