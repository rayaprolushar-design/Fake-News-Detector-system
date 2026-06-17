import os
from fastapi import FastAPI, Form, Response
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

import sys
# Load environment variables first
load_dotenv()

# Add script directory to sys.path to allow absolute/relative imports when running from root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from bot import handle_message

app = FastAPI(title="Twilio WhatsApp Fake News Webhook")

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
        <head>
            <title>WhatsApp Fake News Detector Bot</title>
            <style>
                body { font-family: sans-serif; background-color: #0d0a1c; color: #e2e8f0; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; }
                .container { background: rgba(255, 255, 255, 0.05); padding: 40px; border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 8px 32px rgba(0,0,0,0.3); text-align: center; }
                h1 { color: #8b5cf6; }
                p { color: #cbd5e1; }
                .badge { background: #10b981; color: white; padding: 6px 12px; border-radius: 9999px; font-size: 0.9rem; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔍 WhatsApp Fake News Bot Server</h1>
                <p>FastAPI Server is online and listening for webhook requests.</p>
                <div style="margin-top: 20px;"><span class="badge">STATUS: ACTIVE</span></div>
            </div>
        </body>
    </html>
    """

@app.post("/webhook")
async def webhook(
    Body: str = Form(None),
    MediaUrl0: str = Form(None),
    ContentType0: str = Form(None),
    From: str = Form(None),
    NumMedia: int = Form(0)
):
    # Parse inputs
    message_body = Body or ""
    media_url = MediaUrl0 or None
    media_type = ContentType0 or None
    
    print(f"Received webhook: From={From}, Body='{message_body}', MediaUrl={media_url}, Type={media_type}, NumMedia={NumMedia}")
    
    try:
        reply_text = handle_message(
            body=message_body,
            media_url=media_url,
            media_type=media_type,
            num_media=NumMedia,
            from_number=From or ""
        )
    except Exception as e:
        reply_text = f"❌ *Server Webhook Error:* {str(e)}"
        print(f"Error handling message: {e}")
        
    # Construct Twilio Messaging Response XML
    from twilio.twiml.messaging_response import MessagingResponse
    resp = MessagingResponse()
    resp.message(reply_text)
    
    return Response(content=str(resp), media_type="application/xml")
