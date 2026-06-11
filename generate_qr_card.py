# generate_qr_card.py
# Generates a professional QR card with both app link and WhatsApp link
# pip install qrcode[pil] Pillow

import qrcode
from PIL import Image, ImageDraw, ImageFont
import os

# ── Your links — update these ────────────────────
APP_URL = "https://srikar-verifyai.streamlit.app"
WA_URL  = "https://wa.me/14155238886"  # your Twilio number

def make_qr(url: str, size: int = 300) -> Image.Image:
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10, border=2
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#0D1117", back_color="white")
    return img.resize((size, size))

def generate_card():
    # Card dimensions — A6 size at 150dpi
    W, H = 630, 420
    card = Image.new('RGB', (W, H), '#0D1117')
    draw = ImageDraw.Draw(card)

    # Title
    draw.text((30, 30), "🔍 VerifyAI",
               fill="#E8F0FE", font=ImageFont.load_default())
    draw.text((30, 55), "Fake news detector · Hindi · Telugu · English",
               fill="#4A6070", font=ImageFont.load_default())

    # Divider line
    draw.line([(30,80),(W-30,80)], fill="#1E2D40", width=1)

    # QR codes
    qr_app = make_qr(APP_URL, 260)
    qr_wa  = make_qr(WA_URL,  260)
    card.paste(qr_app, (30,  100))
    card.paste(qr_wa,  (340, 100))

    # Labels under QR codes
    draw.text((80,  368), "🌐 Web App",    fill="#2B7FD4")
    draw.text((390, 368), "💬 WhatsApp Bot", fill="#25D366")

    # Footer
    draw.text((30, 395),
               "99.6% accuracy · DistilBERT · Built by Srikar Rayaprolu",
               fill="#4A6070")

    card.save("verifyai_card.png", dpi=(150,150))
    print("Saved: verifyai_card.png — print this!")
    return card

generate_card()
