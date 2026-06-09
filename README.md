# 🔍 VerifyAI: Multilingual Fake News & AI Media Detector

VerifyAI is an end-to-end, real-time fake news and AI media verification system. Powered by state-of-the-art Natural Language Processing (DistilBERT + IndicBERT) and pixel-level signal analysis, it enables users to verify text claims, news links, AI-generated content, and images directly through a seamless WhatsApp interface.

---

## 🚀 Key Features

* **Multilingual News Classification**: Support for **English, Hindi, and Telugu** claims. It automatically routes the input to specialized models and translates responses natively.
* **AI Text Detection**: Analyzes sentence perplexity, burstiness, vocabulary entropy, and repetition to identify text written by models like ChatGPT, Claude, or Gemini.
* **AI Image Detector**: Evaluates discrete cosine transform (DCT) frequencies, sensor noise consistency, color smoothness, and edge uniformity to identify AI-generated images.
* **URL & Scraper Engine**: Automatically scrapes articles from pasted links, validates domain trust ratings via a credibility index (e.g. NDTV, AltNews, BBC), and runs content audits.
* **WhatsApp Chatbot Integration**: Direct integration with Twilio WhatsApp API for real-time, zero-install messaging.

---

## 🛠️ Tech Stack

* **Web Framework**: FastAPI (Uvicorn server)
* **NLP Models**: Hugging Face Transformers (`DistilBERT` for English, `ai4bharat/indic-bert` for Indian languages)
* **ML/Data Science**: PyTorch, Scikit-Learn (TF-IDF + Stylistic Feature Fusion)
* **Computer Vision**: OpenCV, Pillow, SciPy (for pixel noise and DCT frequency analysis)
* **Language Detection**: `langdetect` + script Unicode range heuristics
* **APIs & Webhooks**: Twilio WhatsApp API

---

## 📦 Installation & Setup

### 1. Clone & Set Up Environment
```bash
# Navigate to project directory
cd "Fake News Detector"

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install twilio fastapi uvicorn python-dotenv python-multipart langdetect deep-translator
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory (refer to `.env.example`):
```env
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=whatsapp:+14155238886  # Twilio Sandbox number
```

---

## 🖥️ How to Run Locally

### 1. Test Webhook Engine
Verify that the routing and classifier layers work properly on local mock payloads:
```bash
python test_bot.py
```

### 2. Start Webhook Server
Launch the FastAPI webhook endpoint:
```bash
uvicorn server:app --reload --port 8000
```

### 3. Expose Server to Internet
Use `localtunnel` (or `ngrok`) to generate a public webhook URL:
```bash
npx -y localtunnel --port 8000
```
Copy the generated URL (e.g. `https://your-tunnel-subdomain.loca.lt/webhook`).

### 4. Configure Twilio WhatsApp Webhook
1. Log in to [console.twilio.com](https://console.twilio.com).
2. Go to **Messaging** > **Try it out** > **Send a WhatsApp message**.
3. Under the **Sandbox settings** tab, paste your public URL into the **"WHEN A MESSAGE COMES IN"** field.
4. Set the HTTP method to **POST** and click **Save**.
5. Save the Sandbox number to your contacts, message `join sandbox-code`, and start verifying!

---

## 🧪 Training IndicBERT for Indian Languages

If you wish to fine-tune the multilingual IndicBERT model on custom Hindi/Telugu datasets:

1. **Compile Dataset**:
   ```bash
   python build_hindi_dataset.py
   ```
   This will assemble Hugging Face datasets and build `hindi_telugu_dataset.csv`.

2. **Run Training Loop**:
   ```bash
   python indic_model.py
   ```
   *Note: Training is optimized for GPU/MPS (Apple Silicon) acceleration. For solo runs, fine-tuning is recommended on a Google Colab GPU notebook.*

---

## 🌟 Demo Flows

Test these sequences on WhatsApp to demo the application's capabilities:

* **Text Verification**: Send any rumor (e.g. `"Government is giving free 5G phones to all citizens."`).
* **Hinglish/Hindi**: Send `"यह खबर बिल्कुल झूठी है और लोगों को गुमराह करती है"` or `"Yeh khabar bilkul jhoot hai"`.
* **AI Text Detection**: Send `/ai It is crucial to leverage comprehensive strategies that facilitate robust outcomes. Furthermore, this ensures streamlined implementation.`
* **AI Image Check**: Send or forward an AI-generated image attachment.
