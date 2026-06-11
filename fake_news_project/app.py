import os
import sys

# Set up paths so that imports inside streamlit_app work
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STREAMLIT_APP_DIR = os.path.join(BASE_DIR, "streamlit_app")
if STREAMLIT_APP_DIR not in sys.path:
    sys.path.insert(0, STREAMLIT_APP_DIR)

# Change directory so relative file lookups (models, config) resolve correctly
os.chdir(STREAMLIT_APP_DIR)

# Execute the actual app logic
with open(os.path.join(STREAMLIT_APP_DIR, "app.py"), "r", encoding="utf-8") as f:
    code = f.read()
    exec(code, globals())
