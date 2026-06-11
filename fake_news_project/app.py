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
actual_app_path = os.path.join(STREAMLIT_APP_DIR, "app.py")
with open(actual_app_path, "r", encoding="utf-8") as f:
    code = f.read()
    # Override __file__ so app.py resolves its own BASE_DIR correctly to streamlit_app/
    globals_dict = globals()
    globals_dict["__file__"] = actual_app_path
    exec(code, globals_dict)
