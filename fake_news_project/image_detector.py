# image_detector.py
# Detects whether an image is AI-generated or a real photo
# Uses pixel-level signal analysis — no API or internet needed

import numpy as np
from PIL import Image
import cv2
from scipy import ndimage
import io

def load_image(source) -> np.ndarray:
    """
    Load image from a file path, bytes, or PIL Image.
    Returns numpy array in RGB format.
    """
    if isinstance(source, np.ndarray):
        return source
    elif isinstance(source, Image.Image):
        return np.array(source.convert('RGB'))
    elif isinstance(source, (bytes, io.BytesIO)):
        if isinstance(source, bytes):
            source = io.BytesIO(source)
        img = Image.open(source).convert('RGB')
        return np.array(img)
    else:
        img = Image.open(source).convert('RGB')
        return np.array(img)

def analyse_noise(img: np.ndarray) -> float:
    """
    Analyse noise pattern.
    AI images have suspiciously structured or absent noise.
    Real camera images have natural gaussian noise from the sensor.
    Returns 0-1 where 1 = very AI-like noise pattern.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float)

    # Extract noise by subtracting a smoothed version
    smooth   = ndimage.gaussian_filter(gray, sigma=1.5)
    noise    = gray - smooth

    # Real camera noise: std ~3-12 with near-gaussian distribution
    # AI noise: std ~0-3 (too smooth) or >15 (overly textured)
    std      = np.std(noise)
    skewness = np.abs(np.mean((noise - noise.mean())**3) /
                       (noise.std()**3 + 1e-8))

    # Too smooth (std<2) or wrong distribution → AI-like
    if std < 2.0:
        score = 0.85
    elif std > 18:
        score = 0.70
    else:
        # Natural range — penalise for non-gaussian skewness
        score = min(skewness / 3.0, 1.0) * 0.5

    return round(score, 3)

def analyse_dct_frequency(img: np.ndarray) -> float:
    """
    DCT (Discrete Cosine Transform) frequency analysis.
    AI image generators produce distinctive high-frequency patterns
    that differ from real JPEG camera photos.
    Returns 0-1 where 1 = very AI-like frequency signature.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Resize to standard size for fair comparison
    gray = cv2.resize(gray, (256, 256))

    # Apply DCT
    dct  = cv2.dct(gray / 255.0)
    dct_abs = np.abs(dct)

    # Split into low/mid/high frequency bands
    # Real images: energy concentrated in low frequencies
    # AI images: unusual energy in mid/high frequencies
    h, w = dct_abs.shape
    low  = dct_abs[:32,  :32].mean()
    mid  = dct_abs[32:96,  32:96].mean()
    high = dct_abs[96:,   96:].mean()

    # Ratio of high-to-low frequency energy
    # AI images often have higher ratio than real photos
    ratio = (mid + high) / (max(low, 1e-8))
    # Real photos: ratio ~0.05-0.20
    # AI images:   ratio ~0.25-0.60
    score = min(max((ratio - 0.05) / 0.55, 0), 1)
    return round(score, 3)

def analyse_colour_smoothness(img: np.ndarray) -> float:
    """
    Colour gradient smoothness.
    AI images render colour transitions too perfectly.
    Real photos have micro-variation from lighting & lens optics.
    Returns 0-1 where 1 = suspiciously smooth (AI-like).
    """
    # Compute local standard deviation in small windows
    local_stds = []
    step = 16
    for c in range(3):  # R, G, B channels
        channel = img[:, :, c].astype(float)
        for i in range(0, channel.shape[0]-step, step):
            for j in range(0, channel.shape[1]-step, step):
                patch = channel[i:i+step, j:j+step]
                local_stds.append(np.std(patch))

    if not local_stds: return 0.5

    avg_std = np.mean(local_stds)

    # AI images: avg_std < 8  → suspiciously smooth
    # Real photos: avg_std typically 10-30
    score = max(1.0 - (avg_std / 15.0), 0.0)
    return round(score, 3)

def analyse_edge_uniformity(img: np.ndarray) -> float:
    """
    Edge sharpness uniformity.
    AI images tend to have uniformly sharp edges everywhere.
    Real photos have depth-of-field effects causing blur variation.
    Returns 0-1 where 1 = unnaturally uniform (AI-like).
    """
    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Split image into grid and measure edge density per region
    h, w     = edges.shape
    grid     = 4
    densities = []
    for i in range(grid):
        for j in range(grid):
            patch = edges[i*h//grid:(i+1)*h//grid,
                         j*w//grid:(j+1)*w//grid]
            densities.append(patch.mean())

    if not densities: return 0.5

    # Low coefficient of variation = all regions equally sharp = AI
    # High CoV = varied sharpness = real photo
    mean = np.mean(densities)
    std  = np.std(densities)
    cov  = std / max(mean, 1e-8)

    # AI CoV ~0.0-0.3, Real CoV ~0.4-1.5
    score = max(1.0 - (cov / 0.8), 0.0)
    return round(score, 3)

def analyse_metadata(source) -> dict:
    """
    Check EXIF metadata.
    Real photos have camera metadata (make, model, GPS, lens).
    AI images are usually metadata-empty.
    Returns dict with findings.
    """
    try:
        if isinstance(source, (bytes, io.BytesIO)):
            if isinstance(source, bytes):
                source = io.BytesIO(source)
            img = Image.open(source)
        elif isinstance(source, np.ndarray):
            return {'has_exif': False, 'camera': None, 'software': None}
        else:
            img = Image.open(source)

        exif = img.getexif()
        if not exif:
            return {'has_exif': False, 'camera': None, 'software': None}

        # EXIF tag IDs
        camera   = exif.get(272)   # Model
        make     = exif.get(271)   # Make
        software = exif.get(305)   # Software

        return {
            'has_exif': True,
            'camera'  : f"{make} {camera}".strip() if (make or camera) else None,
            'software': software
        }
    except:
        return {'has_exif': False, 'camera': None, 'software': None}

def detect_ai_image(source) -> dict:
    """
    Main function. Pass a file path, PIL Image, or bytes.
    Returns full analysis dict with verdict and signal scores.
    """
    img  = load_image(source)
    meta = analyse_metadata(source)

    # ── Run all 4 pixel signals ──────────────────────
    noise_score   = analyse_noise(img)
    dct_score     = analyse_dct_frequency(img)
    colour_score  = analyse_colour_smoothness(img)
    edge_score    = analyse_edge_uniformity(img)

    # ── Metadata bonus ───────────────────────────────
    # No EXIF → slightly more likely AI
    # Has camera model → slightly more likely real
    meta_ai_bias = 0.0
    if not meta['has_exif']:
        meta_ai_bias = 0.15
    elif meta['camera']:
        meta_ai_bias = -0.10  # pull toward real

    # ── Weighted score ───────────────────────────────
    ai_score = (
        noise_score  * 0.30 +
        dct_score    * 0.30 +
        colour_score * 0.25 +
        edge_score   * 0.15
    )
    ai_score = min(max(ai_score + meta_ai_bias, 0), 1)
    ai_pct   = round(ai_score * 100, 1)

    # ── Verdict ──────────────────────────────────────
    if   ai_pct >= 75: verdict, note = "AI GENERATED",  "Strong pixel-level signals of AI generation."
    elif ai_pct >= 58: verdict, note = "LIKELY AI",     "Several AI-like visual patterns detected."
    elif ai_pct >= 42: verdict, note = "UNCERTAIN",     "Mixed signals — could be either."
    elif ai_pct >= 25: verdict, note = "LIKELY REAL",   "Leans toward a real photograph."
    else:              verdict, note = "REAL PHOTO",    "Strong signals of a real camera photo."

    # ── Image stats ──────────────────────────────────
    h, w = img.shape[:2]

    return {
        'verdict'    : verdict,
        'note'       : note,
        'ai_score'   : ai_pct,
        'signals': {
            'noise_pattern'    : round(noise_score  * 100, 1),
            'dct_frequency'    : round(dct_score    * 100, 1),
            'colour_smoothness': round(colour_score * 100, 1),
            'edge_uniformity'  : round(edge_score   * 100, 1),
        },
        'metadata'   : meta,
        'image_size' : (w, h),
        'megapixels' : round((w * h) / 1_000_000, 2)
    }

# Quick test
if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python image_detector.py path/to/image.jpg")
    else:
        r = detect_ai_image(path)
        print(f"\nVerdict    : {r['verdict']}")
        print(f"AI score   : {r['ai_score']}%")
        print(f"Size       : {r['image_size'][0]}x{r['image_size'][1]}")
        print(f"Megapixels : {r['megapixels']} MP")
        print(f"\nSignals:")
        for k, v in r['signals'].items():
            bar = '█' * int(v / 5)
            print(f"  {k:22s} {bar:20s} {v}%")
        print(f"\nMetadata   : {r['metadata']}")
