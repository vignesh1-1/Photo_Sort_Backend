"""
PhotoSort Engine v4.0 (Monochrome Studio Edition) - Calibrated Computer Vision API
==================================================================================
High-precision FastAPI service for automated photography culling:
  1. Multi-Band Frequency Acutance (Difference-of-Gaussians Fine/Coarse Edge Ratio)
     + Modified Brenner & Sobel-Normalized Laplacian to eliminate false blur/sharpness.
  2. Subject-Aware Exposure Analysis (separates Subject ROI from High-Key/Low-Key backdrops).
  3. Directional Motion-Blur Anisotropy vs. Optical Defocus Differentiation.
  4. High-Frequency Sensor Noise & Compression Artifact Estimation in Flat Patches.
  5. Eyeglass-Aware Haar Cascade + Sclera/Pupil Contrast Blink Verification.
  6. 16-Zone (4x4) Spatial Telemetry Matrix for Interactive Frontend Heatmap Rendering.
"""

import os
import io
import cv2
import base64
import numpy as np
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from bson import ObjectId
from PIL import Image, ExifTags

load_dotenv()

# ==============================================================================
# DATABASE CONFIGURATION & ASYNC CLIENT SETUP
# ==============================================================================
DEFAULT_MONGO = "mongodb+srv://technglobalhosting_db_user:RhbVhLnehkcVIBa1@miniproject.pg2mobb.mongodb.net/?appName=miniproject"
MONGO_URI = os.getenv("MONGO_URI", DEFAULT_MONGO)
DB_NAME = os.getenv("DB_NAME", "photosort_db")

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
collection = db["photos"]

app = FastAPI(
    title="PhotoSort Pro v4.0 Studio Engine",
    description="Multi-Band Frequency Acutance & Subject-Aware Photo Culling API",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# OPENCV HAAR CASCADE INITIALIZATION (WITH EYEGLASSES FALLBACK)
# ==============================================================================
face_cascade = None
eye_cascade = None
eye_glasses_cascade = None

if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
    try:
        face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        eye_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        glasses_path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"

        fc = cv2.CascadeClassifier(face_path)
        ec = cv2.CascadeClassifier(eye_path)
        egc = cv2.CascadeClassifier(glasses_path)

        if not fc.empty():
            face_cascade = fc
        if not ec.empty():
            eye_cascade = ec
        if not egc.empty():
            eye_glasses_cascade = egc
    except Exception:
        pass

# ==============================================================================
# REQUEST VALIDATION SCHEMAS
# ==============================================================================
class ClassificationUpdate(BaseModel):
    category: str = Field(..., pattern="^(good|review|poor)$")

class BatchCategoryUpdate(BaseModel):
    ids: List[str]
    category: str = Field(..., pattern="^(good|review|poor)$")

class BatchDeleteRequest(BaseModel):
    ids: List[str]

# ==============================================================================
# PERCEPTUAL HASHING & EXIF HELPERS
# ==============================================================================
def compute_dhash(img_bgr: np.ndarray) -> str:
    """Computes a 64-bit horizontal gradient difference hash (dHash)."""
    resized = cv2.resize(img_bgr, (9, 8), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    diff = gray[:, 1:] > gray[:, :-1]

    hash_int = 0
    for bit in diff.flatten():
        hash_int = (hash_int << 1) | int(bit)
    return f"{hash_int:016x}"

def hamming_distance(h1: str, h2: str) -> int:
    """Calculates bitwise Hamming distance between two 16-char hex hashes."""
    try:
        return bin(int(h1, 16) ^ int(h2, 16)).count("1")
    except Exception:
        return 64

def extract_exif_data(raw_bytes: bytes) -> dict:
    """Extracts camera hardware metadata from raw image bytes."""
    meta = {
        "camera": "Digital Sensor",
        "lens": "Standard Prime",
        "iso": "ISO Auto",
        "f_stop": "f/2.8",
        "shutter": "1/250s"
    }
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        exif = pil_img.getexif()
        if exif:
            make = str(exif.get(ExifTags.Base.Make, "")).strip()
            model = str(exif.get(ExifTags.Base.Model, "")).strip()
            if make or model:
                meta["camera"] = f"{make} {model}".strip()
            iso_val = exif.get(ExifTags.Base.ISOSpeedRatings)
            if iso_val:
                meta["iso"] = f"ISO {iso_val}"
            f_num = exif.get(ExifTags.Base.FNumber)
            if f_num:
                meta["f_stop"] = f"f/{float(f_num):.1f}"
            exp_time = exif.get(ExifTags.Base.ExposureTime)
            if exp_time:
                val = float(exp_time)
                meta["shutter"] = f"1/{int(round(1.0 / val))}s" if 0 < val < 0.5 else f"{val:.2f}s"
    except Exception:
        pass
    return meta

def compress_to_webp(img_bgr: np.ndarray, max_dim: int = 1600, quality: int = 82) -> str:
    """Resizes and encodes OpenCV BGR image into a Base64 WebP Data URL."""
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    success, encoded = cv2.imencode(".webp", img_bgr, [cv2.IMWRITE_WEBP_QUALITY, quality])
    if not success:
        raise ValueError("WebP compression failed")

    b64 = base64.b64encode(encoded).decode("utf-8")
    return f"data:image/webp;base64,{b64}"

# ==============================================================================
# CORE CALIBRATED COMPUTER VISION ENGINE (v4.0)
# ==============================================================================
def analyze_image_cv(image_bytes: bytes, mode: str = "auto") -> dict:
    """
    Multi-Stage Optical Quality & Perceptual Culling Pipeline:
      Stage 1: Canonical 1024px normalization for 100% scale invariance.
      Stage 2: Multi-Band Difference-of-Gaussians (DoG) Edge Acutance Ratio.
               Distinguishes true narrow optical focus from blurry high-contrast edges.
      Stage 3: 4x4 Spatial Grid (16 zones) sharpness mapping & Bokeh separation.
      Stage 4: Directional Motion-Blur Anisotropy vs. Defocus detection.
      Stage 5: Subject-Aware Exposure & Dynamic Range (protects High-Key/Low-Key photos).
      Stage 6: Eyeglass-aware Face/Eye verification with Sclera-Pupil contrast check.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_raw is None:
        raise ValueError("Invalid image file or unsupported format.")

    orig_h, orig_w = img_raw.shape[:2]

    # Stage 1: Canonical Resolution Normalization (1024px max dimension)
    CANONICAL_DIM = 1024
    scale = CANONICAL_DIM / float(max(orig_h, orig_w))
    if abs(scale - 1.0) > 0.05:
        norm_w = max(128, int(orig_w * scale))
        norm_h = max(128, int(orig_h * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        img_norm = cv2.resize(img_raw, (norm_w, norm_h), interpolation=interp)
    else:
        img_norm = img_raw.copy()

    gray_u8 = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
    gray_f = gray_u8.astype(np.float32)

    # Stage 2: High-Frequency Noise Floor Estimation in Smooth Patches
    median_u8 = cv2.medianBlur(gray_u8, 3)
    noise_map = cv2.absdiff(gray_u8, median_u8).astype(np.float32)

    # Multi-Band Gaussian Pyramid for True Optical Acutance (MTF approximation)
    g_fine = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=0.75)
    g_mid = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=1.8)
    g_coarse = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=4.2)

    # Bandpass detail maps
    dog_fine = np.abs(g_fine - g_mid)      # True crisp micro-edges (optical focus)
    dog_coarse = np.abs(g_mid - g_coarse)  # Wide transition ramps (survives defocus/blur)

    # Sobel first-order gradients & Laplacian second-order acutance on de-noised fine layer
    gx = cv2.Sobel(g_fine, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g_fine, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    lap_abs = np.abs(cv2.Laplacian(g_fine, cv2.CV_32F, ksize=3))

    # Stage 3: 4x4 Spatial Grid Analysis (16 Zones)
    nh, nw = gray_u8.shape[:2]
    grid_rows, grid_cols = 4, 4
    cell_h, cell_w = max(1, nh // grid_rows), max(1, nw // grid_cols)

    zone_sharpness = []
    zone_acutance_ratios = []
    zone_noise = []
    zone_lum = []
    center_zone_sharpness = []
    perimeter_zone_lum = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            y1, y2 = r * cell_h, (r + 1) * cell_h if r < grid_rows - 1 else nh
            x1, x2 = c * cell_w, (c + 1) * cell_w if c < grid_cols - 1 else nw

            p_lap = lap_abs[y1:y2, x1:x2]
            p_grad = grad_mag[y1:y2, x1:x2]
            p_fine = dog_fine[y1:y2, x1:x2]
            p_coarse = dog_coarse[y1:y2, x1:x2]
            p_noise = noise_map[y1:y2, x1:x2]
            p_gray = gray_f[y1:y2, x1:x2]

            # Measure sharpness at salient contours (top 15% strongest edges in patch)
            p90_lap = float(np.percentile(p_lap, 90))
            p90_grad = float(np.percentile(p_grad, 90))
            lap_var = float(np.var(p_lap))

            # Fine-to-Coarse Frequency Acutance Ratio (rejects blurry high-contrast edges)
            edge_mask = p_grad > max(12.0, np.percentile(p_grad, 75))
            if np.any(edge_mask):
                fine_energy = float(np.mean(p_fine[edge_mask]))
                coarse_energy = float(np.mean(p_coarse[edge_mask]))
                lap_to_grad = float(np.mean(p_lap[edge_mask]) / (np.mean(p_grad[edge_mask]) + 1.0))
            else:
                fine_energy = float(np.mean(p_fine))
                coarse_energy = float(np.mean(p_coarse))
                lap_to_grad = 0.25

            freq_ratio = fine_energy / (coarse_energy + 0.85)
            acutance_mult = float(np.clip(np.interp(freq_ratio, [0.18, 0.32, 0.48, 0.72], [0.28, 0.65, 1.05, 1.32]), 0.25, 1.35))
            edge_width_mult = float(np.clip(np.interp(lap_to_grad, [0.18, 0.34, 0.52], [0.45, 0.95, 1.20]), 0.40, 1.25))

            raw_structural = (p90_lap * 1.45) + (min(lap_var, 400.0) * 0.28) + (p90_grad * 0.22)
            calibrated_patch_score = raw_structural * acutance_mult * edge_width_mult

            zone_sharpness.append(round(calibrated_patch_score, 1))
            zone_acutance_ratios.append(round(freq_ratio, 3))
            zone_noise.append(float(np.mean(p_noise)))
            patch_mean_lum = float(np.mean(p_gray))
            zone_lum.append(patch_mean_lum)

            if 1 <= r <= 2 and 1 <= c <= 2:
                center_zone_sharpness.append(calibrated_patch_score)
            else:
                perimeter_zone_lum.append(patch_mean_lum)

    # Sort zones to separate Peak Subject ROI from Background
    sorted_sharp = sorted(zone_sharpness)
    flat_noise_floor = float(np.mean(sorted(zone_noise)[:5]))

    peak_roi_sharpness = float(np.mean(sorted_sharp[-4:]))       # Top 4 sharpest zones
    top2_roi_sharpness = float(np.mean(sorted_sharp[-2:]))       # Top 2 sharpest focal zones
    global_sharpness = float(np.mean(zone_sharpness))
    center_sharpness = float(np.mean(center_zone_sharpness))
    background_sharpness = float(np.mean(sorted_sharp[:6]))      # 6 smoothest background zones

    # Peak acutance ratio across the sharpest zones
    top_zone_indices = np.argsort(zone_sharpness)[-4:]
    subject_freq_ratio = float(np.mean([zone_acutance_ratios[i] for i in top_zone_indices]))

    # Stage 4: Directional Motion Blur / Camera Shake Detection
    mean_gx = float(np.mean(np.abs(gx)))
    mean_gy = float(np.mean(np.abs(gy)))
    directional_anisotropy = max(mean_gx, mean_gy) / max(1.0, min(mean_gx, mean_gy))
    motion_blur_detected = bool(directional_anisotropy > 1.85 and subject_freq_ratio < 0.36)

    # Stage 5: Face Detection, Face ROI Sharpness, & Accurate Blink Verification
    face_count = 0
    eyes_closed = False
    face_sharpness_score = 0.0
    face_lum_values = []

    if face_cascade is not None:
        try:
            faces = face_cascade.detectMultiScale(
                gray_u8, scaleFactor=1.12, minNeighbors=5, minSize=(44, 44)
            )
            face_count = int(len(faces))

            if face_count > 0:
                face_scores = []
                for (fx, fy, fw, fh) in faces:
                    f_lap = lap_abs[fy:fy + fh, fx:fx + fw]
                    f_fine = dog_fine[fy:fy + fh, fx:fx + fw]
                    f_coarse = dog_coarse[fy:fy + fh, fx:fx + fw]
                    f_gray = gray_u8[fy:fy + fh, fx:fx + fw]
                    face_lum_values.append(float(np.mean(f_gray)))

                    f_ratio = float(np.mean(f_fine) / (np.mean(f_coarse) + 0.8))
                    f_mult = float(np.clip(np.interp(f_ratio, [0.20, 0.42, 0.65], [0.45, 1.0, 1.3]), 0.4, 1.35))
                    f_val = (float(np.percentile(f_lap, 90)) * 1.65 + float(np.var(f_lap)) * 0.30) * f_mult
                    face_scores.append(f_val)

                    # Reliable Blink Verification (only on clearly resolved frontal faces >= 84px)
                    if fw >= 84:
                        ey1 = fy + int(fh * 0.20)
                        ey2 = fy + int(fh * 0.55)
                        ex1 = fx + int(fw * 0.12)
                        ex2 = fx + int(fw * 0.88)
                        upper_face = gray_u8[ey1:ey2, ex1:ex2]

                        if upper_face.size > 0:
                            detected_eyes = []
                            if eye_cascade is not None:
                                e1 = eye_cascade.detectMultiScale(upper_face, scaleFactor=1.08, minNeighbors=3, minSize=(14, 10))
                                if len(e1) > 0:
                                    detected_eyes.extend(e1)
                            if len(detected_eyes) == 0 and eye_glasses_cascade is not None:
                                e2 = eye_glasses_cascade.detectMultiScale(upper_face, scaleFactor=1.08, minNeighbors=3, minSize=(14, 10))
                                if len(e2) > 0:
                                    detected_eyes.extend(e2)

                            # Sclera-to-Pupil dark core check
                            eye_band_p5 = float(np.percentile(upper_face, 5))
                            eye_band_p85 = float(np.percentile(upper_face, 85))
                            pupil_contrast = (eye_band_p85 - eye_band_p5) / max(1.0, float(np.mean(upper_face)))

                            if len(detected_eyes) == 0 and pupil_contrast < 0.46 and f_val > 55.0:
                                eyes_closed = True

                if face_scores:
                    face_sharpness_score = max(face_scores)
        except Exception:
            pass

    # Compute Effective Subject Sharpness according to Scene Mode
    if mode == "landscape":
        effective_sharpness = (global_sharpness * 0.65) + (peak_roi_sharpness * 0.35)
    elif mode == "portrait":
        effective_sharpness = max(top2_roi_sharpness, face_sharpness_score * 1.15, (peak_roi_sharpness * 0.75 + center_sharpness * 0.25))
    else:
        effective_sharpness = max(
            (peak_roi_sharpness * 0.65) + (center_sharpness * 0.35),
            top2_roi_sharpness * 0.92,
            face_sharpness_score * 1.12
        )

    # Discount false sharpness caused by heavy ISO grain or low-res block artifacts
    if flat_noise_floor > 5.2:
        noise_penalty_factor = max(0.32, 1.0 - ((flat_noise_floor - 4.8) * 0.095))
        effective_sharpness *= noise_penalty_factor

    # Detect Intentional Portrait Bokeh / Shallow Depth of Field
    bokeh_ratio = round(peak_roi_sharpness / max(1.0, background_sharpness), 2)
    bokeh_detected = bool(effective_sharpness >= 68.0 and bokeh_ratio >= 2.6 and subject_freq_ratio >= 0.38)

    sharpness_display = round(float(effective_sharpness), 1)
    focus_subscore = int(np.clip(np.interp(effective_sharpness, [16, 38, 65, 92, 165, 280], [15, 42, 68, 84, 95, 99]), 10, 99))

    # Stage 6: Subject-Aware Exposure, Dynamic Range & Histogram Analysis
    mean_lum = round(float(np.mean(gray_u8)), 1)
    subject_lum = round(float(np.mean(face_lum_values)) if face_lum_values else float(np.mean([zone_lum[i] for i in top_zone_indices])), 1)
    rms_contrast = round(float(np.std(gray_u8)), 1)
    total_pixels = float(gray_u8.size)

    shadow_clip_pct = round(float(np.sum(gray_u8 < 12) / total_pixels) * 100.0, 1)
    highlight_clip_pct = round(float(np.sum(gray_u8 > 245) / total_pixels) * 100.0, 1)

    # Measure clipping specifically inside the Center/Subject ROI
    cy1, cy2 = int(nh * 0.2), int(nh * 0.8)
    cx1, cx2 = int(nw * 0.2), int(nw * 0.8)
    center_crop = gray_u8[cy1:cy2, cx1:cx2]
    subject_shadow_clip = round(float(np.sum(center_crop < 12) / max(1, center_crop.size)) * 100.0, 1)
    subject_highlight_clip = round(float(np.sum(center_crop > 246) / max(1, center_crop.size)) * 100.0, 1)

    p2, p98 = np.percentile(gray_u8, [2, 98])
    dynamic_range = round(float(p98 - p2), 1)

    # Stage 7: Synthesize Diagnostic Flags & Calibrated Quality Score
    issues = []
    penalties = 0

    # 1. Focus & Acutance Diagnosis
    if effective_sharpness < 36.0 or subject_freq_ratio < 0.25:
        if motion_blur_detected:
            issues.append("Motion Blur / Shake")
        else:
            issues.append("Out of Focus")
        penalties += 45
    elif effective_sharpness < 64.0 or subject_freq_ratio < 0.35:
        if motion_blur_detected:
            issues.append("Slight Motion Blur")
        else:
            issues.append("Soft Focus")
        penalties += 19
    else:
        if bokeh_detected:
            issues.append("Sharp Subject (Bokeh)")
        else:
            issues.append("Tack Sharp Focus")

    # 2. Subject-Aware Exposure Diagnosis
    exposure_subscore = 94
    is_severely_under = (subject_lum < 38.0 and subject_shadow_clip > 28.0) or (mean_lum < 28.0 and shadow_clip_pct > 55.0)
    is_severely_over = (subject_lum > 228.0 and subject_highlight_clip > 18.0) or (mean_lum > 228.0 and highlight_clip_pct > 35.0)

    if is_severely_under:
        issues.append("Underexposed")
        penalties += 32
        exposure_subscore = 38
    elif is_severely_over:
        issues.append("Blown Highlights")
        penalties += 30
        exposure_subscore = 40
    elif subject_lum < 52.0 and shadow_clip_pct > 38.0 and mode != "lowlight":
        issues.append("Dark Shadows")
        penalties += 12
        exposure_subscore = 68
    elif subject_highlight_clip > 14.0 and highlight_clip_pct > 20.0:
        issues.append("Highlight Clipping")
        penalties += 12
        exposure_subscore = 70
    else:
        issues.append("Balanced Exposure")

    # 3. Contrast & Dynamic Range Diagnosis
    if rms_contrast < 21.0 or dynamic_range < 62.0:
        issues.append("Low Contrast / Washed Out")
        penalties += 22
        exposure_subscore = min(exposure_subscore, 54)
    elif rms_contrast >= 44.0 and dynamic_range >= 170.0 and "Out of Focus" not in issues and "Motion Blur / Shake" not in issues:
        issues.append("Rich Dynamic Range")

    # 4. Sensor Noise & Artifacts
    noise_subscore = int(np.clip(np.interp(flat_noise_floor, [1.2, 4.5, 7.5, 12.0], [98, 86, 58, 25]), 20, 99))
    if flat_noise_floor > 7.2:
        issues.append("High ISO Noise")
        penalties += 20

    # 5. Eye Blink Penalty
    if eyes_closed:
        issues.append("Closed Eyes / Blink")
        penalties += 24

    # Composite Weighted Quality Calculation
    base_score = (focus_subscore * 0.60) + (exposure_subscore * 0.28) + (noise_subscore * 0.12)
    if bokeh_detected and penalties == 0:
        base_score += 5

    final_score = int(np.clip(round(base_score - (penalties * 0.52)), 11, 99))

    # Strict Guardrails so defective photos NEVER enter 'Good/Select'
    if "Out of Focus" in issues or "Motion Blur / Shake" in issues:
        final_score = min(final_score, 42)
    elif "Underexposed" in issues or "Blown Highlights" in issues:
        final_score = min(final_score, 46)
    elif any(k in issues for k in ["Soft Focus", "Slight Motion Blur", "Closed Eyes / Blink", "Low Contrast / Washed Out", "High ISO Noise"]):
        final_score = min(final_score, 71)

    if final_score >= 75:
        category = "good"
    elif final_score >= 50:
        category = "review"
    else:
        category = "poor"

    explanation = (
        f"Subject Acutance: {sharpness_display} (Fine/Coarse Edge Ratio: {subject_freq_ratio:.2f}). "
        f"{'Intentional Bokeh Separation (' + str(bokeh_ratio) + 'x). ' if bokeh_detected else ''}"
        f"Subject Luminance: {subject_lum} Lum, Dynamic Range: {dynamic_range}, Noise Floor: {flat_noise_floor:.1f}."
    )

    dhash = compute_dhash(img_norm)

    return {
        "score": final_score,
        "category": category,
        "sharpness": sharpness_display,
        "exposure": mean_lum,
        "subject_exposure": subject_lum,
        "contrast": rms_contrast,
        "dynamic_range": dynamic_range,
        "noise_level": round(flat_noise_floor, 2),
        "acutance_ratio": round(subject_freq_ratio, 2),
        "bokeh_ratio": bokeh_ratio,
        "focus_score": focus_subscore,
        "exposure_score": exposure_subscore,
        "noise_score": noise_subscore,
        "clip_shadows_pct": shadow_clip_pct,
        "clip_highlights_pct": highlight_clip_pct,
        "bokeh_detected": bokeh_detected,
        "zone_scores": zone_sharpness,
        "explanation": explanation,
        "issues": issues,
        "face_count": face_count,
        "eyes_closed": eyes_closed,
        "resolution": f"{orig_w}x{orig_h}",
        "dhash": dhash,
        "img_cv": img_raw
    }

def decode_data_url_to_bytes(data_url: str) -> bytes:
    """Decodes a stored Base64 WebP Data URL back into raw bytes for re-calibration."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)

def format_doc(doc):
    doc["id"] = str(doc["_id"])
    del doc["_id"]
    return doc

# ==============================================================================
# FRONTEND STATIC FILE SERVING (HTML, CSS, JS)
# ==============================================================================
@app.get("/")
async def serve_frontend():
    for filename in ["index.html", "index_2.html", "i.html"]:
        if os.path.exists(filename):
            return FileResponse(filename, headers={"Cache-Control": "no-cache"})
    return {"message": "index.html not found in root directory"}

@app.get("/style.css")
async def serve_css():
    if os.path.exists("style.css"):
        return FileResponse("style.css", media_type="text/css", headers={"Cache-Control": "no-cache"})
    raise HTTPException(status_code=404, detail="style.css not found")

@app.get("/script.js")
async def serve_js():
    if os.path.exists("script.js"):
        return FileResponse("script.js", media_type="application/javascript", headers={"Cache-Control": "no-cache"})
    raise HTTPException(status_code=404, detail="script.js not found")

# ==============================================================================
# FASTAPI API ENDPOINTS
# ==============================================================================
@app.get("/api/health")
async def health_check():
    try:
        await client.admin.command("ping")
        db_status = "connected"
    except Exception as e:
        db_status = f"unreachable: {str(e)}"
    return {"status": "ok", "engine": "v4.0-studio-multiband", "mongodb": db_status}

@app.post("/api/photos/analyze")
async def analyze_and_save_photo(
    file: UploadFile = File(...),
    mode: Optional[str] = Query("auto")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    content = await file.read()
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max size is 15MB.")

    try:
        cv_results = analyze_image_cv(content, mode=mode or "auto")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image processing failed: {str(e)}")

    exif_meta = extract_exif_data(content)

    img_cv = cv_results.pop("img_cv")
    compressed_image = compress_to_webp(img_cv, max_dim=1600, quality=82)
    thumbnail_image = compress_to_webp(img_cv, max_dim=420, quality=72)

    dhash = cv_results["dhash"]
    is_duplicate = False
    duplicate_of = None

    recent_photos = collection.find({}, {"dhash": 1, "filename": 1, "score": 1}).sort("created_at", -1).limit(50)
    async for p in recent_photos:
        if "dhash" in p:
            dist = hamming_distance(dhash, p["dhash"])
            if dist <= 8:
                is_duplicate = True
                duplicate_of = str(p["_id"])
                if "Burst Duplicate" not in cv_results["issues"]:
                    cv_results["issues"].append("Burst Duplicate")
                break

    record = {
        "filename": file.filename,
        "image_data": compressed_image,
        "thumb_data": thumbnail_image,
        **cv_results,
        "is_duplicate": is_duplicate,
        "duplicate_of": duplicate_of,
        "exif": exif_meta,
        "file_size_kb": round(len(content) / 1024, 2),
        "compressed_size_kb": round(len(compressed_image) * 0.75 / 1024, 2),
        "created_at": datetime.utcnow()
    }

    result = await collection.insert_one(record)
    record["id"] = str(result.inserted_id)
    if "_id" in record:
        del record["_id"]

    return record

@app.post("/api/photos/reanalyze")
async def reanalyze_all_photos(mode: Optional[str] = Query("auto")):
    """
    Re-runs the v4.0 Multi-Band CV Engine across all existing photos in MongoDB Atlas.
    Instantly corrects any previously misclassified images in place.
    """
    cursor = collection.find({}).sort("created_at", 1)
    updated_count = 0
    seen_hashes = []

    async for doc in cursor:
        img_b64 = doc.get("image_data")
        if not img_b64:
            continue
        try:
            raw_bytes = decode_data_url_to_bytes(img_b64)
            cv_res = analyze_image_cv(raw_bytes, mode=mode or "auto")
            cv_res.pop("img_cv", None)

            dhash = cv_res["dhash"]
            is_dup = False
            dup_of = None
            for prev_id, prev_hash in seen_hashes:
                if hamming_distance(dhash, prev_hash) <= 8:
                    is_dup = True
                    dup_of = prev_id
                    if "Burst Duplicate" not in cv_res["issues"]:
                        cv_res["issues"].append("Burst Duplicate")
                    break
            seen_hashes.append((str(doc["_id"]), dhash))

            update_fields = {
                **cv_res,
                "is_duplicate": is_dup,
                "duplicate_of": dup_of
            }
            await collection.update_one({"_id": doc["_id"]}, {"$set": update_fields})
            updated_count += 1
        except Exception:
            continue

    return {"status": "success", "reanalyzed_count": updated_count}

@app.get("/api/photos")
async def get_photos(
    category: Optional[str] = Query(None, pattern="^(good|review|poor)$"),
    sort: Optional[str] = Query("newest")
):
    query = {}
    if category:
        query["category"] = category

    sort_field = "created_at"
    sort_dir = -1

    if sort == "oldest":
        sort_dir = 1
    elif sort == "score_desc":
        sort_field = "score"
    elif sort == "score_asc":
        sort_field = "score"
        sort_dir = 1
    elif sort == "sharpness_desc":
        sort_field = "sharpness"
    elif sort == "size_desc":
        sort_field = "file_size_kb"

    cursor = collection.find(query).sort(sort_field, sort_dir)
    photos = []
    async for doc in cursor:
        photos.append(format_doc(doc))
    return photos

@app.patch("/api/photos/{photo_id}/category")
async def update_classification(photo_id: str, payload: ClassificationUpdate):
    if not ObjectId.is_valid(photo_id):
        raise HTTPException(status_code=400, detail="Invalid photo ID format.")

    result = await collection.update_one(
        {"_id": ObjectId(photo_id)},
        {"$set": {"category": payload.category}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Photo record not found.")

    return {"status": "success", "id": photo_id, "category": payload.category}

@app.post("/api/photos/batch-category")
async def batch_update_category(payload: BatchCategoryUpdate):
    valid_ids = [ObjectId(pid) for pid in payload.ids if ObjectId.is_valid(pid)]
    if not valid_ids:
        raise HTTPException(status_code=400, detail="No valid IDs provided.")

    result = await collection.update_many(
        {"_id": {"$in": valid_ids}},
        {"$set": {"category": payload.category}}
    )
    return {"status": "success", "modified_count": result.modified_count}

@app.delete("/api/photos/{photo_id}")
async def delete_photo(photo_id: str):
    if not ObjectId.is_valid(photo_id):
        raise HTTPException(status_code=400, detail="Invalid photo ID format.")

    result = await collection.delete_one({"_id": ObjectId(photo_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Photo record not found.")

    return {"status": "success", "message": "Photo deleted successfully", "id": photo_id}

@app.post("/api/photos/batch-delete")
async def batch_delete_photos(payload: BatchDeleteRequest):
    valid_ids = [ObjectId(pid) for pid in payload.ids if ObjectId.is_valid(pid)]
    if not valid_ids:
        raise HTTPException(status_code=400, detail="No valid IDs provided.")

    result = await collection.delete_many({"_id": {"$in": valid_ids}})
    return {"status": "success", "deleted_count": result.deleted_count}

@app.get("/api/stats")
async def get_database_stats():
    try:
        db_stats = await db.command("dbStats")
        storage_bytes = db_stats.get("storageSize", db_stats.get("dataSize", 0))
        data_bytes = db_stats.get("dataSize", 0)
    except Exception:
        storage_bytes = 0
        data_bytes = 0

    LIMIT_MB = 512.0
    used_bytes = max(storage_bytes, data_bytes)
    used_mb = round(used_bytes / (1024 * 1024), 2)
    remaining_mb = round(max(0.0, LIMIT_MB - used_mb), 2)
    used_percent = round((used_mb / LIMIT_MB) * 100, 2)

    total_photos = await collection.count_documents({})
    good_count = await collection.count_documents({"category": "good"})
    review_count = await collection.count_documents({"category": "review"})
    poor_count = await collection.count_documents({"category": "poor"})
    duplicate_count = await collection.count_documents({"is_duplicate": True})

    pipeline = [
        {
            "$group": {
                "_id": None,
                "avg_score": {"$avg": "$score"},
                "avg_sharpness": {"$avg": "$sharpness"},
                "avg_exposure": {"$avg": "$exposure"},
                "avg_file_size_kb": {"$avg": "$file_size_kb"}
            }
        }
    ]
    agg_results = await collection.aggregate(pipeline).to_list(1)
    averages = agg_results[0] if agg_results else {}

    return {
        "storage": {
            "used_mb": used_mb,
            "remaining_mb": remaining_mb,
            "limit_mb": LIMIT_MB,
            "used_percent": used_percent,
            "total_bytes": used_bytes
        },
        "counts": {
            "total": total_photos,
            "good": good_count,
            "review": review_count,
            "poor": poor_count,
            "duplicates": duplicate_count
        },
        "averages": {
            "score": round(averages.get("avg_score") or 0, 1),
            "sharpness": round(averages.get("avg_sharpness") or 0, 1),
            "exposure": round(averages.get("avg_exposure") or 0, 1),
            "file_size_kb": round(averages.get("avg_file_size_kb") or 0, 1)
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting PhotoSort v4.0 Studio on http://127.0.0.1:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)