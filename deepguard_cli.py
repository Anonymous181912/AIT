import argparse
import cv2
import numpy as np
import base64
import webbrowser
import os
import time
import hashlib
import sys
from datetime import datetime
from PIL import Image, ImageChops
from config import MODEL_DIR

# --- DEPENDENCY CHECK ---
try:
    import mediapipe as mp
    LEVEL9_AVAILABLE = True
except ImportError:
    LEVEL9_AVAILABLE = False
    print("[WARNING] MediaPipe not installed. Level 9 Geometry Check will be skipped.")

try:
    from metadata_inspector import MetadataInspector
    METADATA_AVAILABLE = True
except ImportError:
    METADATA_AVAILABLE = False

# --- CONFIGURATION ---
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose Tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left Eye
    (225.0, 170.0, -135.0),      # Right Eye
    (-150.0, -150.0, -125.0),    # Left Mouth
    (150.0, -150.0, -125.0)      # Right Mouth
], dtype=np.double)

# --- ENGINE 1: PIXEL FORENSICS (Levels 5 & 8) ---
def auto_amplify(image):
    img_float = image.astype(np.float32)
    min_val, max_val = np.min(img_float), np.max(img_float)
    if max_val - min_val == 0: return image
    return (((img_float - min_val) / (max_val - min_val)) * 255).astype(np.uint8)

def get_ela(image_path):
    """Level 5: Compression Analysis"""
    original = Image.open(image_path).convert('RGB')
    temp = "temp_ela_master.jpg"
    original.save(temp, 'JPEG', quality=75)
    resaved = Image.open(temp)
    ela = ImageChops.difference(original, resaved)
    ela_map = cv2.applyColorMap(auto_amplify(cv2.cvtColor(np.array(ela), cv2.COLOR_RGB2GRAY)), cv2.COLORMAP_JET)
    if os.path.exists(temp): os.remove(temp)
    return ela_map

def get_noise(image):
    """Level 5: Sensor Noise Analysis"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    return auto_amplify(cv2.absdiff(gray, denoised))

def get_fft(image):
    """Level 8: Frequency Domain Analysis"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)
    norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_INFERNO)

# --- ENGINE 2: GEOMETRY (Level 9 - In-Built) ---
def get_geometry_analysis(image):
    if not LEVEL9_AVAILABLE: return "MODULE_MISSING", 0.0, image
    
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks: return "NO FACE FOUND", 0.0, image
        
        landmarks = results.multi_face_landmarks[0]
        h, w, _ = image.shape
        def to_pixel(idx): return (float(landmarks.landmark[idx].x * w), float(landmarks.landmark[idx].y * h))
        
        image_points = np.array([
            to_pixel(4), to_pixel(152), to_pixel(33), to_pixel(263), to_pixel(61), to_pixel(291)
        ], dtype=np.double)
        
        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.double)
        dist_coeffs = np.zeros((4,1))
        
        success, rot_vec, trans_vec = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        
        if not success: return "MATH ERROR", 0.0, image
        
        proj_points, _ = cv2.projectPoints(MODEL_POINTS, rot_vec, trans_vec, camera_matrix, dist_coeffs)
        error = np.mean([np.linalg.norm(image_points[i] - proj_points[i].ravel()) for i in range(len(image_points))])
        
        img_diag = np.sqrt(h**2 + w**2)
        norm_error = (error / img_diag) * 1000
        verdict = "CONSISTENT" if norm_error < 45 else "DISTORTED (AI)"
        
        # Draw Visuals
        vis = image.copy()
        axis = np.float32([[500,0,0], [0,500,0], [0,0,500]]).reshape(-1,3)
        imgpts, _ = cv2.projectPoints(axis, rot_vec, trans_vec, camera_matrix, dist_coeffs)
        nose = tuple(image_points[0].astype(int))
        cv2.line(vis, nose, tuple(imgpts[2].ravel().astype(int)), (255,0,0), 4)
        cv2.line(vis, nose, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 4)
        cv2.line(vis, nose, tuple(imgpts[0].ravel().astype(int)), (0,0,255), 4)
        
        return verdict, norm_error, vis

# --- ENGINE 3: SOURCE INTELLIGENCE (The WhatsApp Check) ---
def analyze_source_origin(image_path, img_array):
    """Determines if image is Camera, WhatsApp, or AI"""
    
    # 1. Metadata Check
    has_metadata = False
    if METADATA_AVAILABLE:
        inspector = MetadataInspector()
        sig = inspector.extract_metadata_signature(image_path)
        if np.mean(sig) > 0.4: has_metadata = True
        
    # 2. Noise Check
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    noise_diff = cv2.absdiff(gray, cv2.fastNlMeansDenoising(gray, None, 10, 7, 21))
    noise_score = np.mean(noise_diff)
    has_high_noise = noise_score > 1.8 # Threshold for "Real Grain"

    # 3. Logic Matrix
    if has_metadata and has_high_noise:
        return "ORIGINAL CAMERA FILE", "Likely a raw upload from iPhone/Android.", "#16a34a" # Green
    elif not has_metadata and has_high_noise:
        return "SOCIAL MEDIA (REAL)", "Metadata stripped by WhatsApp/FB, but sensor noise remains.", "#eab308" # Yellow
    elif not has_metadata and not has_high_noise:
        return "SYNTHETIC / AI GENERATED", "No Metadata AND No Sensor Noise. 'Too Perfect'.", "#dc2626" # Red
    else:
        return "EDITED / FILTERED", "Metadata exists but texture is unnaturally smooth.", "#f97316" # Orange

# --- REPORT GENERATOR ---
def img_to_base64(img_np):
    _, buffer = cv2.imencode('.png', img_np)
    return base64.b64encode(buffer).decode('utf-8')

def generate_report(image_path):
    print(f"\n[DEEPGUARD] Initializing Full System Scan for: {os.path.basename(image_path)}")
    
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Could not read image.")
        return

    # 2. AI Model Score (Level 1)
    print(" >> [1/4] Running Neural Network Analysis...")
    try:
        pipeline = DeepfakeDetectionPipeline(model_path=os.path.join(MODEL_DIR, "best_model.pth"))
        pred = pipeline.predict(image_path)
        ai_prob = pred['ai_generated_probability']
    except:
        ai_prob = 0.5 # Fallback
        
    # 3. Forensics (Level 5 & 8)
    print(" >> [2/4] Extracting Sensor Noise & FFT Spectrum...")
    noise_map = get_noise(img)
    ela_map = get_ela(image_path)
    fft_map = get_fft(img)
    
    # 4. Geometry (Level 9)
    print(" >> [3/4] Calculating 3D Perspective Geometry...")
    geo_verdict, geo_error, geo_vis = get_geometry_analysis(img)
    
    # 5. Source Intelligence (The "WhatsApp" Check)
    print(" >> [4/4] determining Source Origin...")
    source_type, source_desc, source_color = analyze_source_origin(image_path, img)

    # 6. Generate HTML
    case_id = f"CF-{int(time.time())}"
    is_fake = ai_prob > 0.5
    final_verdict = "SYNTHETIC MEDIA" if is_fake else "AUTHENTIC CAPTURE"
    main_color = "#dc2626" if is_fake else "#16a34a"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>DeepGuard Case {case_id}</title>
        <style>
            body {{ background: #f0f2f5; font-family: 'Courier New', monospace; padding: 40px; color: #333; }}
            .container {{ max_width: 1000px; margin: 0 auto; background: white; padding: 50px; box-shadow: 0 0 20px rgba(0,0,0,0.1); position: relative; }}
            .stamp {{ position: absolute; top: 40px; right: 40px; border: 4px solid {main_color}; color: {main_color}; font-size: 24px; font-weight: bold; padding: 10px 20px; transform: rotate(-10deg); text-transform: uppercase; }}
            
            .header-box {{ background: #f8f9fa; border: 1px solid #ddd; padding: 20px; margin-bottom: 30px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            .source-box {{ background: {source_color}22; border: 2px solid {source_color}; padding: 15px; margin-bottom: 30px; border-radius: 5px; }}
            
            .section-title {{ background: #222; color: white; padding: 8px 15px; font-weight: bold; margin-top: 40px; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 1px; }}
            
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .card {{ border: 1px solid #ddd; padding: 10px; background: #fff; }}
            .card img {{ width: 100%; border: 1px solid #eee; display: block; }}
            .caption {{ font-size: 11px; margin-top: 8px; color: #666; line-height: 1.4; border-top: 1px solid #eee; padding-top: 8px; }}
            
            .metric-bar {{ height: 15px; background: #eee; border: 1px solid #ccc; margin-top: 5px; }}
            .metric-fill {{ height: 100%; background: {main_color}; width: {ai_prob*100}%; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="stamp">{final_verdict}</div>
            
            <h1 style="border-bottom: 2px solid #333; padding-bottom: 10px;">CONFIDENTIAL FORENSIC REPORT</h1>
            
            <div class="source-box">
                <div style="color: {source_color}; font-weight: bold; font-size: 18px; margin-bottom: 5px;">SOURCE ORIGIN: {source_type}</div>
                <div style="font-size: 14px;">{source_desc}</div>
            </div>

            <div class="header-box">
                <div>
                    <b>CASE ID:</b> {case_id}<br>
                    <b>EVIDENCE FILE:</b> {os.path.basename(image_path)}<br>
                    <b>TIMESTAMP:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                </div>
                <div>
                    <b>AI PROBABILITY:</b> {ai_prob*100:.2f}%
                    <div class="metric-bar"><div class="metric-fill"></div></div>
                </div>
            </div>

            <div class="section-title">SECTION I: 3D GEOMETRY ANALYSIS</div>
            <div class="grid" style="grid-template-columns: 1fr 1fr;">
                <div class="card">
                    <b>ORIGINAL CAPTURE</b>
                    <img src="data:image/png;base64,{img_to_base64(img)}">
                </div>
                <div class="card">
                    <b>3D PERSPECTIVE MESH</b>
                    <img src="data:image/png;base64,{img_to_base64(geo_vis)}">
                    <div class="caption">
                        <b>STATUS: {geo_verdict}</b><br>
                        Reprojection Error: {geo_error:.2f}<br>
                        (High error > 45.0 indicates the face is physically impossible in 3D space)
                    </div>
                </div>
            </div>

            <div class="section-title">SECTION II: SIGNAL PROCESSING (THE "FINGERPRINTS")</div>
            <div class="grid" style="grid-template-columns: 1fr 1fr 1fr;">
                <div class="card">
                    <b>SENSOR NOISE (PRNU)</b>
                    <img src="data:image/png;base64,{img_to_base64(noise_map)}">
                    <div class="caption">
                        • Chaos/Static = Real Camera Sensor<br>
                        • Smooth/Ghosting = Synthetic
                    </div>
                </div>
                <div class="card">
                    <b>FREQUENCY (FFT)</b>
                    <img src="data:image/png;base64,{img_to_base64(fft_map)}">
                    <div class="caption">
                        • Starburst = Natural Light<br>
                        • Grid/Cross = AI Upscaler Artifacts
                    </div>
                </div>
                <div class="card">
                    <b>COMPRESSION (ELA)</b>
                    <img src="data:image/png;base64,{img_to_base64(ela_map)}">
                    <div class="caption">
                        • Rainbow = Natural Save<br>
                        • Flat Color = AI Generation
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 50px; text-align: center; color: #888; font-size: 10px;">
                DEEPGUARD SYSTEMS // AUTOMATED FORENSICS ENGINE
            </div>
        </div>
    </body>
    </html>
    """
    
    report_name = f"REPORT_{case_id}.html"
    with open(report_name, "w") as f: f.write(html)
    print(f"\n[SUCCESS] Report Generated: {report_name}")
    webbrowser.open(f"file://{os.path.realpath(report_name)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    generate_report(args.image)