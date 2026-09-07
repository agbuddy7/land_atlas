import io
import json
import math
from typing import List, Optional
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize FastAPI App
app = FastAPI(title="GeoAdhikar SAM Plot Segmentation API", version="1.0.0")

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load SAM model (FastSAM / MobileSAM via Ultralytics)
sam_model = None

def get_model():
    global sam_model
    if sam_model is None:
        try:
            from ultralytics import FastSAM
            print("🚀 Loading FastSAM model for plot segmentation...")
            # FastSAM-s is lightweight (40MB), runs smoothly on CPU
            sam_model = FastSAM("FastSAM-s.pt")
            print("✅ FastSAM model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Failed to load FastSAM: {e}")
    return sam_model

# ─── Helper: Pixel to GPS Affine Transformation ────────────────────────────────
def pixel_to_gps(px, py, bbox, img_width=512, img_height=512):
    """
    bbox: [west (minLng), south (minLat), east (maxLng), north (maxLat)]
    Converts (px, py) in image space to (longitude, latitude).
    """
    min_lng, min_lat, max_lng, max_lat = bbox
    lng = min_lng + (px / img_width) * (max_lng - min_lng)
    # y is inverted in image coordinates (top is 0)
    lat = max_lat - (py / img_height) * (max_lat - min_lat)
    return [round(lng, 7), round(lat, 7)]

# ─── Helper: Calculate Geodesic Polygon Area (Ha & Acres) ─────────────────────
def calculate_polygon_area_sqm(coordinates, lat_ref):
    """
    Computes approximate metric area (in sq meters) for a geographic polygon
    at a given latitude using the Shoelace formula projected to meters.
    """
    # 1 degree of latitude approx 111,139 meters
    # 1 degree of longitude approx 111,139 * cos(lat) meters
    m_per_deg_lat = 111139.0
    m_per_deg_lng = 111139.0 * math.cos(math.radians(lat_ref))

    ring = coordinates[0]
    if len(ring) < 3:
        return 0.0

    # Project to metric meters relative to first vertex
    ref_lng, ref_lat = ring[0]
    pts = []
    for lng, lat in ring:
        x = (lng - ref_lng) * m_per_deg_lng
        y = (lat - ref_lat) * m_per_deg_lat
        pts.append((x, y))

    # Shoelace formula
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0

# ─── Helper: Classify Land Type from Spectral RGB ──────────────────────────────
def classify_plot_type(roi_rgb):
    """
    Classifies plot type using vegetation/soil spectral index in RGB.
    """
    if roi_rgb.size == 0:
        return "Unclassified", "#9ca3af"

    r = np.mean(roi_rgb[:, :, 0])
    g = np.mean(roi_rgb[:, :, 1])
    b = np.mean(roi_rgb[:, :, 2])

    # Visible Atmospherically Resistant Index (VARI) for greenness
    denom = (g + r - b)
    vari = (g - r) / denom if denom != 0 else 0.0

    # Brightness
    brightness = (r + g + b) / 3.0

    # Water: low red, higher blue/green, low overall brightness
    if b > r and b > 40 and brightness < 90:
        return "Water Body", "#38bdf8"

    # Forest / Dense Vegetation: high green, low red, moderate brightness
    if vari > 0.12 and g > r:
        return "Forest / Dense Canopy", "#059669"

    # Agriculture / Cropland: green dominant, moderate-high brightness
    if vari > 0.02 and g >= r:
        return "Agricultural / Cropland", "#10b981"

    # Built-up / Settlement: high brightness, gray/neutral or red roofs
    if brightness > 140 or (abs(r - g) < 15 and abs(g - b) < 15 and brightness > 110):
        return "Habitation / Settlement", "#ef4444"

    # Default to Fallow / Barren Soil
    return "Fallow / Barren Land", "#f59e0b"

# ─── Core Segmentation Function ───────────────────────────────────────────────
def segment_image_plots(image_bytes: bytes, bbox: List[float], cell_id: str = "A1"):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_img = np.array(image)
    h, w, _ = np_img.shape

    min_lng, min_lat, max_lng, max_lat = bbox
    center_lat = (min_lat + max_lat) / 2.0

    features = []
    model = get_model()

    if model is not None:
        try:
            # Run FastSAM inference
            results = model(
                np_img,
                device="cpu",
                retina_masks=True,
                imgsz=512,
                conf=0.25,
                iou=0.6
            )

            if results and len(results) > 0 and results[0].masks is not None:
                masks_data = results[0].masks.data.cpu().numpy() # shape (N, H, W)
                
                plot_idx = 1
                for i in range(len(masks_data)):
                    mask = (masks_data[i] * 255).astype(np.uint8)
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                    # Find external contours
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        # Filter out tiny noise (less than 60 pixels)
                        pixel_area = cv2.contourArea(cnt)
                        if pixel_area < 120:
                            continue

                        # Approximate polygon to reduce coordinates
                        epsilon = 0.008 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)

                        if len(approx) < 3:
                            continue

                        # Convert pixel points to GPS coordinates
                        gps_ring = []
                        for pt in approx:
                            px, py = pt[0]
                            gps_ring.append(pixel_to_gps(px, py, bbox, w, h))

                        # Close ring
                        if gps_ring[0] != gps_ring[-1]:
                            gps_ring.append(gps_ring[0])

                        # Calculate metric area
                        area_sqm = calculate_polygon_area_sqm([gps_ring], center_lat)
                        if area_sqm < 25.0: # ignore fragments smaller than 25 sq meters
                            continue

                        # Mask ROI for classification
                        mask_bool = mask > 0
                        roi = np_img[mask_bool]
                        plot_type, color = classify_plot_type(roi)

                        plot_id = f"PL_{cell_id}_{plot_idx:02d}"
                        plot_idx += 1

                        features.append({
                            "type": "Feature",
                            "properties": {
                                "plot_id": plot_id,
                                "cell_id": cell_id,
                                "land_type": plot_type,
                                "color": color,
                                "area_sq_m": round(area_sqm, 1),
                                "area_hectares": round(area_sqm / 10000.0, 4),
                                "area_acres": round(area_sqm * 0.000247105, 3),
                                "perimeter_m": round(cv2.arcLength(cnt, True) * (max_lng - min_lng) / w * 111139.0, 1)
                            },
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [gps_ring]
                            }
                        })
        except Exception as e:
            print(f"Error during SAM inference: {e}")

    # Fallback to OpenCV Contour / Watershed segmentation if model produced too few plots
    if len(features) < 2:
        print("Falling back to OpenCV Spectral Contour segmentation...")
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        plot_idx = 1
        for cnt in contours:
            pixel_area = cv2.contourArea(cnt)
            if pixel_area < 250 or pixel_area > (w * h * 0.9):
                continue

            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 3:
                continue

            gps_ring = [pixel_to_gps(pt[0][0], pt[0][1], bbox, w, h) for pt in approx]
            if gps_ring[0] != gps_ring[-1]:
                gps_ring.append(gps_ring[0])

            area_sqm = calculate_polygon_area_sqm([gps_ring], center_lat)
            if area_sqm < 30.0:
                continue

            # Classify using bounding box pixels
            x, y, cw, ch = cv2.boundingRect(cnt)
            roi = np_img[y:y+ch, x:x+cw]
            plot_type, color = classify_plot_type(roi)

            plot_id = f"PL_{cell_id}_{plot_idx:02d}"
            plot_idx += 1

            features.append({
                "type": "Feature",
                "properties": {
                    "plot_id": plot_id,
                    "cell_id": cell_id,
                    "land_type": plot_type,
                    "color": color,
                    "area_sq_m": round(area_sqm, 1),
                    "area_hectares": round(area_sqm / 10000.0, 4),
                    "area_acres": round(area_sqm * 0.000247105, 3),
                    "perimeter_m": round(cv2.arcLength(cnt, True) * (max_lng - min_lng) / w * 111139.0, 1)
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [gps_ring]
                }
            })

    return {
        "type": "FeatureCollection",
        "features": features
    }

# ─── API Endpoints ────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "FastSAM (Segment Anything)" if sam_model else "Initializing",
        "backend": "PyTorch CPU"
    }

@app.post("/segment")
async def segment_single(
    image: UploadFile = File(...),
    bbox: str = Form(...),  # "[minLng, minLat, maxLng, maxLat]"
    cell_id: str = Form("A1")
):
    try:
        bbox_list = json.loads(bbox)
        contents = await image.read()
        geojson_result = segment_image_plots(contents, bbox_list, cell_id)
        return geojson_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class BatchPatchItem(BaseModel):
    cell_id: str
    bbox: List[float]
    image_base64: str

class BatchSegmentRequest(BaseModel):
    patches: List[BatchPatchItem]

@app.post("/segment_batch")
async def segment_batch(request: BatchSegmentRequest):
    import base64
    all_features = []

    for patch in request.patches:
        try:
            # Decode base64 image data
            img_data = patch.image_base64
            if "," in img_data:
                img_data = img_data.split(",")[1]
            img_bytes = base64.b64decode(img_data)

            res = segment_image_plots(img_bytes, patch.bbox, patch.cell_id)
            all_features.extend(res.get("features", []))
        except Exception as e:
            print(f"Error processing patch {patch.cell_id}: {e}")

    return {
        "type": "FeatureCollection",
        "features": all_features,
        "total_plots": len(all_features)
    }

if __name__ == "__main__":
    import uvicorn
    # Pre-warm model
    get_model()
    uvicorn.run(app, host="127.0.0.1", port=8000)
