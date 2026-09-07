import io
import json
import math
import os
import re
import base64
import traceback
import urllib.request
import urllib.parse
from typing import List, Optional
import numpy as np
from PIL import Image
import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid

# Load environment variables
load_dotenv()

# Initialize FastAPI App
app = FastAPI(
    title="GeoAdhikar Satellite Plot Segmentation API",
    version="3.0.0",
    description="Morphological rooftop detection + Prompted SAM + HSV classification"
)

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model
sam_model = None

def get_sam():
    global sam_model
    if sam_model is None:
        try:
            from ultralytics import FastSAM
            print("🚀 Loading FastSAM promptable segmentation model...")
            sam_model = FastSAM("FastSAM-s.pt")
            print("✅ FastSAM loaded successfully!")
        except Exception as e:
            print(f"⚠️ Failed to load FastSAM: {e}")
    return sam_model

# ─── Helper: Pixel to GPS ──────────────────────────────────────────────────────
def pixel_to_gps(px, py, bbox, img_width=512, img_height=512):
    min_lng, min_lat, max_lng, max_lat = bbox
    lng = min_lng + (px / img_width) * (max_lng - min_lng)
    lat = max_lat - (py / img_height) * (max_lat - min_lat)
    return [round(lng, 7), round(lat, 7)]

# ─── Helper: Geodesic Area ─────────────────────────────────────────────────────
def calculate_polygon_area_sqm(coordinates, lat_ref):
    m_per_deg_lat = 111139.0
    m_per_deg_lng = 111139.0 * math.cos(math.radians(lat_ref))

    ring = coordinates[0]
    if len(ring) < 3:
        return 0.0

    ref_lng, ref_lat = ring[0]
    pts = []
    for lng, lat in ring:
        x = (lng - ref_lng) * m_per_deg_lng
        y = (lat - ref_lat) * m_per_deg_lat
        pts.append((x, y))

    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0

# ═══════════════════════════════════════════════════════════════════════════════
# TILE-LEVEL CONTEXT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
def detect_tile_context(np_img):
    """
    Analyze the full tile to determine if it's urban, rural, vegetation, or water.
    Returns: dict with context flags and dominant type string.
    """
    h, w, _ = np_img.shape
    hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    # Texture variance (high = built-up, low = uniform field/water)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Color statistics
    r = np_img[:, :, 0].astype(float)
    g = np_img[:, :, 1].astype(float)
    b = np_img[:, :, 2].astype(float)
    mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)
    brightness = (mean_r + mean_g + mean_b) / 3.0

    # HSV analysis
    hue = hsv[:, :, 0].astype(float)
    sat = hsv[:, :, 1].astype(float)
    val = hsv[:, :, 2].astype(float)
    mean_sat = np.mean(sat)
    mean_val = np.mean(val)

    # Green pixel ratio (vegetation indicator)
    green_mask = (g > r + 10) & (g > b + 5) & (g > 60)
    green_ratio = np.sum(green_mask) / (h * w)

    # Gray/neutral pixel ratio (built-up indicator)
    color_diff = np.abs(r - g) + np.abs(g - b) + np.abs(r - b)
    gray_mask = (color_diff < 60) & (gray > 50) & (gray < 220)
    gray_ratio = np.sum(gray_mask) / (h * w)

    # Blue pixel ratio (water indicator)
    blue_mask = (b > r + 15) & (b > g) & (b > 40)
    blue_ratio = np.sum(blue_mask) / (h * w)

    # Edge density (structures have more edges)
    edges = cv2.Canny(gray, 40, 120)
    edge_density = np.sum(edges > 0) / (h * w)

    # Determine dominant context
    is_urban = (laplacian_var > 200 and edge_density > 0.08) or \
               (gray_ratio > 0.35 and edge_density > 0.06) or \
               (laplacian_var > 400)
    is_vegetation = green_ratio > 0.35
    is_water = blue_ratio > 0.30
    is_barren = brightness > 130 and mean_sat < 40 and green_ratio < 0.10

    if is_water and blue_ratio > 0.5:
        dominant = "water"
    elif is_urban:
        dominant = "urban"
    elif is_vegetation and green_ratio > 0.5:
        dominant = "vegetation"
    elif is_barren:
        dominant = "barren"
    elif is_vegetation:
        dominant = "mixed_vegetation"
    else:
        dominant = "mixed_urban"

    ctx = {
        "dominant": dominant,
        "is_urban": bool(is_urban),
        "is_vegetation": bool(is_vegetation),
        "is_water": bool(is_water),
        "laplacian_var": round(float(laplacian_var), 1),
        "edge_density": round(float(edge_density), 4),
        "green_ratio": round(float(green_ratio), 3),
        "gray_ratio": round(float(gray_ratio), 3),
        "blue_ratio": round(float(blue_ratio), 3),
        "brightness": round(float(brightness), 1),
        "mean_sat": round(float(mean_sat), 1)
    }
    print(f"   🏙️ Tile context: {dominant} | edges={edge_density:.3f} laplacian={laplacian_var:.0f} "
          f"green={green_ratio:.2f} gray={gray_ratio:.2f} bright={brightness:.0f}")
    return ctx

# ═══════════════════════════════════════════════════════════════════════════════
# IMPROVED LAND-COVER CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════
def classify_plot_type(roi_rgb, tile_context=None):
    """
    Enhanced classifier using:
    1. HSV color space analysis
    2. Texture variance (Laplacian)
    3. Tile-level context awareness (urban area → strong bias toward settlement)
    """
    if roi_rgb.size == 0:
        return "Unclassified", "#9ca3af"

    # Handle both 2D (N,3) masked arrays and 3D (H,W,3) rectangular ROIs
    if len(roi_rgb.shape) == 2 and roi_rgb.shape[1] >= 3:
        r = float(np.mean(roi_rgb[:, 0]))
        g = float(np.mean(roi_rgb[:, 1]))
        b = float(np.mean(roi_rgb[:, 2]))
    elif len(roi_rgb.shape) == 3:
        r = float(np.mean(roi_rgb[:, :, 0]))
        g = float(np.mean(roi_rgb[:, :, 1]))
        b = float(np.mean(roi_rgb[:, :, 2]))
    else:
        return "Unclassified", "#9ca3af"

    brightness = (r + g + b) / 3.0

    # Convert mean color to HSV
    rgb_pixel = np.array([[[int(r), int(g), int(b)]]], dtype=np.uint8)
    hsv_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2HSV)
    hue = float(hsv_pixel[0, 0, 0])
    sat = float(hsv_pixel[0, 0, 1])
    val = float(hsv_pixel[0, 0, 2])

    # Texture variance from rectangular ROI
    texture_var = 0.0
    if len(roi_rgb.shape) == 3 and roi_rgb.shape[0] > 4 and roi_rgb.shape[1] > 4:
        gray_roi = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        texture_var = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())

    # Green vegetation indices
    green_excess = g - max(r, b)  # how much greener vs other channels
    denom = (g + r - b)
    vari = (g - r) / denom if abs(denom) > 1 else 0.0

    is_tile_urban = tile_context and tile_context.get("is_urban", False)

    # ── 1. Water: strong blue, dark ──
    if b > r + 20 and b > g + 5 and brightness < 120 and sat > 30:
        return "Water Body", "#38bdf8"

    # ── 2. In URBAN tiles: Settlement is the dominant class ──
    if is_tile_urban:
        # Only classify as vegetation if VERY strongly green
        if green_excess > 30 and sat > 80 and 35 < hue < 85 and brightness > 50:
            return "Forest / Dense Canopy", "#059669"

        # Everything else in an urban tile is settlement
        return "Habitation / Settlement", "#ef4444"

    # ── 3. NON-URBAN tiles: Standard spectral classification ──

    # Strong vegetation: clearly green-dominant, high saturation
    if green_excess > 20 and sat > 60 and 35 < hue < 85:
        return "Forest / Dense Canopy", "#059669"

    # Moderate vegetation / cropland
    if green_excess > 8 and sat > 40 and 30 < hue < 90 and brightness > 40:
        return "Agricultural / Cropland", "#10b981"

    # Settlement indicators
    is_settlement = False

    # Gray/neutral rooftops (low saturation)
    if sat < 50 and 50 < brightness < 200:
        is_settlement = True

    # Brown/tan rooftops
    if 8 < hue < 25 and 30 < sat < 150 and brightness > 60:
        is_settlement = True

    # Blue metal sheets
    if 90 < hue < 130 and sat > 40 and brightness > 60:
        is_settlement = True

    # Red/terracotta rooftops
    if (hue < 12 or hue > 165) and sat > 50 and brightness > 50:
        is_settlement = True

    # White/bright concrete
    if brightness > 180 and sat < 40:
        is_settlement = True

    # High texture = complex structures
    if texture_var > 300:
        is_settlement = True

    if is_settlement:
        return "Habitation / Settlement", "#ef4444"

    return "Fallow / Barren Land", "#f59e0b"

# ═══════════════════════════════════════════════════════════════════════════════
# SATELLITE-AWARE STRUCTURE/BUILDING DETECTOR
# (Replaces COCO-trained YOLOv8 which cannot detect buildings)
# ═══════════════════════════════════════════════════════════════════════════════
def detect_building_candidates(np_img, tile_ctx, min_area_px=600):
    """
    Detects building footprint candidates in satellite imagery using:
    1. Adaptive thresholding on intensity to isolate rooftops from roads/shadows
    2. HSV color segmentation to separate rooftop materials
    3. Morphological operations to form clean rectangular regions
    4. Contour extraction with rectangularity filtering
    """
    h, w, _ = np_img.shape
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    candidates = []

    # ── Strategy 1: Adaptive threshold on grayscale ──
    # Buildings create distinct intensity regions compared to roads/shadows
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 51, -5
    )
    # Clean with morphological operations
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px or area > (w * h * 0.60):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        # Aspect ratio filter: buildings are roughly square or mildly rectangular
        aspect = max(bw, bh) / (min(bw, bh) + 1)
        if aspect > 5.0:  # Skip very elongated shapes (roads, walls)
            continue
        if bw < 20 or bh < 20:
            continue
        candidates.append([x, y, x + bw, y + bh])

    # ── Strategy 2: HSV color-based rooftop isolation ──
    # Concrete gray roofs (low saturation)
    gray_roof_mask = cv2.inRange(hsv, (0, 0, 60), (179, 55, 210))
    # Blue metal roofs
    blue_roof_mask = cv2.inRange(hsv, (90, 40, 50), (130, 255, 255))
    # Brown/tan roofs
    brown_roof_mask = cv2.inRange(hsv, (8, 30, 50), (25, 180, 220))
    # Red/terracotta
    red_roof_mask = cv2.inRange(hsv, (0, 50, 50), (8, 255, 255))

    combined_roof = gray_roof_mask | blue_roof_mask | brown_roof_mask | red_roof_mask
    combined_roof = cv2.morphologyEx(combined_roof, cv2.MORPH_OPEN, kernel_open)
    combined_roof = cv2.morphologyEx(combined_roof, cv2.MORPH_CLOSE, kernel_close)

    roof_contours, _ = cv2.findContours(combined_roof, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in roof_contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px or area > (w * h * 0.60):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, bh) / (min(bw, bh) + 1)
        if aspect > 5.0 or bw < 20 or bh < 20:
            continue
        candidates.append([x, y, x + bw, y + bh])

    # ── Strategy 3: Edge-based structure detection ──
    edges = cv2.Canny(blurred, 30, 100)
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated_edges = cv2.dilate(edges, kernel_edge, iterations=2)
    # Fill enclosed regions
    flood_mask = dilated_edges.copy()
    cv2.floodFill(flood_mask, None, (0, 0), 255)
    flood_mask = cv2.bitwise_not(flood_mask)
    filled_structures = flood_mask | dilated_edges
    filled_structures = cv2.morphologyEx(filled_structures, cv2.MORPH_CLOSE, kernel_close)

    edge_contours, _ = cv2.findContours(filled_structures, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in edge_contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px or area > (w * h * 0.60):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, bh) / (min(bw, bh) + 1)
        if aspect > 5.0 or bw < 20 or bh < 20:
            continue
        candidates.append([x, y, x + bw, y + bh])

    # ── NMS to deduplicate ──
    candidates.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept = []
    for b in candidates:
        if any(compute_box_iou(b, k) > 0.30 for k in kept):
            continue
        kept.append(b)

    # If we still have too few candidates in an urban tile, add grid fallback
    if len(kept) < 3 and tile_ctx.get("is_urban"):
        grid_step = w // 4
        for gy in range(0, h - grid_step + 1, grid_step):
            for gx in range(0, w - grid_step + 1, grid_step):
                box = [gx + 8, gy + 8, gx + grid_step - 8, gy + grid_step - 8]
                if not any(compute_box_iou(box, k) > 0.30 for k in kept):
                    kept.append(box)

    print(f"   🏗️ Detected {len(kept)} building/structure candidates")
    return kept[:30]

# ─── Box IoU ────────────────────────────────────────────────────────────────────
def compute_box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

# ─── 4-Sided Shape Regularization ──────────────────────────────────────────────
def regularize_polygon_contour(cnt, extent_threshold=0.55):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box_area = cv2.contourArea(box)
    cnt_area = cv2.contourArea(cnt)

    extent = (cnt_area / box_area) if box_area > 0 else 0.0

    # Mostly rectangular → snap to clean rotated rectangle
    if extent >= extent_threshold:
        pts = box.astype(np.int32)
        return [[int(p[0]), int(p[1])] for p in pts]

    # Simplify with Douglas-Peucker to 4-6 vertices
    peri = cv2.arcLength(cnt, True)
    for eps_factor in [0.04, 0.03, 0.02, 0.015]:
        approx = cv2.approxPolyDP(cnt, eps_factor * peri, True)
        if 4 <= len(approx) <= 6:
            return [[int(p[0][0]), int(p[0][1])] for p in approx]

    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) < 4:
        pts = box.astype(np.int32)
        return [[int(p[0]), int(p[1])] for p in pts]

    return [[int(p[0][0]), int(p[0][1])] for p in approx]

# ═══════════════════════════════════════════════════════════════════════════════
# CORE SEGMENTATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def segment_image_plots(
    image_bytes: bytes,
    bbox: List[float],
    cell_id: str = "A1",
    min_area_sqm: float = 50.0
):
    """
    Pipeline:
    1. Tile context detection (urban vs rural vs water)
    2. Morphological building/structure candidate detection
    3. Prompted FastSAM segmentation within each candidate box
    4. 4-sided cadastral regularization
    5. Shapely topological validation
    6. HSV + texture + context-aware land classification
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_img = np.array(image)
    h, w, _ = np_img.shape

    min_lng, min_lat, max_lng, max_lat = bbox
    center_lat = (min_lat + max_lat) / 2.0

    print(f"\n{'='*60}")
    print(f"📍 Processing tile {cell_id} | bbox={[round(x,5) for x in bbox]}")

    # Step 1: Analyze tile context
    tile_ctx = detect_tile_context(np_img)

    # Step 2: Detect building/structure candidates
    prompt_boxes = detect_building_candidates(np_img, tile_ctx)

    features = []
    sam = get_sam()

    # Step 3: Run SAM with candidate prompt boxes
    raw_masks = []
    if sam is not None and len(prompt_boxes) > 0:
        try:
            results = sam(
                np_img,
                bboxes=prompt_boxes,
                device="cpu",
                imgsz=512,
                conf=0.15,
                retina_masks=True,
                verbose=False
            )
            if results and len(results) > 0 and results[0].masks is not None:
                masks_data = results[0].masks.data.cpu().numpy()
                for i in range(len(masks_data)):
                    m = (masks_data[i] * 255).astype(np.uint8)
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    raw_masks.append(m)
        except Exception as e:
            print(f"   ⚠️ SAM prompted error: {e}")

    # Fallback: use prompt boxes as rectangular masks
    if len(raw_masks) == 0:
        for b in prompt_boxes:
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(m, (b[0], b[1]), (b[2], b[3]), 255, -1)
            raw_masks.append(m)

    print(f"   🎭 Processing {len(raw_masks)} masks...")

    # Step 4: Process each mask → regularized 4-sided plot
    plot_idx = 1
    existing_polys = []

    for mask in raw_masks:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        contours, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 400:
            continue

        # Regularize to 4-sided shape
        reg_pts = regularize_polygon_contour(cnt, extent_threshold=0.55)

        # Convert to GPS
        gps_ring = [pixel_to_gps(px, py, bbox, w, h) for px, py in reg_pts]
        if gps_ring[0] != gps_ring[-1]:
            gps_ring.append(gps_ring[0])

        # Shapely topological cleaning
        try:
            poly = Polygon(gps_ring)
            if not poly.is_valid:
                poly = make_valid(poly)
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda g: g.area)
            poly = poly.simplify(0.000005, preserve_topology=True)
            clean_coords = list(poly.exterior.coords)
            if len(clean_coords) < 4:
                continue
        except Exception:
            continue

        # Area filter
        area_sqm = calculate_polygon_area_sqm([clean_coords], center_lat)
        if area_sqm < min_area_sqm:
            continue

        # Overlap NMS against existing plots
        curr_poly = Polygon(clean_coords)
        if not curr_poly.is_valid:
            continue
        skip = False
        for ep in existing_polys:
            try:
                if curr_poly.intersects(ep):
                    inter_area = curr_poly.intersection(ep).area
                    if inter_area / curr_poly.area > 0.35:
                        skip = True
                        break
            except Exception:
                pass
        if skip:
            continue
        existing_polys.append(curr_poly)

        # Classify using ROI pixels + tile context
        mask_bool = smoothed > 0
        roi = np_img[mask_bool]

        # Also get the rectangular bounding box ROI for texture analysis
        x, y, bw, bh = cv2.boundingRect(cnt)
        rect_roi = np_img[y:y+bh, x:x+bw]
        plot_type, color = classify_plot_type(rect_roi, tile_ctx)

        plot_id = f"PL_{cell_id}_{plot_idx:02d}"
        plot_idx += 1

        n_vertices = len(clean_coords) - 1
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
                "vertices": n_vertices,
                "shape_type": "Rectangle/Quad" if n_vertices == 4 else f"{n_vertices}-Gon",
                "tile_context": tile_ctx["dominant"]
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [clean_coords]
            }
        })

    print(f"   ✅ Produced {len(features)} clean cadastral plots for tile {cell_id}")
    return {
        "type": "FeatureCollection",
        "features": features,
        "tile_context": tile_ctx
    }

# ─── API Endpoints ────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "version": "3.0.0",
        "detector": "Morphological Rooftop/Structure Detector (satellite-aware)",
        "segmenter": "FastSAM (Prompted Mode)",
        "classifier": "HSV + Texture + Tile Context",
        "regularization": "4-Sided Rectangle/Quad Enforced",
        "topology": "Shapely make_valid"
    }

@app.post("/segment")
async def segment_single(
    image: UploadFile = File(...),
    bbox: str = Form(...),
    cell_id: str = Form("A1"),
    min_area_sqm: float = Form(50.0)
):
    try:
        bbox_list = json.loads(bbox)
        contents = await image.read()
        geojson_result = segment_image_plots(contents, bbox_list, cell_id, min_area_sqm)
        return geojson_result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class BatchPatchItem(BaseModel):
    cell_id: str
    bbox: List[float]
    image_base64: str
    min_area_sqm: Optional[float] = 50.0

class BatchSegmentRequest(BaseModel):
    patches: List[BatchPatchItem]

@app.post("/segment_batch")
async def segment_batch(request: BatchSegmentRequest):
    import base64
    all_features = []

    for patch in request.patches:
        try:
            img_data = patch.image_base64
            if "," in img_data:
                img_data = img_data.split(",")[1]
            img_bytes = base64.b64decode(img_data)

            min_area = patch.min_area_sqm if patch.min_area_sqm is not None else 50.0
            res = segment_image_plots(img_bytes, patch.bbox, patch.cell_id, min_area)
            all_features.extend(res.get("features", []))
        except Exception as e:
            print(f"Error processing patch {patch.cell_id}: {e}")

    return {
        "type": "FeatureCollection",
        "features": all_features,
        "total_plots": len(all_features)
    }

# ═══════════════════════════════════════════════════════════════════════════════
# LAND RECORD OCR & GEOCODING
# ═══════════════════════════════════════════════════════════════════════════════

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ocr import extract_record

@app.post("/ocr")
async def perform_ocr(
    image: UploadFile = File(...)
):
    try:
        # Save uploaded image to a temporary file
        temp_image_path = "temp_uploaded_document.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(await image.read())

        # Call extract_record directly from ocr.py
        record = extract_record(temp_image_path)
        
        # Save to extracted_record.json just like ocr.py does
        output_file = "extracted_record.json"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(record.model_dump_json(indent=4))
            
        return record.model_dump()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class GeocodeRecordRequest(BaseModel):
    state: Optional[str] = ""
    district: Optional[str] = ""
    taluka_or_tehsil: Optional[str] = ""
    village: Optional[str] = ""
    survey_or_gat_no: Optional[str] = ""
    query: Optional[str] = ""

@app.post("/geocode-record")
def geocode_record(req: GeocodeRecordRequest):
    token = os.environ.get("VITE_MAPBOX_TOKEN", "")
    
    parts = []
    if req.village and req.village != "N/A":
        parts.append(req.village)
    if req.taluka_or_tehsil and req.taluka_or_tehsil != "N/A":
        parts.append(req.taluka_or_tehsil)
    if req.district and req.district != "N/A":
        parts.append(req.district)
    if req.state and req.state != "N/A":
        parts.append(req.state)
        
    query_str = req.query.strip() if req.query else ", ".join(parts)
    if not query_str:
        query_str = "Maharashtra, India"

    encoded_q = urllib.parse.quote(query_str)
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{encoded_q}.json?access_token={token}&country=in&limit=1"

    try:
        req_obj = urllib.request.Request(url, headers={"User-Agent": "GeoAdhikar/3.0"})
        with urllib.request.urlopen(req_obj, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("features") and len(data["features"]) > 0:
                feat = data["features"][0]
                lng, lat = feat["center"]
                bbox = feat.get("bbox", [lng - 0.005, lat - 0.005, lng + 0.005, lat + 0.005])
                return {
                    "success": True,
                    "query": query_str,
                    "place_name": feat.get("place_name", query_str),
                    "coordinates": [lng, lat],
                    "bbox": bbox
                }
    except Exception as e:
        print(f"Geocoding error: {e}")

    # Default fallback
    return {
        "success": False,
        "query": query_str,
        "place_name": query_str,
        "coordinates": [73.8567, 18.5204],
        "bbox": [73.8167, 18.4804, 73.8967, 18.5604],
        "message": "Mapbox geocoding failed or token missing, defaulted to region center."
    }

if __name__ == "__main__":
    import uvicorn
    get_sam()
    print("🚀 Server ready at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
