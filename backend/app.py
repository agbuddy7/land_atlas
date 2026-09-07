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
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid

# Initialize FastAPI App
app = FastAPI(
    title="GeoAdhikar YOLOv8 + Prompted SAM Plot Segmentation API",
    version="2.0.0"
)

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models
yolo_model = None
sam_model = None

def get_yolo():
    global yolo_model
    if yolo_model is None:
        try:
            from ultralytics import YOLO
            print("🚀 Loading YOLOv8 candidate detector...")
            yolo_model = YOLO("yolov8n.pt")
            print("✅ YOLOv8 loaded successfully!")
        except Exception as e:
            print(f"⚠️ Failed to load YOLOv8: {e}")
    return yolo_model

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
    at a given latitude using the Shoelace formula projected to metric meters.
    """
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

# ─── Helper: Classify Land Type from Spectral RGB ──────────────────────────────
def classify_plot_type(roi_rgb):
    """
    Classifies plot type using vegetation/soil spectral index in RGB.
    Safely handles both 2D (N, 3) and 3D (H, W, 3) arrays.
    """
    if roi_rgb.size == 0:
        return "Unclassified", "#9ca3af"

    if len(roi_rgb.shape) == 2:
        r = float(np.mean(roi_rgb[:, 0]))
        g = float(np.mean(roi_rgb[:, 1]))
        b = float(np.mean(roi_rgb[:, 2]))
    else:
        r = float(np.mean(roi_rgb[:, :, 0]))
        g = float(np.mean(roi_rgb[:, :, 1]))
        b = float(np.mean(roi_rgb[:, :, 2]))

    # Visible Atmospherically Resistant Index (VARI) for greenness
    denom = (g + r - b)
    vari = (g - r) / denom if denom != 0 else 0.0

    # Overall brightness
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

# ─── Box Intersection over Union (IoU) Helper ──────────────────────────────────
def compute_box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

# ─── Candidate Prompt Generation (YOLOv8 + Salient Open Plots) ────────────────
def generate_candidate_prompt_boxes(np_img, min_box_dim=28, min_box_area=800):
    """
    Generates candidate bounding boxes from:
    1. YOLOv8 detector (for structures, rooftops, objects)
    2. Salient edge/texture regional components (for open parcels/agricultural fields)
    3. Non-Maximum Suppression to avoid overlapping duplicates
    """
    h, w, _ = np_img.shape
    candidates = []

    # 1. Run YOLOv8 detection
    yolo = get_yolo()
    if yolo is not None:
        try:
            yolo_res = yolo(np_img, conf=0.12, imgsz=512, verbose=False)
            if yolo_res and len(yolo_res) > 0 and yolo_res[0].boxes is not None:
                for box in yolo_res[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = [int(v) for v in box]
                    bw = x2 - x1
                    bh = y2 - y1
                    if bw >= min_box_dim and bh >= min_box_dim and (bw * bh) >= min_box_area:
                        candidates.append([x1, y1, x2, y2])
        except Exception as e:
            print(f"YOLO candidate error: {e}")

    # 2. Extract salient open parcel boundaries (agricultural fields, open plots)
    try:
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.bilateralFilter(gray, 7, 50, 50)
        edges = cv2.Canny(blurred, 35, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Must be a substantial parcel region (not tiny noise, not whole image)
            if area >= (min_box_area * 1.5) and area < (w * h * 0.85):
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw >= min_box_dim and bh >= min_box_dim:
                    candidates.append([x, y, x + bw, y + bh])
    except Exception as e:
        print(f"Open parcel extraction error: {e}")

    # If too few candidates, generate regular grid prompt boxes
    if len(candidates) < 4:
        step = 160
        for y in range(16, h - 80, step):
            for x in range(16, w - 80, step):
                candidates.append([x, y, min(w - 16, x + step), min(h - 16, y + step)])

    # 3. Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes
    candidates.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept_boxes = []
    for b in candidates:
        if any(compute_box_iou(b, kb) > 0.35 for kb in kept_boxes):
            continue
        kept_boxes.append(b)

    # Limit to top 20 candidate parcels per tile
    return kept_boxes[:20]

# ─── 4-Sided Shape Regularization (Rectangle / Quadrilateral Favoring) ────────
def regularize_polygon_contour(cnt, extent_threshold=0.60):
    """
    Enforces regular 4-sided shapes (rectangles/squares/quads) for cadastral parcels:
    - If contour is predominantly rectangular (extent >= 0.60), returns minimum rotated rectangle (4 points).
    - Otherwise, simplifies contour to 4-6 dominant vertices with Douglas-Peucker.
    """
    # 1. Minimum rotated bounding rectangle
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)  # 4 points
    box_area = cv2.contourArea(box)
    cnt_area = cv2.contourArea(cnt)

    extent = (cnt_area / box_area) if box_area > 0 else 0.0

    # If it is mostly rectangular, snap to the clean 4-sided minimum rotated rectangle
    if extent >= extent_threshold:
        pts = box.astype(np.int32)
        return [[int(p[0]), int(p[1])] for p in pts]

    # Otherwise, simplify with Douglas-Peucker to produce 4 to 6 clean vertices
    peri = cv2.arcLength(cnt, True)
    for eps_factor in [0.035, 0.025, 0.015, 0.01]:
        approx = cv2.approxPolyDP(cnt, eps_factor * peri, True)
        if 4 <= len(approx) <= 6:
            return [[int(p[0][0]), int(p[0][1])] for p in approx]

    # Fallback to standard approximation (at least 4 points)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) < 4:
        pts = box.astype(np.int32)
        return [[int(p[0]), int(p[1])] for p in pts]

    return [[int(p[0][0]), int(p[0][1])] for p in approx]

# ─── Core Segmentation Function ───────────────────────────────────────────────
def segment_image_plots(
    image_bytes: bytes,
    bbox: List[float],
    cell_id: str = "A1",
    min_area_sqm: float = 80.0
):
    """
    Executes:
    1. YOLOv8 candidate detection -> prompt bounding boxes
    2. Prompted FastSAM segmentation
    3. 4-sided cadastral regularisation (square/rectangle favoring)
    4. Topological cleaning via Shapely make_valid (zero self-intersections / no diagonal lines)
    5. Minimum parcel area filtering (min_area_sqm >= 80m²)
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_img = np.array(image)
    h, w, _ = np_img.shape

    min_lng, min_lat, max_lng, max_lat = bbox
    center_lat = (min_lat + max_lat) / 2.0

    features = []
    sam = get_sam()

    # Step 1: Generate clean candidate prompt boxes using YOLOv8
    prompt_boxes = generate_candidate_prompt_boxes(np_img)
    print(f"📦 Generated {len(prompt_boxes)} candidate prompt boxes for tile {cell_id}")

    raw_masks = []
    if sam is not None and len(prompt_boxes) > 0:
        try:
            # Run FastSAM in PROMPTED mode with candidate boxes
            results = sam(
                np_img,
                bboxes=prompt_boxes,
                device="cpu",
                imgsz=512,
                conf=0.20,
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
            print(f"⚠️ Error during prompted SAM inference: {e}")

    # Fallback to prompt boxes if SAM produces no masks
    if len(raw_masks) == 0:
        for b in prompt_boxes:
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(m, (b[0], b[1]), (b[2], b[3]), 255, -1)
            raw_masks.append(m)

    # Step 2: Process masks into clean 4-sided cadastral plots
    plot_idx = 1
    existing_polygons = []

    for mask in raw_masks:
        # Morphological closing to eliminate internal holes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        smoothed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Get largest contour
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 500:  # Skip tiny pixel noise
            continue

        # Regularize into 4-sided shape (rectangle/square/quad)
        reg_pts = regularize_polygon_contour(cnt, extent_threshold=0.60)

        # Convert to GPS ring
        gps_ring = [pixel_to_gps(px, py, bbox, w, h) for px, py in reg_pts]
        if gps_ring[0] != gps_ring[-1]:
            gps_ring.append(gps_ring[0])

        # Step 3: Shapely Topological Validation (Prevents Mapbox Earcut diagonal triangles!)
        try:
            poly = Polygon(gps_ring)
            if not poly.is_valid:
                poly = make_valid(poly)

            # If it split into MultiPolygon, select the largest polygon component
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda g: g.area)

            # Simplify with topology preservation
            poly = poly.simplify(0.00001, preserve_topology=True)

            # Extract clean exterior coordinates
            clean_coords = list(poly.exterior.coords)
            if len(clean_coords) < 4:
                continue
        except Exception as e:
            print(f"Topology cleanup error: {e}")
            continue

        # Calculate metric area
        area_sqm = calculate_polygon_area_sqm([clean_coords], center_lat)
        if area_sqm < min_area_sqm:
            continue

        # Non-Maximum Suppression: Discard if overlapping heavily with an existing plot
        curr_poly_shape = Polygon(clean_coords)
        has_large_overlap = False
        for ep in existing_polygons:
            if curr_poly_shape.intersects(ep):
                inter = curr_poly_shape.intersection(ep).area
                if inter / curr_poly_shape.area > 0.40:
                    has_large_overlap = True
                    break
        if has_large_overlap:
            continue

        existing_polygons.append(curr_poly_shape)

        # Classify land cover safely using ROI pixels
        mask_bool = smoothed_mask > 0
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
                "vertices": len(clean_coords) - 1,
                "shape_type": "4-Sided Quadrilateral/Rectangle" if (len(clean_coords) - 1) == 4 else f"{len(clean_coords)-1}-Gon Parcel"
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [clean_coords]
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
        "detector": "YOLOv8n",
        "segmenter": "FastSAM (Prompted Mode)",
        "regularization": "4-Sided Rectangle/Quad Enforced",
        "topology_cleaner": "Shapely make_valid"
    }

@app.post("/segment")
async def segment_single(
    image: UploadFile = File(...),
    bbox: str = Form(...),  # "[minLng, minLat, maxLng, maxLat]"
    cell_id: str = Form("A1"),
    min_area_sqm: float = Form(80.0)
):
    try:
        bbox_list = json.loads(bbox)
        contents = await image.read()
        geojson_result = segment_image_plots(contents, bbox_list, cell_id, min_area_sqm)
        return geojson_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class BatchPatchItem(BaseModel):
    cell_id: str
    bbox: List[float]
    image_base64: str
    min_area_sqm: Optional[float] = 80.0

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

            min_area = patch.min_area_sqm if patch.min_area_sqm is not None else 80.0
            res = segment_image_plots(img_bytes, patch.bbox, patch.cell_id, min_area)
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
    # Pre-warm models
    get_yolo()
    get_sam()
    uvicorn.run(app, host="127.0.0.1", port=8000)
