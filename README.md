# 🌐 GeoAdhikar — WebGIS Decision Support & AI Land Parcel Segmentation

**GeoAdhikar** is an advanced geospatial decision-support and land-intelligence platform designed for rural and tribal development prioritization in India. It integrates interactive 3D WebGIS, multi-state policy decision analytics, a patch-based high-resolution snapshot tiling engine, and deep learning land parcel segmentation powered by Meta's **Segment Anything Model (SAM / FastSAM)**.

---

## 🚀 Key Modules & Capabilities

### 1. 🌐 Scheme Prioritization & WebGIS Decision Support
* **Multi-State Hierarchy:** Dynamic selection spanning Madhya Pradesh, Odisha, Telangana, Tripura, and expandable to all Indian states.
* **Granular Demographic Indicators:** Area profile cards evaluating tribal population %, forest cover %, agricultural land %, tap water access (JJM), and Forest Rights Act (FRA) claims.
* **Deterministic Rule-Based Scoring:** Prioritizes national welfare and tribal schemes with transparent rule-level explainability.
* **One-Click Official PDF Report:** Generates publication-ready executive summary reports with vector boundaries and score cards.

### 2. 🛰️ AI Plot Snapshot Scanner & Tiling Grid Engine
* **Dedicated Location Search:** Direct coordinate input, address lookup via Mapbox Geocoder, interactive map picking, or JSON payload ingestion.
* **Geodesic AOI Bounding Box:** Mathematically projects exact metric areas ($100\text{m} \times 100\text{m}$, $250\text{m} \times 250\text{m}$, $500\text{m} \times 500\text{m}$, $1\text{ km}^2$, etc.).
* **Patch-Based Subdivision:** Divides the large area of interest into equal mini-squares ($20\text{m}$, $25\text{m}$, $50\text{m}$, $100\text{m}$) labeled systematically (`A1`, `A2`, `B1`...).
* **Deep-Zoom High-Res Captures:** Automatically fetches undistorted, orthographic $512\times512$ satellite snapshots with zero edge margins.
* **ZIP Archive Packaging:** Downloads all patch snapshots bundled with a machine-readable `metadata.json` and spatial `grid_boundaries.geojson`.

### 3. 🧠 SAM AI Land Parcel Segmentation & Classification
* **Zero-Shot Boundary Delineation:** Uses **FastSAM / Segment Anything** to segment physical farm plots, fences, hedgerows, and structures.
* **Affine Coordinate Mapping:** Transforms pixel coordinates $(px, py) \rightarrow \text{GPS } (lng, lat)$ so plot boundaries become real-world vector polygons.
* **Metric Area Computation:** Calculates exact geodesic area in **Square Meters ($m^2$)**, **Hectares ($ha$)**, and **Acres**.
* **Land-Cover Classification:**
  * 🌾 **Agricultural / Cropland**
  * 🌳 **Forest / Dense Tree Canopy**
  * 🏚️ **Habitation / Settlement**
  * 💧 **Water Body**
  * 🪨 **Fallow / Barren Land**
* **Vector Exports:** One-click download of `plots.geojson` (ready for QGIS, ArcGIS, Google Earth) and `plots_summary.csv`.

---

## 🛠️ Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Frontend Core** | Vanilla JavaScript (ES Modules), HTML5, CSS3 |
| **Build & Tooling** | Vite 5 |
| **Map & 3D Terrain** | Mapbox GL JS v3, Mapbox Terrain DEM, Satellite Streets v12 |
| **Spatial Analytics** | Turf.js (`@turf/turf`) |
| **Archiving & PDF** | JSZip, jsPDF, html2canvas |
| **AI / ML Backend** | Python 3.11, FastAPI, Uvicorn |
| **Segmentation Models** | Meta FastSAM (`ultralytics`), PyTorch, OpenCV, Pillow, Shapely |

---

## ⚡ Quick Start & Installation

### 1. Clone & Install Frontend
```bash
git clone <your-repo-url>
cd GeoAdhikar
npm install
```

### 2. Start the Frontend Development Server
```bash
npm run dev
```
Open **`http://localhost:5173/`** in your browser.

### 3. (Optional) Run the AI Segmentation Backend
```bash
python backend/app.py
```
*The FastAPI backend runs on `http://127.0.0.1:8000`. The frontend includes an automatic client-side computer vision engine that works immediately even if the Python service is offline.*

### 4. Build for Production
```bash
npm run build
```

---

## 📁 Repository Structure

```
GeoAdhikar/
├── backend/
│   └── app.py                     # FastAPI backend for FastSAM segmentation
├── public/
│   └── data/
│       ├── blocks.json            # Tribal and administrative demographic profiles
│       ├── schemes.json           # Rule engine definitions for welfare schemes
│       └── admin_boundaries_sample.geojson # Boundary polygon dataset
├── src/
│   ├── app.js                     # DSS logic, Mapbox initialization, Geocoder search
│   ├── main.js                    # Application entrypoint
│   ├── scanner.js                 # 100m grid generator, snapshot engine, SAM integration
│   └── styles.css                 # Palantir dark theme & responsive UI styles
├── index.html                     # Application interface & layout
├── package.json                   # Node dependencies & scripts
├── vite.config.js                 # Vite bundler configuration
└── README.md                      # Project documentation
```

---

## 📜 License & Acknowledgements
- Research and academic prototype designed for public welfare and land rights decision-making.
- Data references: Census of India, Jal Jeevan Mission, Forest Survey of India (ISFR), and Ministry of Tribal Affairs.
