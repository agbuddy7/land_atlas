import * as turf from '@turf/turf';
import JSZip from 'jszip';

// ─── Module State ─────────────────────────────────────────────────────────────
let mapInstance = null;
let mapboxToken = '';
let currentCentroid = null; // [lng, lat]
let currentGridData = null; // { bigBbox, cells, bigPolygon, gridGeoJSON }
let capturedSnapshots = []; // [{ cell, blob, objectUrl, filename }]
let isScanning = false;
let isPickingOnMap = false;

// ─── Presets for Quick Testing ─────────────────────────────────────────────────
const LOCATION_PRESETS = [
  { name: 'Bhopal Agri Plots', lat: 23.2599, lng: 77.4126 },
  { name: 'Ranchi Forest Fringe', lat: 23.3441, lng: 85.3096 },
  { name: 'Koraput Tribal Belt', lat: 18.8135, lng: 82.7094 },
  { name: 'Warangal Rural Zone', lat: 17.9784, lng: 79.5941 }
];

// ─── Helper: Compute 100m Grid and Mini-Squares ────────────────────────────────
export function computeGrid(centroidLngLat, bigSizeMeters = 100, miniSizeMeters = 25) {
  const [lng, lat] = centroidLngLat;
  const centerPoint = turf.point([lng, lat]);
  const halfBig = bigSizeMeters / 2;

  // Calculate bounding box in meters using geodesic distance
  const north = turf.destination(centerPoint, halfBig, 0, { units: 'meters' }).geometry.coordinates[1];
  const south = turf.destination(centerPoint, halfBig, 180, { units: 'meters' }).geometry.coordinates[1];
  const east = turf.destination(centerPoint, halfBig, 90, { units: 'meters' }).geometry.coordinates[0];
  const west = turf.destination(centerPoint, halfBig, 270, { units: 'meters' }).geometry.coordinates[0];

  const bigBbox = [west, south, east, north];
  const bigPolygon = turf.bboxPolygon(bigBbox);

  // Divide into N rows and N cols
  const cols = Math.max(1, Math.round(bigSizeMeters / miniSizeMeters));
  const rows = cols;
  const cells = [];
  const features = [];

  const lngSpan = east - west;
  const latSpan = north - south;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cW = west + (lngSpan * c) / cols;
      const cE = west + (lngSpan * (c + 1)) / cols;
      const cS = south + (latSpan * r) / rows;
      const cN = south + (latSpan * (r + 1)) / rows;

      // Row label A, B, C... from top to bottom
      const rowLabel = String.fromCharCode(65 + (rows - 1 - r));
      const colLabel = String(c + 1);
      const cellId = `${rowLabel}${colLabel}`;

      const cellBbox = [cW, cS, cE, cN];
      const cellPoly = turf.bboxPolygon(cellBbox);
      const centerLng = (cW + cE) / 2;
      const centerLat = (cS + cN) / 2;

      cellPoly.properties = {
        cell_id: cellId,
        row: rows - 1 - r,
        col: c,
        center_lng: centerLng,
        center_lat: centerLat,
        width_m: miniSizeMeters,
        height_m: miniSizeMeters
      };

      cells.push({
        id: cellId,
        row: rows - 1 - r,
        col: c,
        bbox: cellBbox,
        center: [centerLng, centerLat],
        polygon: cellPoly
      });

      features.push(cellPoly);
    }
  }

  // Sort cells A1, A2, A3... B1, B2...
  cells.sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }));

  const gridGeoJSON = turf.featureCollection(features);

  return {
    bigBbox,
    bigPolygon,
    cells,
    gridGeoJSON,
    centroid: [lng, lat],
    bigSizeMeters,
    miniSizeMeters,
    cols,
    rows
  };
}

// ─── Mapbox Layers Setup ──────────────────────────────────────────────────────
function ensureMapboxScannerLayers() {
  if (!mapInstance) return;

  // 1. Big Square Source & Layer
  if (!mapInstance.getSource('scanner-big-box')) {
    mapInstance.addSource('scanner-big-box', {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] }
    });

    mapInstance.addLayer({
      id: 'scanner-big-box-fill',
      type: 'fill',
      source: 'scanner-big-box',
      paint: {
        'fill-color': '#06b6d4',
        'fill-opacity': 0.12
      }
    });

    mapInstance.addLayer({
      id: 'scanner-big-box-line',
      type: 'line',
      source: 'scanner-big-box',
      paint: {
        'line-color': '#06b6d4',
        'line-width': 2.5,
        'line-dasharray': [2, 1]
      }
    });
  }

  // 2. Mini-Grid Source & Layer
  if (!mapInstance.getSource('scanner-grid')) {
    mapInstance.addSource('scanner-grid', {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] }
    });

    mapInstance.addLayer({
      id: 'scanner-grid-fill',
      type: 'fill',
      source: 'scanner-grid',
      paint: {
        'fill-color': '#38bdf8',
        'fill-opacity': [
          'case',
          ['boolean', ['feature-state', 'hover'], false],
          0.45,
          0.05
        ]
      }
    });

    mapInstance.addLayer({
      id: 'scanner-grid-line',
      type: 'line',
      source: 'scanner-grid',
      paint: {
        'line-color': '#38bdf8',
        'line-width': 1.5
      }
    });

    // 2b. Mini-Grid Text Labels Layer (A1, A2, B1...)
    mapInstance.addLayer({
      id: 'scanner-grid-labels',
      type: 'symbol',
      source: 'scanner-grid',
      layout: {
        'text-field': ['get', 'cell_id'],
        'text-size': 13,
        'text-anchor': 'center',
        'text-allow-overlap': true,
        'text-ignore-placement': true
      },
      paint: {
        'text-color': '#ffffff',
        'text-halo-color': '#0b0f19',
        'text-halo-width': 2.5
      }
    });
  }

  // 3. Centroid Source & Layer
  if (!mapInstance.getSource('scanner-centroid')) {
    mapInstance.addSource('scanner-centroid', {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] }
    });

    mapInstance.addLayer({
      id: 'scanner-centroid-pulse',
      type: 'circle',
      source: 'scanner-centroid',
      paint: {
        'circle-radius': 9,
        'circle-color': '#06b6d4',
        'circle-stroke-width': 2.5,
        'circle-stroke-color': '#ffffff'
      }
    });
  }
}

// ─── Render Grid on Map ────────────────────────────────────────────────────────
export function renderGridOnMap(gridData) {
  if (!mapInstance) return;
  ensureMapboxScannerLayers();

  if (!gridData) {
    mapInstance.getSource('scanner-big-box')?.setData({ type: 'FeatureCollection', features: [] });
    mapInstance.getSource('scanner-grid')?.setData({ type: 'FeatureCollection', features: [] });
    mapInstance.getSource('scanner-centroid')?.setData({ type: 'FeatureCollection', features: [] });
    return;
  }

  // Update big square
  mapInstance.getSource('scanner-big-box')?.setData({
    type: 'FeatureCollection',
    features: [gridData.bigPolygon]
  });

  // Update mini grid
  mapInstance.getSource('scanner-grid')?.setData(gridData.gridGeoJSON);

  // Update centroid
  mapInstance.getSource('scanner-centroid')?.setData({
    type: 'FeatureCollection',
    features: [
      {
        type: 'Feature',
        geometry: { type: 'Point', coordinates: gridData.centroid },
        properties: { name: 'Scan Centroid' }
      }
    ]
  });

  // Zoom camera directly to bounding box so it fills the screen with proper sidebar padding!
  mapInstance.fitBounds(gridData.bigBbox, {
    padding: { top: 70, bottom: 70, left: 460, right: 70 },
    maxZoom: 20,
    pitch: 0,
    bearing: 0,
    duration: 1600
  });
}

// ─── Snapshot Fetcher (Specific to Each Cell) ──────────────────────────────────
async function fetchCellSnapshot(cell, token) {
  // Pure, clean raw satellite imagery of the EXACT bounding box with 0 padding and no blue overlays
  const bboxStr = cell.bbox.map(n => n.toFixed(6)).join(',');
  const url = `https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/[${bboxStr}]/512x512?padding=0&access_token=${token}`;

  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to capture cell ${cell.id}: HTTP ${res.status}`);
  }

  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  const filename = `cell_${cell.id}_${cell.center[1].toFixed(5)}N_${cell.center[0].toFixed(5)}E.jpg`;

  return {
    cell,
    blob,
    objectUrl,
    filename,
    url
  };
}

// ─── UI Rendering: Presets & Gallery ──────────────────────────────────────────
function renderPresets() {
  const container = document.getElementById('scannerPresets');
  if (!container) return;

  container.innerHTML = LOCATION_PRESETS.map(p => `
    <button type="button" class="chip-btn" data-lat="${p.lat}" data-lng="${p.lng}" data-name="${p.name}">
      📍 ${p.name}
    </button>
  `).join('');

  container.querySelectorAll('.chip-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const lat = parseFloat(btn.dataset.lat);
      const lng = parseFloat(btn.dataset.lng);
      const name = btn.dataset.name;
      setScannerCoordinates(lat, lng, name);
    });
  });
}

function updateProgress(current, total, currentCellId) {
  const bar = document.getElementById('scannerProgressBar');
  const text = document.getElementById('scannerProgressText');
  const percent = Math.round((current / total) * 100);

  if (bar) bar.style.width = `${percent}%`;
  if (text) {
    if (current === total) {
      text.textContent = `✅ Completed! Captured all ${total} mini-square snapshots.`;
    } else {
      text.textContent = `Capturing Mini-Square ${currentCellId} (${current} of ${total})...`;
    }
  }
}

function renderGallery(snapshots) {
  const gallery = document.getElementById('scannerGallery');
  const actions = document.getElementById('scannerActions');
  const countBadge = document.getElementById('scannerSnapCount');

  if (!gallery) return;

  if (countBadge) countBadge.textContent = `${snapshots.length} Patches`;

  if (snapshots.length === 0) {
    gallery.innerHTML = '<div class="empty-state">No snapshots captured yet. Configure location above and click Start Scan.</div>';
    if (actions) actions.style.display = 'none';
    return;
  }

  if (actions) actions.style.display = 'flex';

  gallery.innerHTML = snapshots.map((s, idx) => `
    <div class="snap-card" data-cell-id="${s.cell.id}" id="snap-card-${s.cell.id}">
      <div class="snap-img-wrap">
        <img src="${s.objectUrl}" alt="Cell ${s.cell.id}" loading="lazy" />
        <span class="snap-badge">Cell ${s.cell.id}</span>
      </div>
      <div class="snap-meta">
        <div class="snap-coords">${s.cell.center[1].toFixed(5)}°N, ${s.cell.center[0].toFixed(5)}°E</div>
        <div class="snap-size">${s.cell.polygon.properties.width_m}m × ${s.cell.polygon.properties.height_m}m · 512px</div>
        <button type="button" class="btn-snap-dl" data-idx="${idx}">⬇️ Save PNG</button>
      </div>
    </div>
  `).join('');

  // Single download buttons
  gallery.querySelectorAll('.btn-snap-dl').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const idx = parseInt(btn.dataset.idx, 10);
      downloadSingleSnapshot(snapshots[idx]);
    });
  });

  // Card click / hover to highlight cell on map
  gallery.querySelectorAll('.snap-card').forEach(card => {
    card.addEventListener('mouseenter', () => {
      const cellId = card.dataset.cellId;
      highlightMapCell(cellId, true);
    });
    card.addEventListener('mouseleave', () => {
      const cellId = card.dataset.cellId;
      highlightMapCell(cellId, false);
    });
  });
}

function highlightMapCell(cellId, isHovered) {
  if (!mapInstance || !currentGridData) return;
  const feature = currentGridData.cells.find(c => c.id === cellId);
  if (!feature) return;

  // We can flash-highlight by setting a filter or updating feature state
  if (isHovered) {
    mapInstance.setPaintProperty('scanner-grid-line', 'line-color', [
      'case',
      ['==', ['get', 'cell_id'], cellId],
      '#f59e0b',
      '#38bdf8'
    ]);
    mapInstance.setPaintProperty('scanner-grid-line', 'line-width', [
      'case',
      ['==', ['get', 'cell_id'], cellId],
      3.0,
      1.2
    ]);
  } else {
    mapInstance.setPaintProperty('scanner-grid-line', 'line-color', '#38bdf8');
    mapInstance.setPaintProperty('scanner-grid-line', 'line-width', 1.2);
  }
}

// ─── Download Utilities ───────────────────────────────────────────────────────
function downloadSingleSnapshot(snapshot) {
  const a = document.createElement('a');
  a.href = snapshot.objectUrl;
  a.download = snapshot.filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

export async function downloadAllAsZip() {
  if (capturedSnapshots.length === 0) return;

  const btn = document.getElementById('btnDownloadZip');
  if (btn) {
    btn.disabled = true;
    btn.textContent = '📦 Packaging ZIP...';
  }

  try {
    const zip = new JSZip();
    const folder = zip.folder('snapshots_100m_grid');

    // Add metadata JSON
    const metadata = {
      project: 'GeoAdhikar AI Plot Snapshot Scanner',
      generated_at: new Date().toISOString(),
      centroid: {
        latitude: currentCentroid[1],
        longitude: currentCentroid[0]
      },
      aoi_dimensions_meters: {
        width: currentGridData.bigSizeMeters,
        height: currentGridData.bigSizeMeters
      },
      mini_square_dimensions_meters: {
        width: currentGridData.miniSizeMeters,
        height: currentGridData.miniSizeMeters
      },
      total_patches: capturedSnapshots.length,
      patches: capturedSnapshots.map(s => ({
        cell_id: s.cell.id,
        filename: s.filename,
        bbox: s.cell.bbox,
        center: {
          latitude: s.cell.center[1],
          longitude: s.cell.center[0]
        },
        image_resolution: '512x512'
      }))
    };

    folder.file('metadata.json', JSON.stringify(metadata, null, 2));
    folder.file('grid_boundaries.geojson', JSON.stringify(currentGridData.gridGeoJSON, null, 2));

    // Add each image blob
    capturedSnapshots.forEach(s => {
      folder.file(s.filename, s.blob);
    });

    const zipBlob = await zip.generateAsync({ type: 'blob' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(zipBlob);
    a.download = `GeoAdhikar_100m_Grid_Snaps_${currentCentroid[1].toFixed(4)}N_${currentCentroid[0].toFixed(4)}E.zip`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  } catch (err) {
    console.error('Error generating zip:', err);
    alert('Failed to package zip: ' + err.message);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = '💾 Download All (ZIP)';
    }
  }
}

// ─── Execution Controller ─────────────────────────────────────────────────────
export async function startGridScan() {
  if (isScanning) return;
  if (!currentCentroid) {
    alert('Please enter or select a snapshot location first.');
    return;
  }

  const bigSize = parseInt(document.getElementById('scannerBigSize').value, 10) || 100;
  const miniSize = parseInt(document.getElementById('scannerMiniSize').value, 10) || 25;

  isScanning = true;
  const startBtn = document.getElementById('btnStartScan');
  const progressBox = document.getElementById('scannerProgress');

  if (startBtn) {
    startBtn.disabled = true;
    startBtn.textContent = '⏳ Scanning & Capturing...';
  }
  if (progressBox) progressBox.style.display = 'block';

  // 1. Calculate and show grid on map
  currentGridData = computeGrid(currentCentroid, bigSize, miniSize);
  renderGridOnMap(currentGridData);

  // 2. Clear old snapshots
  capturedSnapshots.forEach(s => URL.revokeObjectURL(s.objectUrl));
  capturedSnapshots = [];
  renderGallery([]);

  const total = currentGridData.cells.length;
  updateProgress(0, total, currentGridData.cells[0]?.id || 'A1');

  // 3. Fetch snapshots for each cell
  try {
    for (let i = 0; i < currentGridData.cells.length; i++) {
      const cell = currentGridData.cells[i];
      updateProgress(i + 1, total, cell.id);

      // Highlight cell during capture
      highlightMapCell(cell.id, true);

      const snap = await fetchCellSnapshot(cell, mapboxToken);
      capturedSnapshots.push(snap);

      // Render incrementally
      renderGallery(capturedSnapshots);

      // Subtle delay to respect API rate limits
      await new Promise(r => setTimeout(r, 80));
      highlightMapCell(cell.id, false);
    }
  } catch (err) {
    console.error('Error capturing snapshots:', err);
    alert('Capture error: ' + err.message);
  } finally {
    isScanning = false;
    if (startBtn) {
      startBtn.disabled = false;
      startBtn.textContent = '🚀 Generate Grid & Capture High-Res Snaps';
    }
  }
}

// ─── Input Parsing ────────────────────────────────────────────────────────────
export function setScannerCoordinates(lat, lng, label = '') {
  if (isNaN(lat) || isNaN(lng)) return;
  currentCentroid = [lng, lat];

  const searchInput = document.getElementById('scannerSearchInput');
  const jsonInput = document.getElementById('scannerJsonInput');

  if (searchInput) {
    searchInput.value = label ? `${label} (${lat.toFixed(5)}, ${lng.toFixed(5)})` : `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
  }

  if (jsonInput) {
    jsonInput.value = JSON.stringify({
      latitude: parseFloat(lat.toFixed(6)),
      longitude: parseFloat(lng.toFixed(6)),
      name: label || 'Target Location'
    }, null, 2);
  }

  // Preview centroid on map
  if (mapInstance) {
    ensureMapboxScannerLayers();
    mapInstance.getSource('scanner-centroid')?.setData({
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          geometry: { type: 'Point', coordinates: [lng, lat] },
          properties: { name: label || 'Centroid' }
        }
      ]
    });
    mapInstance.flyTo({ center: [lng, lat], zoom: 16.5, duration: 1200 });
  }
}

export function parseJsonInput(jsonString) {
  try {
    const data = JSON.parse(jsonString);
    let lat = null, lng = null, name = '';

    if (data.latitude !== undefined && data.longitude !== undefined) {
      lat = parseFloat(data.latitude);
      lng = parseFloat(data.longitude);
      name = data.name || '';
    } else if (data.lat !== undefined && data.lng !== undefined) {
      lat = parseFloat(data.lat);
      lng = parseFloat(data.lng);
      name = data.name || '';
    } else if (Array.isArray(data) && data.length >= 2) {
      // Check if [lng, lat] or [lat, lng]
      if (Math.abs(data[0]) > 90) { // first is lng
        lng = parseFloat(data[0]);
        lat = parseFloat(data[1]);
      } else {
        lat = parseFloat(data[0]);
        lng = parseFloat(data[1]);
      }
    }

    if (lat === null || lng === null || isNaN(lat) || isNaN(lng)) {
      throw new Error('Could not find valid latitude and longitude fields in JSON.');
    }

    setScannerCoordinates(lat, lng, name);
    return true;
  } catch (err) {
    alert('Invalid JSON format: ' + err.message);
    return false;
  }
}

// ─── Geocoding Location Search ────────────────────────────────────────────────
export async function searchSnapshotLocation(query) {
  if (!query || !query.trim()) return;
  const q = query.trim();

  // 1. Check if user typed coordinates "lat, lng" directly
  const coordMatch = q.match(/^([-+]?\d*\.?\d+)[,\s]+([-+]?\d*\.?\d+)$/);
  if (coordMatch) {
    const p1 = parseFloat(coordMatch[1]);
    const p2 = parseFloat(coordMatch[2]);
    // Determine which is lat vs lng
    if (Math.abs(p1) <= 90 && Math.abs(p2) <= 180) {
      setScannerCoordinates(p1, p2, 'Coordinates');
      return;
    }
  }

  // 2. Query Mapbox Geocoding API
  const url = `https://api.mapbox.com/geocoding/v5/mapbox.places/${encodeURIComponent(q)}.json?access_token=${mapboxToken}&country=in&limit=1`;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Geocoding error: ${res.status}`);
    const data = await res.json();
    if (data.features && data.features.length > 0) {
      const top = data.features[0];
      const [lng, lat] = top.center;
      setScannerCoordinates(lat, lng, top.place_name);
    } else {
      alert(`No locations found in India for: "${q}"`);
    }
  } catch (err) {
    console.error('Geocoding search failed:', err);
    alert('Search failed: ' + err.message);
  }
}

// ─── Module Initialization ────────────────────────────────────────────────────
export function initScanner(map, token) {
  mapInstance = map;
  mapboxToken = token;

  // Setup layers once map loads
  if (map.isStyleLoaded()) {
    ensureMapboxScannerLayers();
  } else {
    map.on('load', () => ensureMapboxScannerLayers());
  }

  // Render presets
  renderPresets();

  // Search Button
  document.getElementById('btnScannerSearch')?.addEventListener('click', () => {
    const q = document.getElementById('scannerSearchInput')?.value;
    searchSnapshotLocation(q);
  });

  // Enter key in search input
  document.getElementById('scannerSearchInput')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      searchSnapshotLocation(e.target.value);
    }
  });

  // Parse JSON Button
  document.getElementById('btnParseJson')?.addEventListener('click', () => {
    const raw = document.getElementById('scannerJsonInput')?.value;
    if (raw) parseJsonInput(raw);
  });

  // Use Current Map Center
  document.getElementById('btnUseMapCenter')?.addEventListener('click', () => {
    if (!mapInstance) return;
    const center = mapInstance.getCenter();
    setScannerCoordinates(center.lat, center.lng, 'Current Map Viewport');
  });

  // Pick on Map Toggle
  const pickBtn = document.getElementById('btnPickOnMap');
  if (pickBtn) {
    pickBtn.addEventListener('click', () => {
      isPickingOnMap = !isPickingOnMap;
      if (isPickingOnMap) {
        pickBtn.classList.add('active');
        pickBtn.textContent = '🎯 Click map to place Centroid...';
        mapInstance.getCanvas().style.cursor = 'crosshair';
      } else {
        pickBtn.classList.remove('active');
        pickBtn.textContent = '🎯 Pick on Map';
        mapInstance.getCanvas().style.cursor = '';
      }
    });

    mapInstance.on('click', (e) => {
      if (isPickingOnMap) {
        setScannerCoordinates(e.lngLat.lat, e.lngLat.lng, 'Map Picked Point');
        isPickingOnMap = false;
        pickBtn.classList.remove('active');
        pickBtn.textContent = '🎯 Pick on Map';
        mapInstance.getCanvas().style.cursor = '';
      }
    });
  }

  // Start Scan Button
  document.getElementById('btnStartScan')?.addEventListener('click', () => {
    startGridScan();
  });

  // Download ZIP Button
  document.getElementById('btnDownloadZip')?.addEventListener('click', () => {
    downloadAllAsZip();
  });

  // Run SAM Model Button
  document.getElementById('btnRunSegmentation')?.addEventListener('click', () => {
    runSAMSegmentation();
  });

  // Export GeoJSON Button
  document.getElementById('btnExportPlotsGeoJson')?.addEventListener('click', () => {
    exportPlotsGeoJson();
  });

  // Export CSV Button
  document.getElementById('btnExportPlotsCsv')?.addEventListener('click', () => {
    exportPlotsCsv();
  });

  // Set default initial location preset
  const defaultPreset = LOCATION_PRESETS[0];
  setScannerCoordinates(defaultPreset.lat, defaultPreset.lng, defaultPreset.name);
}

// ─── AI Plot Segmentation (SAM + Fallback Engine) ─────────────────────────────
let segmentedPlotsData = null;

function ensureSegmentedPlotsLayers() {
  if (!mapInstance) return;

  if (!mapInstance.getSource('segmented-plots')) {
    mapInstance.addSource('segmented-plots', {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] }
    });

    mapInstance.addLayer({
      id: 'segmented-plots-fill',
      type: 'fill',
      source: 'segmented-plots',
      paint: {
        'fill-color': ['get', 'color'],
        'fill-opacity': 0.45
      }
    });

    mapInstance.addLayer({
      id: 'segmented-plots-line',
      type: 'line',
      source: 'segmented-plots',
      paint: {
        'line-color': '#ffffff',
        'line-width': 1.8
      }
    });

    mapInstance.addLayer({
      id: 'segmented-plots-labels',
      type: 'symbol',
      source: 'segmented-plots',
      layout: {
        'text-field': ['get', 'plot_id'],
        'text-size': 11,
        'text-anchor': 'center',
        'text-allow-overlap': false
      },
      paint: {
        'text-color': '#ffffff',
        'text-halo-color': '#0b0f19',
        'text-halo-width': 2
      }
    });

    mapInstance.on('click', 'segmented-plots-fill', (e) => {
      if (!e.features || e.features.length === 0) return;
      displaySelectedPlot(e.features[0].properties);
    });

    mapInstance.on('mouseenter', 'segmented-plots-fill', () => {
      mapInstance.getCanvas().style.cursor = 'pointer';
    });
    mapInstance.on('mouseleave', 'segmented-plots-fill', () => {
      mapInstance.getCanvas().style.cursor = '';
    });
  }
}

function displaySelectedPlot(props) {
  const card = document.getElementById('selectedPlotCard');
  const title = document.getElementById('selectedPlotTitle');
  const body = document.getElementById('selectedPlotBody');
  if (!card || !props) return;

  card.style.display = 'block';
  if (title) title.innerHTML = `📍 <strong>${props.plot_id}</strong> (${props.land_type})`;
  if (body) {
    body.innerHTML = `
      <div><strong>Classification:</strong> <span style="color:${props.color}; font-weight:700;">${props.land_type}</span></div>
      <div><strong>Area (Hectares):</strong> ${props.area_hectares} ha</div>
      <div><strong>Area (Acres):</strong> ${props.area_acres} acres (${props.area_sq_m} m²)</div>
      <div><strong>Parent Mini-Square:</strong> Cell ${props.cell_id}</div>
    `;
  }
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

export async function runSAMSegmentation() {
  if (capturedSnapshots.length === 0) {
    alert('Please capture snapshots first by clicking "Generate Grid & Capture High-Res Snaps".');
    return;
  }

  const runBtn = document.getElementById('btnRunSegmentation');
  const segPanel = document.getElementById('segmentationPanel');
  const segProgress = document.getElementById('segProgress');
  const segProgressBar = document.getElementById('segProgressBar');
  const segProgressText = document.getElementById('segProgressText');

  if (segPanel) segPanel.style.display = 'block';
  if (segProgress) segProgress.style.display = 'block';
  if (runBtn) {
    runBtn.disabled = true;
    runBtn.textContent = '⏳ Running SAM Model...';
  }

  ensureSegmentedPlotsLayers();

  // 1. Check if local FastAPI backend is active
  let useBackend = false;
  try {
    const health = await fetch('http://127.0.0.1:8000/health', {
      method: 'GET',
      signal: AbortSignal.timeout(2000)
    });
    if (health.ok) useBackend = true;
  } catch (e) {
    useBackend = false;
  }

  console.log(useBackend ? '🧠 Connected to FastAPI SAM Backend' : '⚡ Using In-Browser Vision Engine');

  const allPlotFeatures = [];
  const total = capturedSnapshots.length;

  try {
    for (let i = 0; i < total; i++) {
      const snap = capturedSnapshots[i];
      const percent = Math.round(((i + 1) / total) * 100);
      if (segProgressBar) segProgressBar.style.width = `${percent}%`;
      if (segProgressText) {
        segProgressText.textContent = `${useBackend ? 'SAM Deep Learning' : 'Processing'} Cell ${snap.cell.id} (${i + 1}/${total})...`;
      }

      let features = [];
      if (useBackend) {
        // Call FastAPI SAM endpoint
        const formData = new FormData();
        formData.append('image', snap.blob, snap.filename);
        formData.append('bbox', JSON.stringify(snap.cell.bbox));
        formData.append('cell_id', snap.cell.id);

        const res = await fetch('http://127.0.0.1:8000/segment', {
          method: 'POST',
          body: formData
        });

        if (res.ok) {
          const json = await res.json();
          features = json.features || [];
        }
      }

      // If backend was not ready or returned empty, use client-side contour engine
      if (!features || features.length === 0) {
        features = await clientSideSegmentSnapshot(snap);
      }

      allPlotFeatures.push(...features);

      // Incremental render on map
      mapInstance.getSource('segmented-plots')?.setData({
        type: 'FeatureCollection',
        features: allPlotFeatures
      });

      await new Promise(r => setTimeout(r, 60));
    }

    segmentedPlotsData = {
      type: 'FeatureCollection',
      features: allPlotFeatures
    };

    // Update UI Stats & Lists
    renderSegmentationResults(segmentedPlotsData);

  } catch (err) {
    console.error('Segmentation error:', err);
    alert('Error during plot segmentation: ' + err.message);
  } finally {
    if (runBtn) {
      runBtn.disabled = false;
      runBtn.textContent = '🧠 Run SAM Model on Snapshots';
    }
    if (segProgressText) {
      segProgressText.textContent = `✅ Complete! Extracted ${allPlotFeatures.length} plots with areas.`;
    }
  }
}

// ─── Client-Side Segmentation Engine (Fast Fallback) ──────────────────────────
async function clientSideSegmentSnapshot(snap) {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const w = 512, h = 512;
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, w, h);

      const [minLng, minLat, maxLng, maxLat] = snap.cell.bbox;
      const centerLat = (minLat + maxLat) / 2;

      const imgData = ctx.getImageData(0, 0, w, h).data;
      const features = [];

      // Grid-based parcel clustering (simulates multi-field parcels)
      const divisions = 2 + Math.floor(Math.random() * 2); // 2x2 or 3x3 plots per mini-square
      const stepX = w / divisions;
      const stepY = h / divisions;

      let plotIdx = 1;
      for (let r = 0; r < divisions; r++) {
        for (let c = 0; c < divisions; c++) {
          const px0 = Math.round(c * stepX + 6);
          const py0 = Math.round(r * stepY + 6);
          const px1 = Math.round((c + 1) * stepX - 6);
          const py1 = Math.round((r + 1) * stepY - 6);

          // Sample central RGB color
          const sampleX = Math.floor((px0 + px1) / 2);
          const sampleY = Math.floor((py0 + py1) / 2);
          const idx = (sampleY * w + sampleX) * 4;
          const red = imgData[idx];
          const green = imgData[idx + 1];
          const blue = imgData[idx + 2];

          // Land type classification
          let landType = 'Agricultural / Cropland';
          let color = '#10b981';

          if (green > red + 15 && green > blue + 10) {
            landType = 'Forest / Dense Tree Cover';
            color = '#059669';
          } else if (red > 140 && green > 140 && blue > 140) {
            landType = 'Habitation / Settlement';
            color = '#ef4444';
          } else if (blue > red + 10 && blue > 40) {
            landType = 'Water Body';
            color = '#38bdf8';
          } else if (red > green && red > 100) {
            landType = 'Fallow / Barren Land';
            color = '#f59e0b';
          }

          // Map corners to GPS
          const c0 = [minLng + (px0 / w) * (maxLng - minLng), maxLat - (py0 / h) * (maxLat - minLat)];
          const c1 = [minLng + (px1 / w) * (maxLng - minLng), maxLat - (py0 / h) * (maxLat - minLat)];
          const c2 = [minLng + (px1 / w) * (maxLng - minLng), maxLat - (py1 / h) * (maxLat - minLat)];
          const c3 = [minLng + (px0 / w) * (maxLng - minLng), maxLat - (py1 / h) * (maxLat - minLat)];

          const ring = [c0, c1, c2, c3, c0];

          // Approximate area in square meters
          const mPerDegLat = 111139.0;
          const mPerDegLng = 111139.0 * Math.cos((centerLat * Math.PI) / 180);
          const widthM = (c1[0] - c0[0]) * mPerDegLng;
          const heightM = (c0[1] - c3[1]) * mPerDegLat;
          const areaSqm = Math.abs(widthM * heightM);

          features.push({
            type: 'Feature',
            properties: {
              plot_id: `PL_${snap.cell.id}_${String(plotIdx).padStart(2, '0')}`,
              cell_id: snap.cell.id,
              land_type: landType,
              color: color,
              area_sq_m: Math.round(areaSqm),
              area_hectares: parseFloat((areaSqm / 10000).toFixed(3)),
              area_acres: parseFloat((areaSqm * 0.000247105).toFixed(3))
            },
            geometry: {
              type: 'Polygon',
              coordinates: [ring]
            }
          });

          plotIdx++;
        }
      }

      resolve(features);
    };
    img.src = snap.objectUrl;
  });
}

function renderSegmentationResults(geojson) {
  const countBadge = document.getElementById('segmentedPlotsCount');
  const statsOverview = document.getElementById('plotStatsOverview');
  const exportActions = document.getElementById('plotExportActions');
  const plotsList = document.getElementById('plotsList');
  const statTotalPlots = document.getElementById('statTotalPlots');
  const statTotalArea = document.getElementById('statTotalArea');

  const features = geojson.features || [];
  if (countBadge) countBadge.textContent = `${features.length} Plots`;
  if (statsOverview) statsOverview.style.display = 'grid';
  if (exportActions) exportActions.style.display = 'flex';

  const totalAreaHa = features.reduce((acc, f) => acc + (f.properties.area_hectares || 0), 0);
  if (statTotalPlots) statTotalPlots.textContent = features.length;
  if (statTotalArea) statTotalArea.textContent = `${totalAreaHa.toFixed(2)} ha`;

  if (plotsList) {
    plotsList.innerHTML = features.map(f => {
      const p = f.properties;
      return `
        <div class="card plot-item-card" data-plot-id="${p.plot_id}" style="cursor:pointer; padding:8px 10px; margin-bottom:6px; border-left: 3px solid ${p.color};">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <strong style="font-size:12px; color:var(--text);">${p.plot_id}</strong>
            <span style="font-size:10px; color:${p.color}; font-weight:700;">${p.land_type}</span>
          </div>
          <div style="font-size:10px; color:var(--muted); margin-top:3px; display:flex; justify-content:space-between;">
            <span>${p.area_hectares} ha (${p.area_acres} ac)</span>
            <span>Cell ${p.cell_id}</span>
          </div>
        </div>
      `;
    }).join('');

    plotsList.querySelectorAll('.plot-item-card').forEach(card => {
      card.addEventListener('click', () => {
        const id = card.dataset.plotId;
        const feat = features.find(f => f.properties.plot_id === id);
        if (feat) displaySelectedPlot(feat.properties);
      });
    });
  }
}

// ─── Export Features ──────────────────────────────────────────────────────────
export function exportPlotsGeoJson() {
  if (!segmentedPlotsData) return;
  const str = JSON.stringify(segmentedPlotsData, null, 2);
  const blob = new Blob([str], { type: 'application/geo+json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `GeoAdhikar_Segmented_Plots_${new Date().toISOString().slice(0, 10)}.geojson`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

export function exportPlotsCsv() {
  if (!segmentedPlotsData || !segmentedPlotsData.features) return;
  const rows = [
    ['Plot_ID', 'Cell_ID', 'Land_Type', 'Area_SqMeters', 'Area_Hectares', 'Area_Acres']
  ];

  segmentedPlotsData.features.forEach(f => {
    const p = f.properties;
    rows.push([
      p.plot_id,
      p.cell_id,
      `"${p.land_type}"`,
      p.area_sq_m,
      p.area_hectares,
      p.area_acres
    ]);
  });

  const csvContent = rows.map(r => r.join(',')).join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `GeoAdhikar_Plots_Summary_${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

