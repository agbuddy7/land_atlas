import { jsPDF } from 'jspdf';
import html2canvas from 'html2canvas';
import { initScanner } from './scanner.js';

// ─── Config ─────────────────────────────────────────────────────────────────
const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN || '';

// ─── Helpers ─────────────────────────────────────────────────────────────────
const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
const fmtNum = (v) => Number(v).toLocaleString('en-IN');
const $ = (id) => document.getElementById(id);

// ─── Global State ─────────────────────────────────────────────────────────────
const state = {
  blocks: [],
  schemes: [],
  boundaries: null,
  map: null,
  selectedBlock: null,
  lastRecommendations: [],
};

// ─── Fetch Helper ─────────────────────────────────────────────────────────────
async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  return res.json();
}

// ─── Map Initialization ───────────────────────────────────────────────────────
function initMap() {
  mapboxgl.accessToken = MAPBOX_TOKEN;

  const map = new mapboxgl.Map({
    container: 'map',
    style: 'mapbox://styles/mapbox/satellite-streets-v12',
    center: [79.5, 22.8],
    zoom: 5,
    pitch: 45,
    bearing: -10,
    preserveDrawingBuffer: true,
  });

  map.addControl(new mapboxgl.NavigationControl(), 'bottom-right');
  map.addControl(new mapboxgl.ScaleControl({ maxWidth: 120, unit: 'metric' }), 'bottom-left');

  // Mapbox Geocoder (Global Search Bar)
  initGeocoder(map);

  map.on('load', () => {
    // 3D terrain
    map.addSource('mapbox-dem', {
      type: 'raster-dem',
      url: 'mapbox://mapbox.mapbox-terrain-dem-v1',
      tileSize: 512,
    });
    map.setTerrain({ source: 'mapbox-dem', exaggeration: 1.5 });

    // Boundary source + layers
    map.addSource('boundary', {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] },
    });

    map.addLayer({
      id: 'boundary-fill',
      type: 'fill',
      source: 'boundary',
      paint: { 'fill-color': '#3b82f6', 'fill-opacity': 0.25 },
    });

    map.addLayer({
      id: 'boundary-line',
      type: 'line',
      source: 'boundary',
      paint: { 'line-color': '#2563eb', 'line-width': 2.5, 'line-dasharray': [2, 1] },
    });

    // Hide map loading overlay
    const overlay = $('mapLoadingOverlay');
    if (overlay) overlay.classList.add('hidden');
  });

  return map;
}

// ─── Dropdown Helpers ─────────────────────────────────────────────────────────
function unique(arr, key) {
  const set = new Set();
  const out = [];
  for (const item of arr) {
    const val = item[key];
    if (!set.has(val)) { set.add(val); out.push(val); }
  }
  return out.sort((a, b) => a.localeCompare(b));
}

function populateStateSelect(blocks) {
  const states = unique(blocks, 'state');
  const el = $('stateSelect');
  el.innerHTML = '<option value="">Select State</option>' +
    states.map(s => `<option value="${s}">${s}</option>`).join('');
}

function populateDistrictSelect(blocks, stateName) {
  const el = $('districtSelect');
  const filtered = blocks.filter(b => b.state === stateName);
  const dists = unique(filtered, 'district');
  el.innerHTML = '<option value="">Select District</option>' +
    dists.map(d => `<option value="${d}">${d}</option>`).join('');
  el.disabled = dists.length === 0;
  const blockEl = $('blockSelect');
  blockEl.innerHTML = '<option value="">Select Area</option>';
  blockEl.disabled = true;
}

function populateBlockSelect(blocks, stateName, district) {
  const el = $('blockSelect');
  const filtered = blocks.filter(b => b.state === stateName && b.district === district);
  el.innerHTML = '<option value="">Select Area</option>' +
    filtered.map(b => `<option value="${b.block_id}">${b.block_name}</option>`).join('');
  el.disabled = filtered.length === 0;
}

// ─── Map Boundary ─────────────────────────────────────────────────────────────
function renderBoundaryOnMap(feature) {
  if (!state.map || !state.map.getSource('boundary')) return;
  if (!feature) {
    state.map.getSource('boundary').setData({ type: 'FeatureCollection', features: [] });
    return;
  }
  state.map.getSource('boundary').setData({
    type: 'FeatureCollection',
    features: [feature]
  });
  try {
    const allCoords = [];
    const collectCoords = (coords) => {
      if (Array.isArray(coords[0])) coords.forEach(collectCoords);
      else allCoords.push(coords);
    };
    collectCoords(feature.geometry.coordinates);
    const bounds = allCoords.reduce(
      (b, c) => b.extend(c),
      new mapboxgl.LngLatBounds(allCoords[0], allCoords[0])
    );
    state.map.fitBounds(bounds, { padding: 60, maxZoom: 13, duration: 1200 });
  } catch (e) { /* ignore */ }
}

// ─── Profile Render ───────────────────────────────────────────────────────────
function renderProfile(profile) {
  const list = $('profileList');
  const btn = $('downloadPdfBtn');

  const metrics = [
    { label: 'Population', val: fmtNum(profile.population), isPercent: false },
    { label: 'Area', val: `${fmtNum(profile.area_sq_km)} sq km`, isPercent: false },
    { label: 'FRA Claims Filed', val: fmtNum(profile.fra_claims), isPercent: false },
    
    { label: 'Tribal Population', val: profile.tribal_population_percent, isPercent: true, color: 'primary' },
    { label: 'Forest Cover', val: profile.forest_cover_percent, isPercent: true, color: 'accent' },
    { label: 'Agricultural Land', val: profile.agri_land_percent, isPercent: true, color: 'warn' },
    { label: 'Households with Tap Water', val: profile.households_with_water, isPercent: true, color: 'primary' },
    { label: 'Villages ODF+ Status', val: profile.villages_odf_plus, isPercent: true, color: 'accent' }
  ];

  list.innerHTML = metrics.map((m, idx) => {
    if (m.isPercent) {
      return `
        <div class="card">
          <div class="meta">${m.label}</div>
          <div class="gauge-wrap">
            <div class="gauge-header">
              <span>MT_${String(idx + 1).padStart(2, '0')}</span>
              <span>${m.val.toFixed(1)}%</span>
            </div>
            <div class="gauge-bar">
              <div class="gauge-fill ${m.color}" style="width: ${m.val}%;"></div>
            </div>
          </div>
        </div>`;
    } else {
      return `
        <div class="card">
          <div class="meta">${m.label}</div>
          <div class="value-row">
            <span class="value-num">${m.val}</span>
          </div>
        </div>`;
    }
  }).join('');

  btn.disabled = false;
}

// ─── Challenge Detection ──────────────────────────────────────────────────────
function detectChallenges(profile) {
  const ch = [];
  if (profile.tribal_population_percent >= 30) ch.push('High tribal population — development gaps likely');
  if (profile.households_with_water < 90) ch.push('Tap water coverage gap — JJM priority area');
  if (profile.villages_odf_plus < 80) ch.push('Sanitation (ODF+) gap — SBM-G intervention needed');
  if (profile.forest_cover_percent >= 20 && profile.tribal_population_percent >= 30)
    ch.push('Forest-fringe land rights — FRA implementation critical');
  if (profile.agri_land_percent < 20 && profile.tribal_population_percent >= 30)
    ch.push('Low agricultural land — livelihood diversification needed');
  if (profile.fra_claims > 300) ch.push('High FRA claim backlog — expedited processing required');
  return ch;
}

function renderChallenges(ch) {
  const list = $('challengesList');
  if (!ch || ch.length === 0) {
    list.innerHTML = '<li style="background:#f0fdf4;border-color:#bbf7d0;color:#166534;">No critical challenges detected — area performing well</li>';
    return;
  }
  list.innerHTML = ch.map(t => `<li>⚠️ ${t}</li>`).join('');
}

// ─── Scoring Engine ───────────────────────────────────────────────────────────
function isConditionTrue(fieldVal, op, value) {
  switch (op) {
    case '<':  return fieldVal < value;
    case '<=': return fieldVal <= value;
    case '>':  return fieldVal > value;
    case '>=': return fieldVal >= value;
    case '==': return fieldVal == value; // eslint-disable-line eqeqeq
    default:   return false;
  }
}

function computeSchemeScore(profile, scheme) {
  let eligible = true;
  let sum = 0;
  const rulesOut = [];

  for (const rule of scheme.rules) {
    const v = profile[rule.field];
    let raw = 0;
    let passed = false;
    let sentence = '';

    if (rule.type === 'eligibility') {
      passed = isConditionTrue(v, rule.op, rule.value);
      sentence = `${rule.field.replace(/_/g, ' ')} is ${v} ${rule.op} ${rule.value} → ${passed ? '✅ Eligible' : '❌ Ineligible'}`;
      if (!passed) {
        eligible = false;
        rulesOut.push({ field: rule.field, value: v, op: rule.op, raw_score: 0, contribution: 0, passed, sentence, weight: rule.weight });
        return { eligible: false, score: 0, explanation: rulesOut };
      }
    } else if (rule.type === 'need') {
      if (rule.op === '<') {
        const min = rule.scaling?.min ?? 0;
        const denom = rule.value - min;
        raw = denom > 0 ? (rule.value - v) / denom : 0;
        raw = clamp(raw, 0, 1);
        passed = v < rule.value;
        sentence = `${rule.field.replace(/_/g, ' ')} = ${v}% (threshold ${rule.value}%) → need score ${(raw * 100).toFixed(0)}%`;
      } else if (rule.op === '>=') {
        const max = rule.scaling?.max ?? 100;
        const denom = max - rule.value;
        raw = denom > 0 ? (v - rule.value) / denom : 0;
        raw = clamp(raw, 0, 1);
        passed = v >= rule.value;
        sentence = `${rule.field.replace(/_/g, ' ')} = ${v}% (threshold ${rule.value}%) → need score ${(raw * 100).toFixed(0)}%`;
      }
    } else if (rule.type === 'modifier') {
      passed = isConditionTrue(v, rule.op, rule.value);
      raw = passed ? 1 : 0;
      sentence = `${rule.field.replace(/_/g, ' ')} = ${v} ${rule.op} ${rule.value} → modifier ${passed ? 'applied ✅' : 'not applied'}`;
    }

    const contribution = raw * (rule.weight ?? 0);
    sum += contribution;
    rulesOut.push({ field: rule.field, value: v, op: rule.op, raw_score: raw, contribution, passed, sentence, weight: rule.weight });
  }

  sum = clamp(sum, 0, 1);
  const scorePct = Math.round(sum * 1000) / 10;
  return { eligible, score: scorePct, explanation: rulesOut };
}

function scoreAllSchemes(profile, schemes) {
  return schemes
    .map(s => {
      const r = computeSchemeScore(profile, s);
      return { scheme: s, eligible: r.eligible, score: r.score, explanation: { rules: r.explanation } };
    })
    .sort((a, b) => (b.eligible - a.eligible) || (b.score - a.score));
}

function renderRecommendations(results) {
  const wrap = $('recommendations');
  if (!results || results.length === 0) {
    wrap.innerHTML = '<div class="card">No schemes to display.</div>';
    return;
  }
  const top = results.slice(0, 5);
  wrap.innerHTML = top.map((r, i) => {
    const scoreClass = r.eligible ? '' : 'ineligible';
    const scoreText = r.eligible ? `${r.score.toFixed(1)}%` : 'Ineligible';
    const rankLabel = `PR_${i + 1}`;
    const fillWidth = r.eligible ? r.score : 0;
    
    return `
      <div class="scheme-card">
        <div class="scheme-head">
          <div style="display:flex;align-items:center;gap:8px;">
            <span style="font-family:monospace;font-size:9px;background:var(--panel-light);color:var(--muted);padding:2px 6px;border-radius:3px;font-weight:700;">${rankLabel}</span>
            <div class="scheme-name">${r.scheme.name}</div>
          </div>
          <div class="scheme-score ${scoreClass}">${scoreText}</div>
        </div>
        <div class="scheme-benefit">${r.scheme.benefit}</div>
        <div class="scheme-progress">
          <div class="scheme-progress-fill" style="width: ${fillWidth}%;"></div>
        </div>
      </div>`;
  }).join('');
}

// ─── Interactions ─────────────────────────────────────────────────────────────
function clearPanels() {
  $('recommendations').innerHTML = '';
  $('profileList').innerHTML = '';
  $('challengesList').innerHTML = '';
  $('downloadPdfBtn').disabled = true;
  renderBoundaryOnMap(null);
  state.selectedBlock = null;
  state.lastRecommendations = [];
}

function getBoundaryByBlockId(blockId) {
  if (!state.boundaries) return null;
  return state.boundaries.features.find(f => f.properties?.block_id === blockId) || null;
}

function selectBlockById(id) {
  const profile = state.blocks.find(b => b.block_id === id);
  if (!profile) return;

  state.selectedBlock = profile;

  // Sync cascading dropdowns
  const stateSel = $('stateSelect');
  const distSel = $('districtSelect');
  const blockSel = $('blockSelect');

  if (stateSel.value !== profile.state) {
    stateSel.value = profile.state;
    populateDistrictSelect(state.blocks, profile.state);
  }
  if (distSel.value !== profile.district) {
    distSel.value = profile.district;
    populateBlockSelect(state.blocks, profile.state, profile.district);
  }
  blockSel.value = profile.block_id;

  // Map Boundary
  renderBoundaryOnMap(getBoundaryByBlockId(id));

  // Profile & Challenges
  renderProfile(profile);
  renderChallenges(detectChallenges(profile));

  // Scoring
  const results = scoreAllSchemes(profile, state.schemes);
  state.lastRecommendations = results;
  renderRecommendations(results);

  $('downloadPdfBtn').disabled = false;
}

// ─── Mapbox Geocoder (Global & Local Search) ──────────────────────────────────
function localGeocoderSearch(query) {
  if (!query || !state.blocks || state.blocks.length === 0) return [];
  const q = query.toLowerCase().trim();
  const matches = [];

  for (const b of state.blocks) {
    if (
      b.block_name.toLowerCase().includes(q) ||
      b.district.toLowerCase().includes(q) ||
      b.state.toLowerCase().includes(q)
    ) {
      const boundary = getBoundaryByBlockId(b.block_id);
      let center = [78.9629, 20.5937];
      if (boundary && boundary.properties && boundary.properties.center_lng && boundary.properties.center_lat) {
        center = [boundary.properties.center_lng, boundary.properties.center_lat];
      }
      matches.push({
        id: b.block_id,
        type: 'Feature',
        text: b.block_name,
        place_name: `📍 ${b.block_name}, ${b.district}, ${b.state} (Tribal Profile)`,
        place_type: ['place'],
        center: center,
        geometry: {
          type: 'Point',
          coordinates: center,
        },
        properties: {
          block_id: b.block_id,
          isGeoAdhikarBlock: true,
        },
      });
    }
  }
  return matches.slice(0, 5);
}

function initGeocoder(map) {
  if (!window.MapboxGeocoder) {
    console.warn('MapboxGeocoder plugin not loaded yet.');
    return;
  }

  const geocoder = new window.MapboxGeocoder({
    accessToken: mapboxgl.accessToken,
    mapboxgl: mapboxgl,
    marker: {
      color: '#3b82f6',
    },
    placeholder: 'Search any city, district, landmark in India...',
    countries: 'in',
    localGeocoder: localGeocoderSearch,
    localGeocoderOnly: false,
    zoom: 12,
  });

  map.addControl(geocoder, 'top-left');

  geocoder.on('result', (e) => {
    const result = e.result;
    if (result.properties?.block_id) {
      selectBlockById(result.properties.block_id);
    } else {
      const q = (result.text || result.place_name || '').toLowerCase();
      const matched = state.blocks.find(b =>
        q.includes(b.block_name.toLowerCase()) ||
        b.block_name.toLowerCase().includes(q) ||
        q.includes(b.district.toLowerCase())
      );
      if (matched) {
        selectBlockById(matched.block_id);
      }
    }
  });
}

function setupInteractions() {
  const stateSel = $('stateSelect');
  const distSel = $('districtSelect');
  const blockSel = $('blockSelect');
  const downloadBtn = $('downloadPdfBtn');

  stateSel.addEventListener('change', () => {
    populateDistrictSelect(state.blocks, stateSel.value);
    clearPanels();
  });

  distSel.addEventListener('change', () => {
    populateBlockSelect(state.blocks, stateSel.value, distSel.value);
    clearPanels();
  });

  blockSel.addEventListener('change', () => {
    const id = blockSel.value;
    if (!id) { clearPanels(); return; }
    selectBlockById(id);
  });

  // PDF download
  downloadBtn.addEventListener('click', async () => {
    if (!state.selectedBlock) return;
    downloadBtn.disabled = true;
    downloadBtn.textContent = 'Generating...';
    try {
      const doc = await buildPdf(state.selectedBlock, state.lastRecommendations);
      doc.save(`GeoAdhikar_${state.selectedBlock.block_id}_Report.pdf`);
    } finally {
      downloadBtn.disabled = false;
      downloadBtn.textContent = 'Download Report';
    }
  });

  // Mode Tabs Switching
  const tabDSS = $('tabDSS');
  const tabScanner = $('tabScanner');
  const dssContainer = $('dssContainer');
  const scannerContainer = $('scannerContainer');

  if (tabDSS && tabScanner && dssContainer && scannerContainer) {
    tabDSS.addEventListener('click', () => {
      tabDSS.classList.add('active');
      tabScanner.classList.remove('active');
      dssContainer.style.display = 'block';
      scannerContainer.style.display = 'none';
    });

    tabScanner.addEventListener('click', () => {
      tabScanner.classList.add('active');
      tabDSS.classList.remove('active');
      dssContainer.style.display = 'none';
      scannerContainer.style.display = 'block';
    });
  }
}

// ─── PDF Generation ───────────────────────────────────────────────────────────
async function buildPdf(profile, recommendations) {
  const doc = new jsPDF({ unit: 'pt', format: 'a4' });
  const PW = 595;   // Page width
  const PH = 842;   // Page height
  const ML = 40;    // Margin left
  const MR = 40;    // Margin right
  const CW = PW - ML - MR;  // Content width = 515pt
  let y = 0;

  // ─── Helpers ────────────────────────────────────────────
  const txt = (text, x, yy, size, style = 'normal', r = 0, g = 0, b = 0) => {
    doc.setFont('helvetica', style);
    doc.setFontSize(size);
    doc.setTextColor(r, g, b);
    doc.text(String(text), x, yy);
  };

  const hrule = (yy, gray = 210) => {
    doc.setDrawColor(gray, gray, gray);
    doc.setLineWidth(0.5);
    doc.line(ML, yy, ML + CW, yy);
  };

  const sectionLabel = (label, yy) => {
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(7.5);
    doc.setTextColor(100, 116, 139); // slate-500
    doc.text(label.toUpperCase(), ML, yy);
  };

  // ─── HEADER BAND ────────────────────────────────────────
  doc.setFillColor(15, 23, 42);   // Slate-950
  doc.rect(0, 0, PW, 56, 'F');

  // Text titles (perfectly aligned to ML margin)
  txt('GEOADHIKAR — DECISION SUPPORT SYSTEM', ML, 24, 11, 'bold', 255, 255, 255);
  txt('Tribal & Rural Development Scheme Prioritization Report  |  Ministry of Tribal Affairs, Government of India', ML, 38, 7.5, 'normal', 180, 196, 220);
  
  // Right-aligned header metadata (aligned to PW - MR margin)
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(7.5);
  doc.setTextColor(180, 196, 220);
  doc.text(`Generated: ${new Date().toLocaleDateString('en-IN', { day:'2-digit', month:'short', year:'numeric' })}`, PW - MR, 24, { align: 'right' });
  doc.setFontSize(7);
  doc.setTextColor(100, 130, 180);
  doc.text(`Ref: ${profile.block_id}`, PW - MR, 38, { align: 'right' });

  // ─── LOCATION TITLE ─────────────────────────────────────
  y = 76;
  txt(`${profile.block_name}`, ML, y, 16, 'bold', 15, 23, 42);
  y += 15;
  txt(`${profile.district}, ${profile.state}`, ML, y, 9.5, 'normal', 71, 85, 105);

  y += 6;
  hrule(y);

  // ─── SOCIO-ECONOMIC METRICS (2-col grid) ────────────────
  y += 14;
  sectionLabel('Area Socio-Economic Metrics', y);
  y += 13;

  const metrics = [
    { label: 'Total Population',         val: fmtNum(profile.population) },
    { label: 'Area (sq km)',              val: `${fmtNum(profile.area_sq_km)} sq km` },
    { label: 'Tribal Population',         val: `${profile.tribal_population_percent}%` },
    { label: 'Forest Cover',              val: `${profile.forest_cover_percent}%` },
    { label: 'Agricultural Land',         val: `${profile.agri_land_percent}%` },
    { label: 'Households with Tap Water', val: `${profile.households_with_water}%` },
    { label: 'Villages ODF+ Status',      val: `${profile.villages_odf_plus}%` },
    { label: 'FRA Claims Filed',          val: fmtNum(profile.fra_claims) },
  ];

  const colW = CW / 2;
  metrics.forEach((m, i) => {
    const col = i % 2;
    const x = ML + col * colW;
    if (col === 0 && i > 0) y += 16;
    txt(m.label + ':', x, y, 8.5, 'normal', 100, 116, 139);
    txt(m.val, x + 148, y, 8.5, 'bold', 15, 23, 42);
  });

  y += 14;
  hrule(y);

  // ─── MAP SECTION ────────────────────────────────────────
  y += 13;
  sectionLabel('Geospatial Boundary Assessment', y);
  y += 10;

  // Dynamically calculate map height so it fills the full page
  // Footer starts at PH-34. Schemes section below map takes fixed height:
  //   hrule(12) + label(13) + gap(12) + tableHeader(16) + headerRule(10) + 5rows*22(110) = 173pt
  // Add 20pt buffer.
  const footerTop = PH - 34;
  const SCHEMES_BELOW = 12 + 13 + 12 + 16 + 10 + 110; // 173
  const BUFFER = 8;
  const MAP_H = Math.max(200, footerTop - y - SCHEMES_BELOW - BUFFER);

  try {
    const mapEl = document.getElementById('map');
    // Grab the Mapbox canvas directly for best quality
    const mapCanvas = mapEl.querySelector('canvas');
    let imgData;
    if (mapCanvas) {
      imgData = mapCanvas.toDataURL('image/png');
    } else {
      const fallback = await html2canvas(mapEl, { useCORS: true, backgroundColor: '#0b0f19', scale: 1.5 });
      imgData = fallback.toDataURL('image/png');
    }

    // Border + image
    doc.setDrawColor(203, 213, 225);
    doc.setLineWidth(1);
    doc.rect(ML, y, CW, MAP_H, 'D');
    doc.addImage(imgData, 'PNG', ML + 1, y + 1, CW - 2, MAP_H - 2);
  } catch (e) {
    // Fallback placeholder if capture fails
    doc.setFillColor(241, 245, 249);
    doc.rect(ML, y, CW, MAP_H, 'F');
    doc.setDrawColor(203, 213, 225);
    doc.rect(ML, y, CW, MAP_H, 'D');
    txt('[ Map capture unavailable — ensure preserveDrawingBuffer is enabled ]', ML + CW / 2, y + MAP_H / 2, 8, 'normal', 148, 163, 184);
  }

  y += MAP_H + 12;
  hrule(y);

  // ─── SCHEME RECOMMENDATIONS ──────────────────────────────
  y += 13;
  sectionLabel('Scheme Recommendations & Prioritization', y);
  y += 12;

  // Table header
  doc.setFillColor(241, 245, 249); // slate-100
  doc.rect(ML, y - 10, CW, 16, 'F');
  txt('Rank', ML + 4, y, 7.5, 'bold', 71, 85, 105);
  txt('Scheme Name', ML + 38, y, 7.5, 'bold', 71, 85, 105);
  txt('Benefit Summary', ML + 185, y, 7.5, 'bold', 71, 85, 105);
  txt('Priority Score', ML + CW - 62, y, 7.5, 'bold', 71, 85, 105);

  y += 10;
  hrule(y);

  const top5 = recommendations.slice(0, 5);
  const rowH = 22;
  top5.forEach((r, i) => {
    y += rowH;
    if (i % 2 === 1) {
      doc.setFillColor(248, 250, 252);
      doc.rect(ML, y - rowH + 4, CW, rowH, 'F');
    }

    const rankColor = r.eligible ? [30, 64, 175] : [156, 163, 175]; // blue or gray
    txt(`#${i + 1}`, ML + 4, y, 8.5, 'bold', ...rankColor);
    txt(r.scheme.name.length > 26 ? r.scheme.name.substring(0, 24) + '…' : r.scheme.name, ML + 38, y, 8.5, 'bold', 15, 23, 42);
    const benefit = r.scheme.benefit.length > 55 ? r.scheme.benefit.substring(0, 53) + '…' : r.scheme.benefit;
    txt(benefit, ML + 185, y, 8, 'normal', 71, 85, 105);
    const scoreText = r.eligible ? `${r.score.toFixed(1)}%` : 'Ineligible';
    const scoreColor = r.eligible ? [22, 163, 74] : [239, 68, 68]; // green or red
    txt(scoreText, ML + CW - 55, y, 8.5, 'bold', ...scoreColor);
  });

  // ─── FOOTER ──────────────────────────────────────────────
  const footerY = PH - 28;
  doc.setFillColor(15, 23, 42);
  doc.rect(0, footerY - 6, PW, 34, 'F');

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(6.5);
  
  // Clean, non-overlapping two-line footer layout
  doc.setTextColor(148, 163, 184);
  doc.text(`GeoAdhikar System Report  |  Page 1 of 1`, ML, footerY + 4);
  doc.text(`Confidential — Academic Research Prototype`, PW - MR, footerY + 4, { align: 'right' });
  
  doc.setTextColor(100, 116, 139);
  doc.text('Data Sources: Census of India 2011 · Forest Survey of India (ISFR) · Jal Jeevan Mission Dashboard · Swachh Bharat Mission · Ministry of Tribal Affairs (FRA)', ML, footerY + 14);

  return doc;
}

// ─── App Init ─────────────────────────────────────────────────────────────────
export async function initApp() {
  // Show loading overlay
  const overlay = $('mapLoadingOverlay');

  state.map = initMap();

  try {
    const [blocks, schemes, boundaries] = await Promise.all([
      fetchJSON('/data/blocks.json'),
      fetchJSON('/data/schemes.json'),
      fetchJSON('/data/admin_boundaries_sample.geojson'),
    ]);

    state.blocks = blocks;
    state.schemes = schemes;
    state.boundaries = boundaries;

    console.log(`✅ Loaded ${blocks.length} blocks, ${schemes.length} schemes, ${boundaries.features.length} boundaries`);

    populateStateSelect(blocks);
    setupInteractions();
    initScanner(state.map, MAPBOX_TOKEN);
  } catch (err) {
    console.error('Failed to load data:', err);
    if (overlay) {
      overlay.innerHTML = `<div style="color:#dc2626;font-size:14px;text-align:center;padding:20px;">
        ⚠️ Failed to load data<br><small>${err.message}</small>
      </div>`;
    }
  }
}
