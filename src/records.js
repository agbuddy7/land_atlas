import { getSegmentedPlots, highlightMatchedPlots, zoomToPlot, setScannerCoordinates } from './scanner.js';

// ─── State ───────────────────────────────────────────────────────────────────
let mapInstance = null;
let mapboxToken = '';
let currentRecord = null;
let currentMatchedPlot = null;
let recordLocationMarker = null;

const STORAGE_KEY = 'geoadhikar_land_records';

// ─── Record Book Storage Helpers ─────────────────────────────────────────────
export function getStoredRecords() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (e) {
    console.error('Failed to parse stored records:', e);
    return [];
  }
}

export function saveStoredRecords(records) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(records));
  } catch (e) {
    console.error('Failed to save records to localStorage:', e);
  }
}

// ─── Sample Preset Document (7/12 Maharashtra) ────────────────────────────────
const SAMPLE_RECORD = {
  state: "Maharashtra",
  district: "Pune",
  taluka_or_tehsil: "Haveli",
  village: "Kothrud",
  survey_or_gat_no: "142/3A",
  khata_no: "562",
  total_area: "0.85 Hectare",
  owners: ["Santosh Tukaram Patil", "Sunita Santosh Patil"],
  liabilities_or_loans: ["State Bank of India — Agri KCC Loan ₹1,50,000"],
  summary: "7/12 Land Record for Gat No. 142/3A located in Village Kothrud, Haveli, Pune. Total registered parcel area is 0.85 Hectare under joint ownership of Santosh & Sunita Patil. Active encumbrance notes an SBI agricultural crop loan.",
  full_raw_text: "गाव नमुना सात (अधिकार अभिलेख पत्रक) - गाव: कोथरूड, तालुका: हवेली, जिल्हा: पुणे. भूमापन क्रमांक व उपविभाग: १४२/३अ. खाते क्रमांक: ५६२. एकूण क्षेत्र: ०.८५ हेक्टर. धारकाचे नाव: संतोष तुकाराम पाटील, सुनीता संतोष पाटील. इतर हक्क व कर्जे: भारतीय स्टेट बँक पीक कर्ज रु. १,५०,०००.",
  parsed_area_sqm: 8500.0,
  parsed_area_ha: 0.85,
  parsed_area_acres: 2.10,
  ai_provider: "Verified Land Record (Sample Preset)"
};

// ─── Geocode Record & Fly Map ─────────────────────────────────────────────────
export async function geocodeAndFly(record) {
  if (!mapInstance) return;

  const btn = document.getElementById('btnNavLocation');
  if (btn) {
    btn.disabled = true;
    btn.textContent = '🛰️ Geocoding Location...';
  }

  try {
    const res = await fetch('http://127.0.0.1:8000/geocode-record', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        state: record.state,
        district: record.district,
        taluka_or_tehsil: record.taluka_or_tehsil,
        village: record.village,
        survey_or_gat_no: record.survey_or_gat_no
      })
    });

    const data = await res.json();
    if (data.coordinates && data.coordinates.length === 2) {
      const [lng, lat] = data.coordinates;

      // Also set the Scanner coordinates so user can directly snapshot this village!
      setScannerCoordinates(lat, lng, `${record.village}, ${record.district}`);

      // Add or move marker on map
      if (recordLocationMarker) recordLocationMarker.remove();

      const el = document.createElement('div');
      el.className = 'cadastral-pulse-marker';
      el.innerHTML = `
        <div class="pulse-pin">📍</div>
        <div class="pulse-label">${record.village} (${record.survey_or_gat_no})</div>
      `;

      recordLocationMarker = new mapboxgl.Marker(el)
        .setLngLat([lng, lat])
        .setPopup(new mapboxgl.Popup({ offset: 25 }).setHTML(`
          <div style="font-family:Inter,sans-serif; padding:4px;">
            <strong style="color:#0f172a; font-size:13px;">${record.village}</strong>
            <div style="font-size:11px; color:#475569;">Gat/Survey: ${record.survey_or_gat_no}</div>
            <div style="font-size:11px; color:#475569;">Area: ${record.total_area}</div>
            <div style="font-size:10px; color:#059669; font-weight:600; margin-top:2px;">📍 ${data.place_name}</div>
          </div>
        `))
        .addTo(mapInstance);

      recordLocationMarker.togglePopup();

      mapInstance.flyTo({
        center: [lng, lat],
        zoom: 16.5,
        pitch: 35,
        duration: 2000
      });

      showToast(`📍 Centered on ${record.village}, ${record.district}`);
    } else {
      alert('Could not pinpoint coordinates for this location.');
    }
  } catch (err) {
    console.error('Geocoding error:', err);
    alert('Geocoding service unavailable: ' + err.message);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = '🗺️ Navigate Map to Location';
    }
  }
}

// ─── Area Matching Against Segmented Plots ────────────────────────────────────
export function matchStatedAreaWithPlots(record) {
  const matchResultCard = document.getElementById('plotMatchResults');
  const matchSummary = document.getElementById('plotMatchSummary');
  const candidatesList = document.getElementById('plotMatchCandidates');

  if (!matchResultCard) return;

  const statedAreaSqm = record.parsed_area_sqm;
  if (!statedAreaSqm || isNaN(statedAreaSqm)) {
    alert('The document does not contain a recognizable numeric land area (e.g. 1.2 Hectare, 500 sqm).');
    return;
  }

  const plotsCollection = getSegmentedPlots();
  if (!plotsCollection || !plotsCollection.features || plotsCollection.features.length === 0) {
    alert('No segmented satellite plots found yet!\n\n1. Go to the "Plot Snapshot Scanner" tab\n2. Click "Generate Grid & Capture High-Res Snaps"\n3. Click "Run SAM Model on Snapshots"\nThen come back here to match!');
    return;
  }

  const features = plotsCollection.features;

  // Rank plots by proximity to stated area in square meters
  const ranked = features.map(f => {
    const plotSqm = f.properties.area_sq_m || 0;
    const diffSqm = Math.abs(plotSqm - statedAreaSqm);
    const diffPct = Math.round((diffSqm / statedAreaSqm) * 100);
    const similarity = Math.max(0, Math.round(100 - diffPct));

    return {
      plot: f,
      plot_id: f.properties.plot_id,
      cell_id: f.properties.cell_id,
      land_type: f.properties.land_type,
      color: f.properties.color,
      plotSqm,
      plotHa: f.properties.area_hectares,
      plotAcres: f.properties.area_acres,
      diffSqm: Math.round(diffSqm),
      diffPct,
      similarity
    };
  }).sort((a, b) => a.diffPct - b.diffPct);

  const bestMatch = ranked[0];
  currentMatchedPlot = bestMatch;

  matchResultCard.style.display = 'block';

  // Highlight top matching plots on map
  const topPlotIds = ranked.slice(0, 3).map(r => r.plot_id);
  highlightMatchedPlots(topPlotIds);

  const isStrongMatch = bestMatch.diffPct <= 20;
  const matchBadgeColor = isStrongMatch ? '#10b981' : (bestMatch.diffPct <= 40 ? '#f59e0b' : '#ef4444');
  const matchStatusText = isStrongMatch ? 'High Confidence Match' : (bestMatch.diffPct <= 40 ? 'Moderate Match' : 'Area Variance Detected');

  if (matchSummary) {
    matchSummary.innerHTML = `
      <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
        <span style="font-size:11px; font-weight:700; color:${matchBadgeColor}; text-transform:uppercase; letter-spacing:0.05em;">
          ● ${matchStatusText} (${bestMatch.similarity}% match)
        </span>
        <span class="counter-badge" style="background:rgba(251, 191, 36, 0.15); color:#fbbf24; border-color:rgba(251, 191, 36, 0.3);">
          Stated: ${record.total_area}
        </span>
      </div>
      <div style="font-size:12px; line-height:1.6; color:var(--text);">
        Closest satellite parcel is <strong>${bestMatch.plot_id}</strong> (${bestMatch.plotHa} ha / ${bestMatch.plotSqm.toLocaleString()} m²), differing by only <strong>${bestMatch.diffPct}%</strong> (${bestMatch.diffSqm.toLocaleString()} m² difference).
      </div>
      <div style="margin-top:8px; display:flex; gap:6px;">
        <button id="btnZoomToBestPlot" class="btn btn-sm btn-primary" type="button">
          🎯 Zoom to Plot ${bestMatch.plot_id}
        </button>
        <button id="btnBindPlotToRecord" class="btn btn-sm btn-secondary" type="button">
          🔗 Bind to Record
        </button>
      </div>
    `;

    document.getElementById('btnZoomToBestPlot')?.addEventListener('click', () => {
      zoomToPlot(bestMatch.plot_id);
    });

    document.getElementById('btnBindPlotToRecord')?.addEventListener('click', () => {
      if (currentRecord) {
        currentRecord.matched_plot_id = bestMatch.plot_id;
        currentRecord.matched_plot_diff = `${bestMatch.diffPct}%`;
        saveRecordToBook(currentRecord);
      }
    });
  }

  // Render Candidate List
  if (candidatesList) {
    candidatesList.innerHTML = ranked.slice(0, 5).map((c, i) => `
      <div class="card match-candidate-card ${i === 0 ? 'best-candidate' : ''}" data-plot-id="${c.plot_id}" style="margin-top:6px; padding:8px 10px; cursor:pointer;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <strong style="font-size:12px; color:var(--text);">${c.plot_id}</strong>
          <span style="font-size:10px; font-weight:700; color:${c.color};">${c.land_type}</span>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:10px; color:var(--muted); margin-top:4px;">
          <span>Satellite: ${c.plotHa} ha (${c.plotSqm.toLocaleString()} m²)</span>
          <span style="color:${c.diffPct <= 20 ? '#10b981' : '#f59e0b'}; font-weight:700;">Diff: ${c.diffPct}%</span>
        </div>
      </div>
    `).join('');

    candidatesList.querySelectorAll('.match-candidate-card').forEach(card => {
      card.addEventListener('click', () => {
        const plotId = card.dataset.plotId;
        zoomToPlot(plotId);
      });
    });
  }

  matchResultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ─── Display Extracted Record ────────────────────────────────────────────────
export function displayExtractedRecord(record) {
  currentRecord = record;

  const resultPanel = document.getElementById('ocrResultPanel');
  if (!resultPanel) return;

  resultPanel.style.display = 'block';

  // Basic Details
  document.getElementById('recVillage').textContent = record.village || 'N/A';
  document.getElementById('recTaluka').textContent = record.taluka_or_tehsil || 'N/A';
  document.getElementById('recDistrict').textContent = record.district || 'N/A';
  document.getElementById('recState').textContent = record.state || 'N/A';

  document.getElementById('recSurveyNo').textContent = record.survey_or_gat_no || 'N/A';
  document.getElementById('recKhataNo').textContent = record.khata_no || 'N/A';
  document.getElementById('recTotalArea').textContent = record.total_area || 'N/A';

  // Area conversions
  const areaBreakdown = document.getElementById('recAreaBreakdown');
  if (areaBreakdown && record.parsed_area_sqm) {
    areaBreakdown.textContent = `≈ ${record.parsed_area_sqm.toLocaleString()} m² · ${record.parsed_area_acres} Acres`;
  } else if (areaBreakdown) {
    areaBreakdown.textContent = '';
  }

  // Owners
  const ownersList = document.getElementById('recOwnersList');
  if (ownersList) {
    const owners = record.owners || [];
    if (owners.length === 0) {
      ownersList.innerHTML = '<span class="chip-owner">Not specified</span>';
    } else {
      ownersList.innerHTML = owners.map(o => `<span class="chip-owner">👤 ${o}</span>`).join('');
    }
  }

  // Liabilities
  const liabilitiesList = document.getElementById('recLiabilitiesList');
  if (liabilitiesList) {
    const liabs = record.liabilities_or_loans || [];
    if (liabs.length === 0) {
      liabilitiesList.innerHTML = '<span style="color:#10b981; font-size:11px;">✅ No active loans or encumbrances recorded</span>';
    } else {
      liabilitiesList.innerHTML = liabs.map(l => `<div class="liability-item">⚠️ ${l}</div>`).join('');
    }
  }

  // Summary
  document.getElementById('recSummary').textContent = record.summary || 'Summary unavailable.';

  // Raw transcription
  const rawTextEl = document.getElementById('recRawText');
  if (rawTextEl) {
    rawTextEl.textContent = record.full_raw_text || '[Raw transcription omitted in fast vision mode]';
  }

  // AI Provider badge
  const aiBadge = document.getElementById('recAiProviderBadge');
  if (aiBadge) {
    aiBadge.textContent = record.ai_provider || 'AI Extracted';
  }

  // Scroll into view
  resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ─── Record Book Storage & Table Rendering ────────────────────────────────────
export function saveRecordToBook(record) {
  const records = getStoredRecords();

  const recordEntry = {
    id: `REC_${Date.now().toString().slice(-6)}`,
    created_at: new Date().toISOString(),
    village: record.village || 'N/A',
    taluka_or_tehsil: record.taluka_or_tehsil || 'N/A',
    district: record.district || 'N/A',
    state: record.state || 'N/A',
    survey_or_gat_no: record.survey_or_gat_no || 'N/A',
    khata_no: record.khata_no || 'N/A',
    total_area: record.total_area || 'N/A',
    parsed_area_sqm: record.parsed_area_sqm || null,
    owners: record.owners || [],
    liabilities_or_loans: record.liabilities_or_loans || [],
    summary: record.summary || '',
    matched_plot_id: record.matched_plot_id || (currentMatchedPlot ? currentMatchedPlot.plot_id : null),
    ai_provider: record.ai_provider || 'AI Vision'
  };

  // Add to top of list
  records.unshift(recordEntry);
  saveStoredRecords(records);
  renderRecordBook();

  showToast(`✅ Saved to Record Book (${recordEntry.id})`);
}

export function deleteRecordFromBook(recordId) {
  const records = getStoredRecords().filter(r => r.id !== recordId);
  saveStoredRecords(records);
  renderRecordBook();
  showToast('🗑️ Record deleted');
}

export function clearRecordBook() {
  if (confirm('Are you sure you want to clear all saved land records?')) {
    localStorage.removeItem(STORAGE_KEY);
    renderRecordBook();
    showToast('Record book cleared');
  }
}

export function renderRecordBook(filterQuery = '') {
  const container = document.getElementById('recordBookItems');
  const countBadge = document.getElementById('recordBookCount');
  if (!container) return;

  let records = getStoredRecords();

  if (filterQuery.trim()) {
    const q = filterQuery.toLowerCase().trim();
    records = records.filter(r =>
      r.village.toLowerCase().includes(q) ||
      r.district.toLowerCase().includes(q) ||
      r.survey_or_gat_no.toLowerCase().includes(q) ||
      (r.owners && r.owners.some(o => o.toLowerCase().includes(q))) ||
      (r.matched_plot_id && r.matched_plot_id.toLowerCase().includes(q))
    );
  }

  if (countBadge) countBadge.textContent = `${records.length} Records`;

  if (records.length === 0) {
    container.innerHTML = '<div class="empty-state">No land records in register yet. Upload a document or load sample record.</div>';
    return;
  }

  container.innerHTML = records.map(r => {
    const ownerNames = (r.owners && r.owners.length > 0) ? r.owners.join(', ') : 'N/A';
    const dateStr = new Date(r.created_at).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' });
    const matchedBadge = r.matched_plot_id
      ? `<span class="badge-matched" data-plot="${r.matched_plot_id}">🔗 Plot ${r.matched_plot_id}</span>`
      : `<span class="badge-unmatched">Unlinked</span>`;

    return `
      <div class="card record-book-card" data-rec-id="${r.id}">
        <div class="rec-header">
          <div>
            <strong style="font-size:13px; color:var(--text);">${r.village}, ${r.district}</strong>
            <div style="font-size:10px; color:var(--muted); margin-top:1px;">
              Survey: <strong>${r.survey_or_gat_no}</strong> · Khata: ${r.khata_no} · ${dateStr}
            </div>
          </div>
          <div>${matchedBadge}</div>
        </div>

        <div class="rec-owners-row" title="${ownerNames}">
          👤 ${ownerNames}
        </div>

        <div class="rec-footer-row">
          <span class="rec-area-badge">📐 ${r.total_area}</span>
          <div class="rec-actions">
            <button class="btn btn-xs btn-fly" data-id="${r.id}" title="Navigate to Location">🗺️ Fly</button>
            <button class="btn btn-xs btn-load" data-id="${r.id}" title="View Full Details">👁️ View</button>
            <button class="btn btn-xs btn-del" data-id="${r.id}" title="Delete Record">✕</button>
          </div>
        </div>
      </div>
    `;
  }).join('');

  // Wire action buttons
  container.querySelectorAll('.btn-fly').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const id = btn.dataset.id;
      const rec = records.find(r => r.id === id);
      if (rec) geocodeAndFly(rec);
    });
  });

  container.querySelectorAll('.btn-load').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const id = btn.dataset.id;
      const rec = records.find(r => r.id === id);
      if (rec) displayExtractedRecord(rec);
    });
  });

  container.querySelectorAll('.btn-del').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const id = btn.dataset.id;
      deleteRecordFromBook(id);
    });
  });

  container.querySelectorAll('.badge-matched').forEach(badge => {
    badge.addEventListener('click', (e) => {
      e.stopPropagation();
      const plotId = badge.dataset.plot;
      if (plotId) zoomToPlot(plotId);
    });
  });
}

// ─── Export Record Book ───────────────────────────────────────────────────────
export function exportRecordBookJson() {
  const records = getStoredRecords();
  if (records.length === 0) {
    alert('No records to export.');
    return;
  }
  const str = JSON.stringify(records, null, 2);
  const blob = new Blob([str], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `GeoAdhikar_Land_Records_${new Date().toISOString().slice(0, 10)}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

export function exportRecordBookCsv() {
  const records = getStoredRecords();
  if (records.length === 0) {
    alert('No records to export.');
    return;
  }

  const rows = [
    ['Record_ID', 'Date', 'State', 'District', 'Taluka', 'Village', 'Survey_Gat_No', 'Khata_No', 'Total_Area', 'Area_SqM', 'Owners', 'Liabilities', 'Matched_Plot_ID']
  ];

  records.forEach(r => {
    rows.push([
      r.id,
      r.created_at ? r.created_at.slice(0, 10) : '',
      `"${r.state || ''}"`,
      `"${r.district || ''}"`,
      `"${r.taluka_or_tehsil || ''}"`,
      `"${r.village || ''}"`,
      `"${r.survey_or_gat_no || ''}"`,
      `"${r.khata_no || ''}"`,
      `"${r.total_area || ''}"`,
      r.parsed_area_sqm || '',
      `"${(r.owners || []).join('; ')}"`,
      `"${(r.liabilities_or_loans || []).join('; ')}"`,
      r.matched_plot_id || ''
    ]);
  });

  const csvContent = rows.map(r => r.join(',')).join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `GeoAdhikar_Land_Records_${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// ─── Process Uploaded File With OCR ───────────────────────────────────────────
export async function processDocumentUpload(file) {
  const statusBox = document.getElementById('ocrStatus');
  const statusText = document.getElementById('ocrStatusText');
  const uploadBtn = document.getElementById('btnStartOcr');

  if (statusBox) statusBox.style.display = 'block';
  if (uploadBtn) {
    uploadBtn.disabled = true;
    uploadBtn.textContent = '⏳ Processing Document with AI...';
  }
  if (statusText) statusText.textContent = `Analyzing ${file.name} with Gemini / Groq Vision...`;

  const formData = new FormData();
  formData.append('image', file);

  try {
    const res = await fetch('http://127.0.0.1:8000/ocr', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      throw new Error(`OCR failed with HTTP ${res.status}`);
    }

    const record = await res.json();
    displayExtractedRecord(record);
    showToast('✨ Document processed successfully!');

    // Automatically check if user wants to geocode
    if (record.village && record.village !== 'N/A') {
      geocodeAndFly(record);
    }

  } catch (err) {
    console.error('OCR Error:', err);
    alert('OCR extraction error: ' + err.message + '\n\nEnsure backend server is running at http://127.0.0.1:8000.');
  } finally {
    if (statusBox) statusBox.style.display = 'none';
    if (uploadBtn) {
      uploadBtn.disabled = false;
      uploadBtn.textContent = '🧠 Run AI OCR Extraction';
    }
  }
}

// ─── Toast Notification Helper ────────────────────────────────────────────────
function showToast(msg) {
  let toast = document.getElementById('appToast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'appToast';
    toast.className = 'app-toast';
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.classList.add('visible');
  setTimeout(() => toast.classList.remove('visible'), 3200);
}

// ─── Module Initialization ────────────────────────────────────────────────────
export function initRecords(map, token) {
  mapInstance = map;
  mapboxToken = token;

  let selectedFile = null;

  const dropzone = document.getElementById('ocrDropzone');
  const fileInput = document.getElementById('ocrFileInput');
  const previewImg = document.getElementById('ocrFilePreview');
  const filenameEl = document.getElementById('ocrFileName');
  const startBtn = document.getElementById('btnStartOcr');
  const sampleBtn = document.getElementById('btnLoadSampleRecord');

  // File selection
  if (dropzone && fileInput) {
    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('drag-over');
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('drag-over');
    });

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag-over');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
      }
    });
  }

  function handleFileSelect(file) {
    selectedFile = file;
    if (filenameEl) filenameEl.textContent = `📄 ${file.name} (${Math.round(file.size / 1024)} KB)`;
    if (startBtn) startBtn.disabled = false;

    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        if (previewImg) {
          previewImg.src = e.target.result;
          previewImg.style.display = 'block';
        }
      };
      reader.readAsDataURL(file);
    }
  }

  // Start OCR Button
  if (startBtn) {
    startBtn.addEventListener('click', () => {
      if (selectedFile) {
        processDocumentUpload(selectedFile);
      }
    });
  }

  // Sample Record Button
  if (sampleBtn) {
    sampleBtn.addEventListener('click', () => {
      displayExtractedRecord(SAMPLE_RECORD);
      geocodeAndFly(SAMPLE_RECORD);
      showToast('📄 Loaded sample 7/12 land document');
    });
  }

  // Navigate Button
  document.getElementById('btnNavLocation')?.addEventListener('click', () => {
    if (currentRecord) geocodeAndFly(currentRecord);
  });

  // Match Area Button
  document.getElementById('btnMatchArea')?.addEventListener('click', () => {
    if (currentRecord) matchStatedAreaWithPlots(currentRecord);
  });

  // Save to Book Button
  document.getElementById('btnSaveToBook')?.addEventListener('click', () => {
    if (currentRecord) saveRecordToBook(currentRecord);
  });

  // Export JSON Button
  document.getElementById('btnExportRecordJson')?.addEventListener('click', () => {
    exportRecordBookJson();
  });

  // Export CSV Button
  document.getElementById('btnExportRecordCsv')?.addEventListener('click', () => {
    exportRecordBookCsv();
  });

  // Clear Records Button
  document.getElementById('btnClearRecords')?.addEventListener('click', () => {
    clearRecordBook();
  });

  // Search in Record Book
  document.getElementById('recordBookSearchInput')?.addEventListener('input', (e) => {
    renderRecordBook(e.target.value);
  });

  // Initial render of saved records
  renderRecordBook();

  // If there are existing records, pre-populate if requested
  const stored = getStoredRecords();
  if (stored.length === 0) {
    // Save sample record by default for immediate exploration
    saveStoredRecords([SAMPLE_RECORD]);
    renderRecordBook();
  }
}
