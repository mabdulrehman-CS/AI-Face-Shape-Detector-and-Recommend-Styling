const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewSection = document.getElementById('preview-section');
const previewImg = document.getElementById('preview-img');
const resultsSection = document.getElementById('results-section');
const statusText = document.getElementById('status-text');
const modelSelect = document.getElementById('model-select');

// Load Models on Startup
async function loadModels() {
    try {
        const response = await fetch('/models');
        const data = await response.json();

        modelSelect.innerHTML = "";
        data.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.innerText = model.name;
            if (model.name === data.active) {
                option.selected = true;
            }
            modelSelect.appendChild(option);
        });

        if (data.active) {
            fetchMetrics(data.active);
        }
    } catch (e) {
        console.error("Failed to load models", e);
    }
}
loadModels();

// Handle Model Switch
modelSelect.addEventListener('change', async () => {
    const selectedModel = modelSelect.value;
    statusText.innerText = "Switching AI Model...";
    statusText.style.color = "#bb86fc";

    try {
        const response = await fetch('/switch_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: selectedModel })
        });
        const res = await response.json();
        if (response.ok) {
            alert(`Switched to: ${selectedModel}`);
            statusText.innerText = "Model Ready. Upload an image.";
            fetchMetrics(selectedModel);
        } else {
            alert("Error switching model: " + res.error);
        }
    } catch (e) {
        alert("Connection failed");
    }
});

// Drag & Drop
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#bb86fc';
});

dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = '#b3b3b3';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#b3b3b3';
    const files = e.dataTransfer.files;
    if (files.length) handleFile(files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }

    // Show Preview Immediately
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        // Hide Input Selection (The whole section)
        document.querySelector('.upload-section').style.display = 'none';
        previewSection.style.display = 'block';

        // Start Analysis
        uploadAndAnalyze(file);
    };
    reader.readAsDataURL(file);
}

async function uploadAndAnalyze(file) {
    if (!file) return;

    statusText.innerText = "Analyzing...";
    statusText.style.color = "#bb86fc";
    resultsSection.classList.add('hidden');

    // Create FormData
    const formData = new FormData();
    formData.append('file', file);

    // Get Selected Gender
    const gender = document.querySelector('input[name="gender"]:checked').value;
    formData.append('gender', gender);

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            statusText.innerText = "Analysis Complete!";
            statusText.style.color = "#03dac6";
            // Wait a bit for animation
            setTimeout(() => {
                showResults(data);
            }, 1000);
        } else {
            statusText.innerText = "Error: " + (data.error || "Analysis failed");
            statusText.style.color = "#ff453a";
            // Show reset button so they can try again
            // We can inject a 'Try Again' button or just rely on ResetApp
            // Let's create a temporary try again link/button if error
            if (!document.getElementById('temp-reset-btn')) {
                const btn = document.createElement('button');
                btn.id = 'temp-reset-btn';
                btn.className = 'btn-reset';
                btn.innerText = "Try Different Photo";
                btn.style.marginTop = "20px";
                btn.onclick = resetApp;
                previewSection.appendChild(btn);
            }
        }
    } catch (error) {
        statusText.innerText = "Error: Connection failed";
        statusText.style.color = "#ff453a";
        if (!document.getElementById('temp-reset-btn')) {
            const btn = document.createElement('button');
            btn.id = 'temp-reset-btn';
            btn.className = 'btn-reset';
            btn.innerText = "Try Again";
            btn.style.marginTop = "20px";
            btn.onclick = resetApp;
            previewSection.appendChild(btn);
        }
    }
}

function showResults(data) {
    previewSection.style.display = 'none';
    resultsSection.style.display = 'block';

    document.getElementById('shape-name').innerText = data.predicted_shape;

    // Confidence as %
    const conf = Math.round(data.confidence_score * 100);
    document.getElementById('conf-score').innerText = conf + "%";

    // Populate Recommendations
    const recs = data.recommendations;

    document.getElementById('shape-desc').innerText = recs.description || "";

    fillList('hair-list', recs.hairstyles);
    fillList('glasses-list', recs.glasses);

    // Dynamic Middle Card (Beard vs Makeup)
    const middleList = document.getElementById('beard-list');
    const middleTitle = middleList.previousElementSibling; // The <h3> tag

    if (recs.makeup && recs.makeup.length > 0) {
        middleTitle.innerText = "💄 Makeup";
        fillList('beard-list', recs.makeup);
    } else {
        middleTitle.innerText = "🧔 Beards";
        fillList('beard-list', recs.beards);
    }

    document.getElementById('avoid-text').innerText = recs.avoid || "None";
}

function fillList(elementId, items) {
    const ul = document.getElementById(elementId);
    ul.innerHTML = "";
    if (items && Array.isArray(items)) { // Check array validity
        items.forEach(item => {
            const li = document.createElement('li');
            li.innerText = item;
            ul.appendChild(li);
        });
    }
}

function resetApp() {
    resultsSection.style.display = 'none';
    document.querySelector('.upload-section').style.display = 'block'; // Show inputs again
    previewSection.style.display = 'none'; // Hide preview
    fileInput.value = '';
    statusText.innerText = "Scanning facial features...";
    statusText.style.color = "#bb86fc";

    // Remove temp button if exists
    const tempBtn = document.getElementById('temp-reset-btn');
    if (tempBtn) tempBtn.remove();
}

// Metrics Display
async function fetchMetrics(modelId) {
    const metricsSection = document.getElementById('metrics-section');
    try {
        const response = await fetch(`/model_metrics?model_id=${encodeURIComponent(modelId)}`);

        if (!response.ok) {
            metricsSection.style.display = 'none';
            return;
        }

        const data = await response.json();
        metricsSection.style.display = 'block';

        document.getElementById('metrics-model-name').innerText = data.model_name;
        document.getElementById('metric-acc').innerText = (data.accuracy * 100).toFixed(1) + "%";
        document.getElementById('metric-f1').innerText = data.f1_score.toFixed(2);

        // Table
        const tbody = document.querySelector('#metrics-table tbody');
        tbody.innerHTML = "";
        for (const [cls, stats] of Object.entries(data.classification_report)) {
            const row = `<tr>
                <td>${cls}</td>
                <td>${stats.precision.toFixed(2)}</td>
                <td>${stats.recall.toFixed(2)}</td>
                <td>${stats["f1-score"].toFixed(2)}</td>
            </tr>`;
            tbody.innerHTML += row;
        }

        // Matrix
        renderMatrix(data.confusion_matrix, data.classes);

    } catch (e) {
        console.error("Metric fetch failed", e);
        metricsSection.style.display = 'none';
    }
}

function renderMatrix(matrix, classes) {
    const container = document.getElementById('confusion-matrix');
    container.innerHTML = "";

    // Create Grid
    container.style.display = 'grid';
    container.style.gridTemplateColumns = `auto repeat(${classes.length}, 1fr)`;
    container.style.gap = '2px';

    // Headers (Top)
    container.appendChild(document.createElement('div')); // Corner
    classes.forEach(c => {
        const div = document.createElement('div');
        div.innerText = c.substring(0, 3); // Short name
        div.className = 'matrix-header';
        container.appendChild(div);
    });

    // Rows
    matrix.forEach((row, i) => {
        // Feature Header (Left)
        const rowHead = document.createElement('div');
        rowHead.innerText = classes[i].substring(0, 3);
        rowHead.className = 'matrix-header';
        container.appendChild(rowHead);

        const rowMax = Math.max(...row);

        row.forEach(val => {
            const cell = document.createElement('div');
            cell.className = 'matrix-cell';
            cell.innerText = val;

            // Heatmap logic
            const intensity = val / rowMax;
            cell.style.backgroundColor = `rgba(187, 134, 252, ${intensity * 0.8})`;
            cell.style.color = intensity > 0.5 ? '#000' : '#fff';

            container.appendChild(cell);
        });
    });
}

// ================= Webcam Logic =================
const cameraBtn = document.getElementById('start-camera-btn');
const cameraContainer = document.getElementById('camera-container');
const videoFeed = document.getElementById('camera-feed');
const canvas = document.getElementById('camera-canvas');
const captureBtn = document.getElementById('capture-btn');
const closeCameraBtn = document.getElementById('close-camera-btn');
const rtResult = document.getElementById('rt-result');
const rtShape = document.getElementById('rt-shape');
const rtConf = document.getElementById('rt-conf');

let stream = null;
let pollInterval = null;

// Start Camera
if (cameraBtn) {
    cameraBtn.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "user" }
            });
            videoFeed.srcObject = stream;
            cameraContainer.classList.remove('hidden');

            // Start Polling 
            startPolling();

        } catch (err) {
            alert("Error accessing camera: " + err);
        }
    });
}

// Polling Loop
function startPolling() {
    if (pollInterval) clearInterval(pollInterval);
    rtResult.classList.remove('hidden');
    rtShape.innerText = "Initializing...";
    rtConf.innerText = "";

    pollInterval = setInterval(async () => {
        if (!stream) return;

        // Capture Frame
        canvas.width = videoFeed.videoWidth;
        canvas.height = videoFeed.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(blob => {
            if (!blob) return;
            const file = new File([blob], "rt_frame.jpg", { type: "image/jpeg" });
            analyzeFrame(file);
        }, 'image/jpeg', 0.8); // 0.8 quality for speed

    }, 800); // Every 800ms (~1.25 FPS)
}

async function analyzeFrame(file) {
    const formData = new FormData();
    formData.append('file', file);
    const gender = document.querySelector('input[name="gender"]:checked').value;
    formData.append('gender', gender);

    try {
        const response = await fetch('/analyze', { method: 'POST', body: formData });
        const data = await response.json();

        if (response.ok) {
            rtShape.innerText = data.predicted_shape;
            rtConf.innerText = `${Math.round(data.confidence_score * 100)}%`;
            rtShape.style.color = "#03dac6"; // Success color
        } else {
            // Silent fail or "No Face"
            if (data.error === "No face detected" || response.status === 400) {
                rtShape.innerText = "No Face Found";
                rtShape.style.color = "#ff453a"; // Error color
                rtConf.innerText = "";
            } else {
                rtShape.innerText = "Scanning...";
                rtShape.style.color = "#fff";
            }
        }
    } catch (e) {
        // console.log("Frame drop");
    }
}

// Close Camera
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    if (pollInterval) clearInterval(pollInterval);
    cameraContainer.classList.add('hidden');
    rtResult.classList.add('hidden');
}

if (closeCameraBtn) {
    closeCameraBtn.addEventListener('click', stopCamera);
}

// Capture Photo (Final Report)
if (captureBtn) {
    captureBtn.addEventListener('click', () => {
        if (!stream) return;

        // One last high-quality capture
        canvas.width = videoFeed.videoWidth;
        canvas.height = videoFeed.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(blob => {
            const file = new File([blob], "webcam-capture.jpg", { type: "image/jpeg" });
            stopCamera();

            // Show Preview & Full Analysis
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                document.querySelector('.upload-section').style.display = 'none';
                previewSection.style.display = 'block';
                uploadAndAnalyze(file);
            };
            reader.readAsDataURL(file);
        }, 'image/jpeg', 1.0);
    });
}
