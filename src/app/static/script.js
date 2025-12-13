const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewSection = document.getElementById('preview-section');
const previewImg = document.getElementById('preview-img');
const resultsSection = document.getElementById('results-section');
const statusText = document.getElementById('status-text');

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

    // Show Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        dropZone.parentNode.style.display = 'none';
        previewSection.style.display = 'block';
        
        // Start Analysis
        uploadAndAnalyze(file);
    };
    reader.readAsDataURL(file);
}

async function uploadAndAnalyze(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            // Wait a bit for animation
            setTimeout(() => {
                showResults(data);
            }, 2000);
        } else {
            statusText.innerText = "Error: " + (data.error || "Analysis failed");
            statusText.style.color = "#ff453a";
        }
    } catch (error) {
        statusText.innerText = "Error: Connection failed";
        statusText.style.color = "#ff453a";
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
    fillList('beard-list', recs.beards);
    fillList('glasses-list', recs.glasses);
    
    document.getElementById('avoid-text').innerText = recs.avoid || "None";
}

function fillList(elementId, items) {
    const ul = document.getElementById(elementId);
    ul.innerHTML = "";
    if (items) {
        items.forEach(item => {
            const li = document.createElement('li');
            li.innerText = item;
            ul.appendChild(li);
        });
    }
}

function resetApp() {
    resultsSection.style.display = 'none';
    dropZone.parentNode.style.display = 'flex';
    fileInput.value = '';
    statusText.innerText = "Scanning facial features...";
    statusText.style.color = "#bb86fc";
}
