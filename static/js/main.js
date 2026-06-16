document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    const loader = document.getElementById('loader');
    const resultSection = document.getElementById('resultSection');
    const imagePreview = document.getElementById('imagePreview');
    const greedyCard = document.getElementById('greedyCard');
    const beamCard = document.getElementById('beamCard');
    const greedyText = document.getElementById('greedyText');
    const beamText = document.getElementById('beamText');
    const beamHeader = document.getElementById('beamHeader');
    const beamWidthSelect = document.getElementById('beamWidth');
    const maxLengthInput = document.getElementById('maxLength');

    // Navigation Tabs Router
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    function switchTab(tabId) {
        tabButtons.forEach(btn => {
            btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
        });
        tabContents.forEach(content => {
            content.classList.toggle('active-content', content.id === `${tabId}Tab`);
        });

        // Lazy load dashboard data on tab activation
        if (tabId === 'stats') {
            loadStats();
        } else if (tabId === 'vocab') {
            loadVocab();
        }
    }

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            switchTab(btn.getAttribute('data-tab'));
        });
    });

    // 1. Tab 1: Caption Generator Drag and Drop event handlers
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        }, false);
    });

    uploadArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        // Preview image
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = () => {
            imagePreview.src = reader.result;
            resultSection.style.display = 'grid';
            greedyCard.classList.remove('show');
            beamCard.classList.remove('show');
        };

        // Call caption API
        generateCaption(file);
    }

    async function generateCaption(file) {
        loader.style.display = 'block';
        
        const formData = new FormData();
        formData.append('image', file);
        formData.append('beam_width', beamWidthSelect.value);
        formData.append('max_length', maxLengthInput.value);

        try {
            const response = await fetch('/api/caption', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                greedyText.textContent = `"${data.greedy}"`;
                beamText.textContent = `"${data.beam}"`;
                beamHeader.textContent = `Beam Search (Width ${beamWidthSelect.value})`;
                
                // Show cards with fade-in effect
                setTimeout(() => greedyCard.classList.add('show'), 100);
                setTimeout(() => beamCard.classList.add('show'), 300);
            } else {
                alert(data.error || 'Failed to generate captions.');
            }
        } catch (error) {
            console.error('Error generating captions:', error);
            alert('An error occurred during caption generation. Check backend server logs.');
        } finally {
            loader.style.display = 'none';
        }
    }

    // Demo images integration
    async function loadDemoImages() {
        try {
            const response = await fetch('/api/demo-images');
            const data = await response.json();
            if (data.success && data.images && data.images.length > 0) {
                const demoSection = document.getElementById('demoSection');
                const demoGrid = document.getElementById('demoGrid');
                demoGrid.innerHTML = '';
                
                data.images.forEach((filename, idx) => {
                    const item = document.createElement('div');
                    item.className = 'demo-item';
                    item.innerHTML = `
                        <img src="/api/demo-image/${filename}" alt="Demo image ${idx + 1}">
                        <div class="demo-label">Image ${idx + 1}</div>
                    `;
                    item.addEventListener('click', () => {
                        handleDemoImageClick(filename);
                    });
                    demoGrid.appendChild(item);
                });
                
                demoSection.style.display = 'block';
            }
        } catch (error) {
            console.error('Error loading demo images:', error);
        }
    }

    async function handleDemoImageClick(filename) {
        const imageUrl = `/api/demo-image/${filename}`;
        imagePreview.src = imageUrl;
        resultSection.style.display = 'grid';
        greedyCard.classList.remove('show');
        beamCard.classList.remove('show');
        
        try {
            loader.style.display = 'block';
            const imgResponse = await fetch(imageUrl);
            const blob = await imgResponse.blob();
            const file = new File([blob], filename, { type: blob.type });
            generateCaption(file);
        } catch (error) {
            console.error('Error fetching demo image file:', error);
            loader.style.display = 'none';
        }
    }

    // Initial load of demo images
    loadDemoImages();

    // 2. Tab 2: Training Stats Rendering
    let lossChartInstance = null;

    async function loadStats() {
        if (lossChartInstance !== null) return;
        
        try {
            const response = await fetch('/api/history');
            const data = await response.json();
            if (data.success && data.history) {
                document.getElementById('noStatsInfo').style.display = 'none';
                document.getElementById('statsContainer').style.display = 'block';
                
                const history = data.history;
                const epochsCount = history.train_loss.length;
                
                document.getElementById('epochsMetric').textContent = epochsCount;
                document.getElementById('trainLossMetric').textContent = history.train_loss[epochsCount - 1].toFixed(4);
                document.getElementById('valLossMetric').textContent = history.val_loss[epochsCount - 1].toFixed(4);
                
                // Draw Chart.js Line Chart
                const ctx = document.getElementById('lossChart').getContext('2d');
                const epochsLabels = Array.from({ length: epochsCount }, (_, i) => i + 1);
                
                lossChartInstance = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: epochsLabels,
                        datasets: [
                            {
                                label: 'Train Loss',
                                data: history.train_loss,
                                borderColor: '#6366F1',
                                backgroundColor: 'rgba(99, 102, 241, 0.05)',
                                tension: 0.2,
                                fill: true,
                                borderWidth: 2,
                                pointBackgroundColor: '#6366F1',
                                pointHoverRadius: 6
                            },
                            {
                                label: 'Val Loss',
                                data: history.val_loss,
                                borderColor: '#06B6D4',
                                backgroundColor: 'rgba(6, 182, 212, 0.05)',
                                tension: 0.2,
                                fill: true,
                                borderWidth: 2,
                                pointBackgroundColor: '#06B6D4',
                                pointHoverRadius: 6
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                labels: {
                                    color: '#94A3B8',
                                    font: { family: 'Plus Jakarta Sans', size: 12 }
                                }
                            },
                            tooltip: {
                                mode: 'index',
                                intersect: false
                            }
                        },
                        scales: {
                            x: {
                                title: { display: true, text: 'Epoch', color: '#94A3B8' },
                                grid: { color: 'rgba(255, 255, 255, 0.03)' },
                                ticks: { color: '#64748B' }
                            },
                            y: {
                                title: { display: true, text: 'Loss', color: '#94A3B8' },
                                grid: { color: 'rgba(255, 255, 255, 0.03)' },
                                ticks: { color: '#64748B' }
                            }
                        }
                    }
                });
            } else {
                showNoStats();
            }
        } catch (error) {
            console.error('Error fetching training stats:', error);
            showNoStats();
        }
    }

    function showNoStats() {
        document.getElementById('statsContainer').style.display = 'none';
        document.getElementById('noStatsInfo').style.display = 'block';
    }

    // 3. Tab 3: Vocabulary Explorer Filtering
    let vocabData = null;

    async function loadVocab() {
        if (vocabData !== null) return;
        
        try {
            const response = await fetch('/api/vocab');
            const data = await response.json();
            if (data.success) {
                vocabData = data;
                
                // Ingest special tokens
                const tokensRow = document.getElementById('specialTokensRow');
                tokensRow.innerHTML = `
                    <div class="special-token-badge">
                        <span class="special-token-label">Pad:</span>
                        <span class="special-token-val">${data.special_tokens.pad}</span>
                    </div>
                    <div class="special-token-badge">
                        <span class="special-token-label">End:</span>
                        <span class="special-token-val">${data.special_tokens.end}</span>
                    </div>
                    <div class="special-token-badge">
                        <span class="special-token-label">Unk:</span>
                        <span class="special-token-val">${data.special_tokens.unk}</span>
                    </div>
                `;
                
                renderVocabWords('');
            }
        } catch (error) {
            console.error('Error loading vocabulary:', error);
        }
    }

    function renderVocabWords(query) {
        if (!vocabData) return;
        
        const grid = document.getElementById('vocabGrid');
        grid.innerHTML = '';
        
        const countDisplay = document.getElementById('vocabCount');
        const normalizedQuery = query.trim().toLowerCase();
        
        const filtered = vocabData.words.filter(word => word.toLowerCase().includes(normalizedQuery));
        
        // Limit word render list for high performance scrolling
        const itemsToRender = filtered.slice(0, 200);
        
        itemsToRender.forEach((word) => {
            const idx = vocabData.words.indexOf(word);
            const item = document.createElement('div');
            item.className = 'vocab-word-item';
            item.innerHTML = `
                <span class="vocab-word-index">#${idx}</span>
                <span class="vocab-word-text">${word}</span>
            `;
            grid.appendChild(item);
        });
        
        if (filtered.length > 200) {
            const moreItem = document.createElement('div');
            moreItem.className = 'vocab-word-item';
            moreItem.style.gridColumn = '1 / -1';
            moreItem.style.textAlign = 'center';
            moreItem.style.justifyContent = 'center';
            moreItem.innerHTML = `<span class="vocab-word-index">... and ${filtered.length - 200} more matching words</span>`;
            grid.appendChild(moreItem);
        }
        
        countDisplay.textContent = `Showing ${Math.min(filtered.length, 200)} of ${filtered.length} words`;
    }

    // Input event for search filter
    document.getElementById('vocabSearchInput').addEventListener('input', (e) => {
        renderVocabWords(e.target.value);
    });
});
