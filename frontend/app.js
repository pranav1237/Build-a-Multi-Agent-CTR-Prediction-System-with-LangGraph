// Global Application State
let currentDataset = null;
let currentRunId = null;
let currentApiKey = "";
let pollInterval = null;
let previousLogCount = 0;
let rocChartInstance = null;
let importanceChartInstance = null;
let pipelineMetadata = null;

// DOM Elements
const navItems = document.querySelectorAll('.nav-item');
const viewPanels = document.querySelectorAll('.view-panel');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const generateDemoBtn = document.getElementById('generate-demo-btn');
const fileDetailsCard = document.getElementById('file-details-card');
const dataPreviewContainer = document.getElementById('data-preview-container');
const previewTable = document.getElementById('preview-table');
const proceedToConfigBtn = document.getElementById('proceed-to-config-btn');
const startPipelineBtn = document.getElementById('start-pipeline-btn');
const targetColSelect = document.getElementById('target-col-select');
const excludeColsInput = document.getElementById('exclude-cols-input');
const userInstructions = document.getElementById('user-instructions');
const logTerminal = document.getElementById('log-terminal');
const pipelineStatusText = document.getElementById('pipeline-status-text');
const runCompletionBanner = document.getElementById('run-completion-banner');
const viewAnalyticsBtn = document.getElementById('view-analytics-btn');
const goToSandboxBtn = document.getElementById('go-to-sandbox-btn');
const llmStatus = document.getElementById('llm-status');
const settingsBtn = document.getElementById('settings-btn');
const settingsModal = document.getElementById('settings-modal');
const closeModalBtn = document.getElementById('close-modal-btn');
const saveKeyBtn = document.getElementById('save-key-btn');
const apiKeyInput = document.getElementById('api-key-input');
const sandboxFieldsContainer = document.getElementById('sandbox-fields-container');
const sandboxPredictionForm = document.getElementById('sandbox-prediction-form');

// All API requests use the current deployed origin so the static frontend and FastAPI backend stay together on Vercel.
function apiUrl(path) {
    return new URL(path, window.location.origin).toString();
}

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
    setupNavigation();
    setupUploadHandlers();
    setupModalHandlers();
    checkServerApiKeyStatus();
});

// 1. Navigation Controller
function setupNavigation() {
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            if (item.classList.contains('disabled')) return;
            switchView(item.getAttribute('data-target'));
        });
    });
}

function switchView(targetViewId) {
    // Update nav items
    navItems.forEach(item => {
        if (item.getAttribute('data-target') === targetViewId) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });

    // Update panels
    viewPanels.forEach(panel => {
        if (panel.id === targetViewId) {
            panel.classList.add('active');
        } else {
            panel.classList.remove('active');
        }
    });
}

function enableNavTab(tabId) {
    const tab = document.getElementById(tabId);
    if (tab) {
        tab.classList.remove('disabled');
    }
}

// 2. API Key Modal & Status Checks
function setupModalHandlers() {
    settingsBtn.addEventListener('click', () => {
        settingsModal.classList.remove('hidden');
        apiKeyInput.value = currentApiKey;
    });

    closeModalBtn.addEventListener('click', () => {
        settingsModal.classList.add('hidden');
    });

    settingsModal.addEventListener('click', (e) => {
        if (e.target === settingsModal) {
            settingsModal.classList.add('hidden');
        }
    });

    saveKeyBtn.addEventListener('click', () => {
        currentApiKey = apiKeyInput.value.trim();
        settingsModal.classList.add('hidden');
        updateLlmBadge(currentApiKey ? true : false, currentApiKey ? "Custom LLM Enabled" : null);
    });
}

async function checkServerApiKeyStatus() {
    try {
        const res = await fetch(apiUrl("/api-key-status");
        const data = await res.json();
        updateLlmBadge(data.has_key, data.has_key ? "Server LLM Connected" : "Local Heuristic Mode");
    } catch (e) {
        updateLlmBadge(false, "Offline Heuristic Mode");
    }
}

function updateLlmBadge(hasKey, customLabel = null) {
    if (hasKey) {
        llmStatus.className = "status-badge llm";
        llmStatus.querySelector(".status-label").textContent = customLabel || "Google Gemini LLM Connected";
    } else {
        llmStatus.className = "status-badge fallback";
        llmStatus.querySelector(".status-label").textContent = customLabel || "Local Heuristic Mode";
    }
}

// 3. Upload & Dataset Inspection
function setupUploadHandlers() {
    // Drop Zone Events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) handleFileUpload(files[0]);
    });

    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFileUpload(e.target.files[0]);
    });

    generateDemoBtn.addEventListener('click', handleGenerateDemo);
    proceedToConfigBtn.addEventListener('click', () => {
        enableNavTab('nav-config');
        switchView('config-view');
    });
}

async function handleFileUpload(file) {
    showLoadingState(true, "Uploading and analyzing CSV...");
    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(apiUrl("/upload", {
            method: "POST",
            body: formData
        });
        
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        displayDatasetPreview(data);
    } catch (e) {
        alert("Upload failed: " + e.message);
    } finally {
        showLoadingState(false);
    }
}

async function handleGenerateDemo() {
    showLoadingState(true, "Generating synthetic CTR records...");
    const formData = new FormData();
    formData.append("num_rows", 2500);

    try {
        const res = await fetch(apiUrl("/generate-sample", {
            method: "POST",
            body: formData
        });
        
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        displayDatasetPreview(data);
    } catch (e) {
        alert("Demo generation failed: " + e.message);
    } finally {
        showLoadingState(false);
    }
}

function showLoadingState(isLoading, message = "") {
    if (isLoading) {
        dropZone.style.pointerEvents = "none";
        dropZone.style.opacity = "0.6";
        dropZone.querySelector('h3').textContent = message;
    } else {
        dropZone.style.pointerEvents = "auto";
        dropZone.style.opacity = "1";
        dropZone.querySelector('h3').textContent = "Drag & drop your CSV file here";
    }
}

function displayDatasetPreview(data) {
    currentDataset = data;
    
    // File Card
    document.getElementById('info-filename').textContent = data.filename;
    document.getElementById('info-rows').textContent = data.total_rows.toLocaleString();
    document.getElementById('info-cols-count').textContent = data.columns.length;
    fileDetailsCard.classList.remove('hidden');

    // Populate dropdown in config
    targetColSelect.innerHTML = "";
    data.columns.forEach(col => {
        const opt = document.createElement('option');
        opt.value = col;
        opt.textContent = col;
        // Default select 'clicked' or common variants
        if (col.toLowerCase() === 'clicked' || col.toLowerCase() === 'click' || col.toLowerCase() === 'label') {
            opt.selected = true;
        }
        targetColSelect.appendChild(opt);
    });

    // Populate preview table
    const thead = previewTable.querySelector('thead');
    const tbody = previewTable.querySelector('tbody');
    thead.innerHTML = "";
    tbody.innerHTML = "";

    // Headers
    const trHead = document.createElement('tr');
    data.columns.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col;
        trHead.appendChild(th);
    });
    thead.appendChild(trHead);

    // Rows
    data.preview.forEach(row => {
        const trRow = document.createElement('tr');
        data.columns.forEach(col => {
            const td = document.createElement('td');
            td.textContent = row[col] !== null ? row[col] : 'NaN';
            trRow.appendChild(td);
        });
        tbody.appendChild(trRow);
    });

    dataPreviewContainer.classList.remove('hidden');
    dataPreviewContainer.scrollIntoView({ behavior: 'smooth' });
}

// 4. LangGraph Multi-Agent Execution Control
startPipelineBtn.addEventListener('click', startPipelineRun);

async function startPipelineRun() {
    if (!currentDataset) return;
    
    // Gather selections
    const targetCol = targetColSelect.value;
    const excludeCols = excludeColsInput.value;
    const instructions = userInstructions.value;
    
    // Gather checked models
    const checkedModels = [];
    document.querySelectorAll('.model-checkbox:checked').forEach(cb => {
        checkedModels.push(cb.value);
    });

    if (checkedModels.length === 0) {
        alert("Please select at least one algorithm model to train.");
        return;
    }

    const formData = new FormData();
    formData.append("dataset_path", currentDataset.dataset_path);
    formData.append("target_column", targetCol);
    formData.append("exclude_columns", excludeCols);
    formData.append("user_instructions", instructions);
    if (currentApiKey) {
        formData.append("api_key", currentApiKey);
    }

    try {
        startPipelineBtn.disabled = true;
        startPipelineBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Initializing Agents...';
        
        const res = await fetch(apiUrl("/run", {
            method: "POST",
            body: formData
        });

        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        
        currentRunId = data.run_id;
        previousLogCount = 0;
        
        // Reset console terminal
        logTerminal.innerHTML = '<div class="log-line system-msg">> Multi-Agent run initialized with session token: ' + currentRunId + '</div>';
        
        // Reset completion banner
        runCompletionBanner.classList.add('hidden');
        
        // Enable Nav and Navigate
        enableNavTab('nav-run');
        switchView('run-view');
        
        // Start Polling
        pipelineStatusText.className = "run-status-badge started";
        pipelineStatusText.textContent = "RUNNING";
        
        pollInterval = setInterval(pollRunStatus, 1500);
    } catch (e) {
        alert("Failed to start run: " + e.message);
        startPipelineBtn.disabled = false;
        startPipelineBtn.innerHTML = '<i class="fa-solid fa-play"></i> Initialize Multi-Agent Pipeline';
    }
}

async function pollRunStatus() {
    if (!currentRunId) return;

    try {
        const res = await fetch(apiUrl(`/status/${currentRunId}`));
        if (!res.ok) throw new Error("Connection error");
        const data = await res.json();

        const state = data.state;
        updateLogsTerminal(state.logs);
        updateGraphVisualization(state.active_agent, state.status, state.active_target);

        if (data.completed) {
            clearInterval(pollInterval);
            pollInterval = null;
            startPipelineBtn.disabled = false;
            startPipelineBtn.innerHTML = '<i class="fa-solid fa-play"></i> Initialize Multi-Agent Pipeline';

            if (data.error) {
                pipelineStatusText.className = "run-status-badge failed";
                pipelineStatusText.textContent = "FAILED";
                appendConsoleLine("System", `Workflow execution failed: ${data.error}`, "error-tag");
            } else {
                pipelineStatusText.className = "run-status-badge completed";
                pipelineStatusText.textContent = "COMPLETED";
                
                // Show completion banner
                runCompletionBanner.classList.remove('hidden');
                enableNavTab('nav-analytics');
                enableNavTab('nav-sandbox');

                // Render metrics, charts and report
                pipelineMetadata = state.metadata;
                renderModelPerformance(state.train_results, state.best_model_type, state.evaluation_report);
            }
        }
    } catch (e) {
        console.error("Polling error: ", e);
    }
}

function updateLogsTerminal(logs) {
    if (!logs || logs.length === 0) return;
    
    // Only append new logs
    if (logs.length > previousLogCount) {
        for (let i = previousLogCount; i < logs.length; i++) {
            const log = logs[i];
            appendConsoleLine(log.agent, log.message, getLogClass(log.agent));
        }
        previousLogCount = logs.length;
    }
}

function appendConsoleLine(agent, message, cssClass) {
    const line = document.createElement('div');
    line.className = `log-line ${cssClass}`;
    
    const timeSpan = document.createElement('span');
    timeSpan.className = 'timestamp';
    timeSpan.textContent = `[${new Date().toLocaleTimeString()}]`;
    
    const agentSpan = document.createElement('span');
    agentSpan.className = 'agent-tag';
    agentSpan.textContent = `[${agent}]`;
    
    const textSpan = document.createElement('span');
    textSpan.textContent = message;
    
    line.appendChild(timeSpan);
    line.appendChild(agentSpan);
    line.appendChild(textSpan);
    
    logTerminal.appendChild(line);
    logTerminal.scrollTop = logTerminal.scrollHeight;
}

function getLogClass(agent) {
    const norm = agent.toLowerCase().replace(/_/g, '-');
    if (norm.includes('supervisor')) return 'supervisor-tag';
    if (norm.includes('data')) return 'data-agent-tag';
    if (norm.includes('model')) return 'model-agent-tag';
    if (norm.includes('evaluator')) return 'evaluator-agent-tag';
    if (norm.includes('error') || norm.includes('system-error')) return 'error-tag';
    return 'system-msg';
}

function updateGraphVisualization(activeAgent, status, activeTarget) {
    // Reset all nodes and link lines
    document.querySelectorAll('.graph-node').forEach(node => {
        node.classList.remove('active');
        node.classList.remove('completed');
    });
    document.querySelectorAll('.graph-path').forEach(path => {
        path.classList.remove('active');
    });

    const activeNodeId = activeAgent ? 'node-' + activeAgent.toLowerCase().replace(/ /g, '_') : null;
    const activeNode = document.getElementById(activeNodeId);
    
    // Set active node
    if (activeNode) {
        activeNode.classList.add('active');
    }

    // Set paths glow based on state transitions
    if (activeAgent === "Supervisor") {
        if (status === "preprocessing" || activeTarget === "data_agent") {
            document.getElementById('link-sup-data').classList.add('active');
        } else if (status === "training" || activeTarget === "model_agent") {
            document.getElementById('link-sup-model').classList.add('active');
        } else if (status === "evaluating" || activeTarget === "evaluator_agent") {
            document.getElementById('link-sup-eval').classList.add('active');
        }
    } else if (activeAgent === "Data Agent") {
        document.getElementById('link-data-sup').classList.add('active');
        document.getElementById('node-supervisor').classList.add('completed');
    } else if (activeAgent === "Model Agent") {
        document.getElementById('link-model-sup').classList.add('active');
        document.getElementById('node-supervisor').classList.add('completed');
        document.getElementById('node-data_agent').classList.add('completed');
    } else if (activeAgent === "Evaluator Agent") {
        document.getElementById('link-eval-sup').classList.add('active');
        document.getElementById('node-supervisor').classList.add('completed');
        document.getElementById('node-data_agent').classList.add('completed');
        document.getElementById('node-model_agent').classList.add('completed');
    }
}

viewAnalyticsBtn.addEventListener('click', () => switchView('analytics-view'));

// 5. Analytics View (Metrics, Charts, Report)
function renderModelPerformance(trainResults, bestModelType, rawReport) {
    const bestMetrics = trainResults[bestModelType].metrics;

    // Fill Scorecards
    document.getElementById('champion-model-name').textContent = bestModelType.replace(/_/g, ' ').toUpperCase();
    document.getElementById('champion-auc').textContent = bestMetrics.auc.toFixed(4);
    document.getElementById('champion-logloss').textContent = bestMetrics.logloss.toFixed(4);
    document.getElementById('champion-accuracy').textContent = (bestMetrics.accuracy * 100).toFixed(1) + "%";

    // Set download link
    document.getElementById('download-model-link').href = `/download/model`;

    // Render curves chart (Chart.js)
    renderRocChart(trainResults);
    
    // Render Feature Importance chart
    renderFeatureImportanceChart(trainResults[bestModelType].feature_importance);

    // Render Evaluator Report
    const reportContainer = document.getElementById('markdown-report-container');
    reportContainer.innerHTML = parseMarkdown(rawReport);

    // Build Inference Sandbox form fields
    buildSandboxFields();
}

function renderRocChart(trainResults) {
    const ctx = document.getElementById('roc-chart').getContext('2d');
    
    if (rocChartInstance) {
        rocChartInstance.destroy();
    }

    const datasets = [];
    const colors = {
        logistic_regression: '#06b6d4',
        random_forest: '#10b981',
        xgboost: '#8b5cf6',
        lightgbm: '#f59e0b'
    };

    Object.keys(trainResults).forEach(model => {
        if (model === 'llm_insights') return;
        const res = trainResults[model];
        if (res.charts && res.charts.roc) {
            const dataPoints = res.charts.roc.fpr.map((fpr, i) => ({
                x: fpr,
                y: res.charts.roc.tpr[i]
            }));

            datasets.push({
                label: model.replace(/_/g, ' ').toUpperCase(),
                data: dataPoints,
                borderColor: colors[model] || '#94a3b8',
                borderWidth: 2,
                fill: false,
                tension: 0.1,
                pointRadius: 0
            });
        }
    });

    // Add baseline random guessing diagonal
    datasets.push({
        label: 'Random Guess',
        data: [{x:0, y:0}, {x:1, y:1}],
        borderColor: 'rgba(255,255,255,0.15)',
        borderWidth: 1,
        borderDash: [5, 5],
        fill: false,
        pointRadius: 0
    });

    rocChartInstance = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: { display: true, text: 'False Positive Rate', color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    title: { display: true, text: 'True Positive Rate', color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#e2e8f0', font: { size: 10 } }
                }
            }
        }
    });
}

function renderFeatureImportanceChart(importances) {
    const ctx = document.getElementById('importance-chart').getContext('2d');
    
    if (importanceChartInstance) {
        importanceChartInstance.destroy();
    }

    const labels = Object.keys(importances);
    const values = Object.values(importances);

    // Create nice horizontal gradients
    const gradient = ctx.createLinearGradient(0, 0, 400, 0);
    gradient.addColorStop(0, '#8b5cf6');
    gradient.addColorStop(1, '#06b6d4');

    importanceChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Relative Importance Score',
                data: values,
                backgroundColor: gradient,
                borderRadius: 4,
                borderWidth: 0
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#e2e8f0', font: { size: 11 } }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// Simple Markdown to HTML Parser
function parseMarkdown(md) {
    if (!md) return "";
    let html = md;
    
    // Escaping simple HTML tags to avoid breaks
    html = html.replace(/</g, "&lt;").replace(/>/g, "&gt;");
    
    // Headers
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
    
    // Bold
    html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
    
    // Blockquotes
    html = html.replace(/^\&gt;\s(.*$)/gim, '<blockquote>$1</blockquote>');
    
    // Lists
    html = html.replace(/^\-\s(.*$)/gim, '<li>$1</li>');
    
    // Group adjacent li items into ul lists
    html = html.replace(/(<li>.*?<\/li>)+/g, '<ul>$&</ul>');
    
    // Paragraph splits (blank lines to p tags)
    html = html.split(/\n\n+/).map(p => {
        if (p.trim().startsWith('<h') || p.trim().startsWith('<ul') || p.trim().startsWith('<blockquote')) {
            return p;
        }
        return `<p>${p.replace(/\n/g, '<br>')}</p>`;
    }).join('');
    
    return html;
}

goToSandboxBtn.addEventListener('click', () => switchView('sandbox-view'));

// 6. Interactive Sandbox Panel Builders
function buildSandboxFields() {
    if (!pipelineMetadata) return;

    sandboxFieldsContainer.innerHTML = "";
    
    const numericals = pipelineMetadata.numerical_features || [];
    const categoricals = pipelineMetadata.categorical_features || [];

    // Combine features to render
    // 1. Render Categorical Selects
    categoricals.forEach(cat => {
        const fieldCard = document.createElement('div');
        fieldCard.className = 'form-group';
        
        const label = document.createElement('label');
        label.textContent = cat.replace(/_/g, ' ').toUpperCase();
        
        const select = document.createElement('select');
        select.name = cat;
        select.className = 'form-control';
        
        // Find unique categories from our CSV Preview rows
        const uniqueValues = new Set();
        currentDataset.preview.forEach(row => {
            if (row[cat] !== undefined && row[cat] !== null) {
                uniqueValues.add(row[cat]);
            }
        });

        // Add options
        if (uniqueValues.size === 0) {
            // Default select options
            ["Option A", "Option B", "Option C"].forEach(opt => {
                const o = document.createElement('option');
                o.value = opt;
                o.textContent = opt;
                select.appendChild(o);
            });
        } else {
            uniqueValues.forEach(val => {
                const o = document.createElement('option');
                o.value = val;
                o.textContent = val;
                select.appendChild(o);
            });
        }
        
        fieldCard.appendChild(label);
        fieldCard.appendChild(select);
        sandboxFieldsContainer.appendChild(fieldCard);
    });

    // 2. Render Numerical Inputs / Sliders
    numericals.forEach(num => {
        const fieldCard = document.createElement('div');
        fieldCard.className = 'form-group';
        
        const label = document.createElement('label');
        label.textContent = num.replace(/_/g, ' ').toUpperCase();
        
        // Gather min/max values from preview to set slider ranges
        const values = currentDataset.preview.map(row => Number(row[num])).filter(v => !isNaN(v));
        const minVal = values.length ? Math.min(...values) : 0;
        const maxVal = values.length ? Math.max(...values) : 100;
        const avgVal = values.length ? Math.round(values.reduce((a,b)=>a+b, 0) / values.length) : 50;

        const inputGroup = document.createElement('div');
        inputGroup.style.display = 'flex';
        inputGroup.style.alignItems = 'center';
        inputGroup.style.gap = '10px';

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.name = num;
        slider.min = minVal;
        slider.max = maxVal;
        slider.value = avgVal;
        slider.className = 'form-control';
        slider.style.flex = '1';
        slider.style.padding = '0';
        
        const valIndicator = document.createElement('span');
        valIndicator.textContent = avgVal;
        valIndicator.style.fontFamily = 'var(--font-mono)';
        valIndicator.style.fontSize = '13px';
        valIndicator.style.minWidth = '40px';
        valIndicator.style.textAlign = 'right';

        slider.addEventListener('input', (e) => {
            valIndicator.textContent = e.target.value;
        });

        inputGroup.appendChild(slider);
        inputGroup.appendChild(valIndicator);

        fieldCard.appendChild(label);
        fieldCard.appendChild(inputGroup);
        sandboxFieldsContainer.appendChild(fieldCard);
    });
}

// Sandbox Form Submission & Prediction Gauge update
sandboxPredictionForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(sandboxPredictionForm);
    const featuresPayload = {};

    // Populate payload features
    for (let [key, val] of formData.entries()) {
        // Check if feature is numeric
        if (pipelineMetadata.numerical_features.includes(key)) {
            featuresPayload[key] = Number(val);
        } else {
            featuresPayload[key] = val;
        }
    }

    try {
        const res = await fetch(apiUrl("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ features: featuresPayload })
        });

        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        
        updatePredictionOutcome(data.click_probability);
    } catch (e) {
        alert("Prediction failed: " + e.message);
    }
});

function updatePredictionOutcome(probability) {
    const percentage = Math.round(probability * 100);
    const dial = document.getElementById('probability-dial');
    const valContainer = document.getElementById('probability-value');
    const verdictTitle = document.getElementById('verdict-title');
    const verdictDesc = document.getElementById('verdict-desc');
    const explanationBox = document.getElementById('sandbox-explanation');

    // Update circular progress conic gradient
    let color = 'var(--accent)';
    if (probability > 0.45) {
        color = 'var(--success)';
        verdictTitle.textContent = "High Click Likelihood";
        verdictTitle.style.color = 'var(--success)';
        verdictDesc.textContent = "This impression satisfies several criteria of active user click behavior.";
    } else if (probability > 0.15) {
        color = 'var(--warning)';
        verdictTitle.textContent = "Moderate Click Likelihood";
        verdictTitle.style.color = 'var(--warning)';
        verdictDesc.textContent = "The user has shown occasional clicks under these feature demographics.";
    } else {
        color = 'var(--error)';
        verdictTitle.textContent = "Low Click Likelihood";
        verdictTitle.style.color = 'var(--error)';
        verdictDesc.textContent = "Features suggest a highly passive context or low historical conversion.";
    }

    dial.style.background = `conic-gradient(${color} ${percentage * 3.6}deg, rgba(255,255,255,0.05) 0deg)`;
    valContainer.textContent = `${percentage}%`;
    valContainer.style.color = color;

    // Build brief text explanation based on features
    explanationBox.classList.remove('hidden');
    explanationBox.innerHTML = `<strong>Champion Model Inference Verdict:</strong><br>
    The trained model evaluated this input and mapped it to a <strong>${probability.toFixed(4)}</strong> positive outcome probability. 
    Ad placement and bid modifiers should scale according to this click coefficient.`;
}
