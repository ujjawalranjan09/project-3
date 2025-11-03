// KPI Panel and Dashboard Interaction Logic

const API_BASE_URL = 'http://localhost:5000/api';

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log('Railway Traffic Control System Dashboard Initialized');

    // Load KPI metrics
    loadKPIMetrics();
    setInterval(loadKPIMetrics, 30000); // Update every 30 seconds

    // Initialize charts
    initializeCharts();

    // Set up form handlers
    setupConflictForm();
    setupDelayPrediction();
    setupSimulation();
});

// Load and display KPI metrics
async function loadKPIMetrics() {
    try {
        const response = await fetch(`${API_BASE_URL}/metrics/kpi`);
        const data = await response.json();

        // Update throughput
        document.getElementById('throughputValue').textContent = data.throughput.trains_per_hour;
        updateProgressBar('throughputProgress', data.throughput.percentage);

        // Update punctuality
        document.getElementById('punctualityValue').textContent = data.punctuality.on_time_percentage.toFixed(1);
        updateProgressBar('punctualityProgress', data.punctuality.on_time_percentage);

        // Update average delay
        document.getElementById('avgDelayValue').textContent = data.average_delay.current.toFixed(1);

        // Update conflicts
        document.getElementById('conflictsValue').textContent = data.conflicts.pending;
        if (data.conflicts.pending > 5) {
            document.getElementById('conflictAlert').innerHTML = '⚠️ High alert';
            document.getElementById('conflictAlert').style.color = '#e74c3c';
        }

        // Update last update time
        const now = new Date();
        document.getElementById('lastUpdate').textContent = 
            `Last updated: ${now.toLocaleTimeString()}`;

    } catch (error) {
        console.error('Error loading KPI metrics:', error);
    }
}

function updateProgressBar(elementId, percentage) {
    const bar = document.getElementById(elementId);
    bar.style.width = `${percentage}%`;
}

// Conflict Detection Form
function setupConflictForm() {
    const form = document.getElementById('conflictForm');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        const formData = {
            trains_in_section: parseInt(document.getElementById('trainsInSection').value),
            available_platforms: parseInt(document.getElementById('availablePlatforms').value),
            platform_utilization_pct: parseFloat(document.getElementById('platformUtil').value),
            weather_severity: parseFloat(document.getElementById('weatherSeverity').value),
            rainfall_mm: parseFloat(document.getElementById('rainfall').value),
            fog_intensity: parseFloat(document.getElementById('fogIntensity').value),
            temperature_c: parseFloat(document.getElementById('temperature').value),
            is_peak_hour: parseInt(document.getElementById('isPeakHour').value)
        };

        try {
            const response = await fetch(`${API_BASE_URL}/predict/conflict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();
            displayConflictResult(result);

            // Store for delay prediction
            window.currentFormData = formData;

        } catch (error) {
            console.error('Error predicting conflict:', error);
            alert('Error analyzing conflict risk. Please check if the API is running.');
        }
    });
}

function displayConflictResult(result) {
    const resultBox = document.getElementById('conflictResult');
    const riskBadge = document.getElementById('riskBadge');
    const conflictProb = document.getElementById('conflictProb');
    const recommendations = document.getElementById('recommendations');

    // Show result box
    resultBox.classList.remove('hidden');

    // Update risk badge
    riskBadge.textContent = result.risk_level;
    riskBadge.className = 'risk-badge ' + result.risk_level.toLowerCase();

    // Update probability
    conflictProb.textContent = `${(result.conflict_probability * 100).toFixed(1)}%`;

    // Display recommendations
    if (result.recommendations && result.recommendations.length > 0) {
        let recHTML = '<h4>Recommendations:</h4><ul>';
        result.recommendations.forEach(rec => {
            recHTML += `<li><strong>${rec.priority}:</strong> ${rec.action} - ${rec.details}</li>`;
        });
        recHTML += '</ul>';
        recommendations.innerHTML = recHTML;
    } else {
        recommendations.innerHTML = '<p>No specific recommendations at this time.</p>';
    }
}

// Delay Prediction
function setupDelayPrediction() {
    const btn = document.getElementById('predictDelayBtn');

    btn.addEventListener('click', async function() {
        if (!window.currentFormData) {
            alert('Please run conflict analysis first');
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/predict/delay`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(window.currentFormData)
            });

            const result = await response.json();
            displayDelayResult(result);

        } catch (error) {
            console.error('Error predicting delay:', error);
            alert('Error predicting delay. Please check if the API is running.');
        }
    });
}

function displayDelayResult(result) {
    const resultBox = document.getElementById('delayResult');
    const severityBadge = document.getElementById('severityBadge');
    const predictedDelay = document.getElementById('predictedDelay');
    const delayImpact = document.getElementById('delayImpact');
    const mitigation = document.getElementById('mitigationStrategies');

    // Show result box
    resultBox.classList.remove('hidden');

    // Update severity badge
    severityBadge.textContent = result.severity;
    severityBadge.className = 'severity-badge ' + result.severity.toLowerCase();

    // Update delay value
    predictedDelay.textContent = result.predicted_delay_minutes.toFixed(1);

    // Update impact
    delayImpact.textContent = result.impact_description;

    // Display mitigation strategies
    if (result.mitigation_strategies && result.mitigation_strategies.length > 0) {
        let mitHTML = '<h4>Mitigation Strategies:</h4><ul>';
        result.mitigation_strategies.forEach(strategy => {
            mitHTML += `<li><strong>${strategy.strategy}:</strong> ${strategy.implementation} 
                       <br><em>Expected reduction: ${strategy.expected_reduction}</em></li>`;
        });
        mitHTML += '</ul>';
        mitigation.innerHTML = mitHTML;
    }
}

// What-If Simulation
function setupSimulation() {
    const btn = document.getElementById('runSimulation');

    btn.addEventListener('click', async function() {
        if (!window.currentFormData) {
            alert('Please run conflict analysis first');
            return;
        }

        const simTrains = document.getElementById('simTrains').value;
        const simPlatforms = document.getElementById('simPlatforms').value;

        if (!simTrains && !simPlatforms) {
            alert('Please enter at least one modification');
            return;
        }

        const modifications = {};
        if (simTrains) modifications.trains_in_section = parseInt(simTrains);
        if (simPlatforms) modifications.available_platforms = parseInt(simPlatforms);

        try {
            const response = await fetch(`${API_BASE_URL}/simulate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    baseline: window.currentFormData,
                    modifications: modifications
                })
            });

            const result = await response.json();
            displaySimulationResult(result);

        } catch (error) {
            console.error('Error running simulation:', error);
            alert('Error running simulation. Please check if the API is running.');
        }
    });
}

function displaySimulationResult(result) {
    const resultBox = document.getElementById('simulationResult');
    const baselineDelay = document.getElementById('baselineDelay');
    const modifiedDelay = document.getElementById('modifiedDelay');
    const improvement = document.getElementById('improvement');

    // Show result box
    resultBox.classList.remove('hidden');

    // Update values
    baselineDelay.textContent = `${result.baseline.predicted_delay_minutes.toFixed(1)} min`;
    modifiedDelay.textContent = `${result.modified_scenario.predicted_delay_minutes.toFixed(1)} min`;

    const improvementText = result.delay_reduction_minutes > 0 
        ? `↓ ${result.delay_reduction_minutes.toFixed(1)} min (${result.improvement_percentage.toFixed(1)}%)`
        : `↑ ${Math.abs(result.delay_reduction_minutes).toFixed(1)} min worse`;

    improvement.textContent = improvementText;
    improvement.style.color = result.delay_reduction_minutes > 0 ? '#27ae60' : '#e74c3c';
}

// Initialize Charts
let throughputChart, delayTrendChart;

function initializeCharts() {
    // Throughput Chart
    const throughputCtx = document.getElementById('throughputChart').getContext('2d');
    throughputChart = new Chart(throughputCtx, {
        type: 'line',
        data: {
            labels: ['6:00', '9:00', '12:00', '15:00', '18:00', '21:00'],
            datasets: [{
                label: 'Trains/Hour',
                data: [35, 48, 42, 45, 52, 38],
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Throughput Trend'
                }
            }
        }
    });

    // Delay Trend Chart
    const delayCtx = document.getElementById('delayTrendChart').getContext('2d');
    delayTrendChart = new Chart(delayCtx, {
        type: 'bar',
        data: {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            datasets: [{
                label: 'Avg Delay (min)',
                data: [12, 8, 10, 7, 9, 6, 5],
                backgroundColor: '#e74c3c'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Weekly Delay Trend'
                }
            }
        }
    });
}
