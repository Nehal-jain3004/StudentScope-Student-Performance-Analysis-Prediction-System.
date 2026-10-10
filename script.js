/**
 * StudentScope — Student Performance Analysis & Prediction System
 * Connected Frontend Logic: Live Flask API Integration for Analytics & ML Inference
 */

// Global state holders
let studentRecords = [];
let filteredRecords = [];
let currentPage = 1;
const pageSize = 5;
const chartInstances = {};

document.addEventListener('DOMContentLoaded', () => {
    // 1. Navigation router & mobile drawer
    initNavigation();

    // 2. Global search bar
    initGlobalSearch();

    // 3. Initialize prediction form handler
    initPredictionEngine();

    // 4. Initialize charts with default skeleton
    initCharts();

    // 5. Fetch live data from Flask backend
    fetchDashboardData();
    fetchModelInsights();
});

/* ==========================================================================
   1. NAVIGATION ROUTER & SIDEBAR CONTROLS
   ========================================================================== */
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const contentSections = document.querySelectorAll('.content-section');
    const pageTitle = document.getElementById('pageTitle');
    const pageSubtitle = document.getElementById('pageSubtitle');
    const hamburgerBtn = document.getElementById('hamburgerBtn');
    const closeSidebarBtn = document.getElementById('closeSidebarBtn');
    const sidebar = document.getElementById('sidebar');
    const sidebarOverlay = document.getElementById('sidebarOverlay');

    const titlesMap = {
        'dashboard-section': {
            title: 'Dashboard Overview',
            subtitle: 'Student performance statistics and real-time analytical metrics'
        },
        'analysis-section': {
            title: 'Exploratory Data Analysis',
            subtitle: 'Deep dive into study habits, absences, and academic progression trends'
        },
        'predict-section': {
            title: 'Predict Student Performance',
            subtitle: 'Live machine learning inference pipeline for estimating final academic grades (G3)'
        },
        'insights-section': {
            title: 'Model Insights & Benchmarks',
            subtitle: 'Comparative evaluation of regression algorithms and genuine feature importance'
        },
        'about-section': {
            title: 'About StudentScope',
            subtitle: 'Project objective, problem statement, system architecture, and ML methodology'
        }
    };

    function switchSection(targetSectionId) {
        navItems.forEach(item => {
            if (item.dataset.section === targetSectionId) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });

        contentSections.forEach(section => {
            if (section.id === targetSectionId) {
                section.classList.add('active');
            } else {
                section.classList.remove('active');
            }
        });

        if (titlesMap[targetSectionId]) {
            pageTitle.textContent = titlesMap[targetSectionId].title;
            pageSubtitle.textContent = titlesMap[targetSectionId].subtitle;
        }

        closeSidebar();
    }

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const sectionId = item.dataset.section;
            switchSection(sectionId);
            window.location.hash = item.querySelector('a').getAttribute('href');
        });
    });

    window.addEventListener('hashchange', () => {
        const hash = window.location.hash.replace('#', '');
        if (hash) {
            const matchingNav = Array.from(navItems).find(item => item.querySelector('a').getAttribute('href') === `#${hash}`);
            if (matchingNav) {
                switchSection(matchingNav.dataset.section);
            }
        }
    });

    if (window.location.hash) {
        const initialHash = window.location.hash.replace('#', '');
        const matchingNav = Array.from(navItems).find(item => item.querySelector('a').getAttribute('href') === `#${initialHash}`);
        if (matchingNav) {
            switchSection(matchingNav.dataset.section);
        }
    }

    hamburgerBtn.addEventListener('click', () => {
        sidebar.classList.add('open');
        sidebarOverlay.classList.add('active');
    });

    closeSidebarBtn.addEventListener('click', closeSidebar);
    sidebarOverlay.addEventListener('click', closeSidebar);

    function closeSidebar() {
        sidebar.classList.remove('open');
        sidebarOverlay.classList.remove('active');
    }
}

/* ==========================================================================
   2. API INTEGRATION: DASHBOARD DATA & STUDENT RECORDS TABLE
   ========================================================================== */
async function fetchDashboardData() {
    try {
        const response = await fetch('/api/dashboard-data');
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        const result = await response.json();

        if (result.status === 'success') {
            updateDashboardMetrics(result.summary);
            updateDashboardCharts(result.charts);

            if (result.students && Array.isArray(result.students)) {
                studentRecords = result.students;
                filteredRecords = [...studentRecords];
                initStudentTable();
            }
        } else {
            console.warn('Dashboard data returned error:', result.message);
        }
    } catch (err) {
        console.error('Failed to fetch /api/dashboard-data:', err);
    }
}

function updateDashboardMetrics(summary) {
    if (!summary) return;

    const totalStudentsElem = document.getElementById('totalStudentsCount');
    const avgFinalGradeElem = document.getElementById('avgFinalGrade');
    const avgStudyHoursElem = document.getElementById('avgStudyHours');
    const passRateElem = document.getElementById('passRate');

    if (totalStudentsElem) totalStudentsElem.textContent = summary.total_students;
    if (avgFinalGradeElem) {
        avgFinalGradeElem.innerHTML = `${summary.avg_final_grade} <span class="unit">/ 20</span>`;
    }
    if (avgStudyHoursElem) {
        avgStudyHoursElem.innerHTML = `${summary.avg_study_hours_tier} <span class="unit">tier avg</span>`;
    }
    if (passRateElem) {
        passRateElem.textContent = `${summary.pass_rate}%`;
        const subtext = passRateElem.nextElementSibling;
        if (subtext) {
            subtext.innerHTML = `<i class="fa-solid fa-check"></i> ${summary.passing_count} Students Passing`;
        }
    }

    // Update header badge
    const headerBadge = document.querySelector('.header-badge span');
    if (headerBadge) {
        headerBadge.textContent = `Dataset Size: ${summary.total_students} Students`;
    }
}

function updateDashboardCharts(chartsData) {
    if (!chartsData) return;

    // 1. Grade Distribution
    if (chartInstances.gradeDist && chartsData.grade_distribution) {
        chartInstances.gradeDist.data.datasets[0].data = chartsData.grade_distribution.counts;
        chartInstances.gradeDist.update();
    }

    // 2. Study Time vs Grade
    if (chartInstances.studyVsGrade && chartsData.study_vs_grade) {
        chartInstances.studyVsGrade.data.datasets[0].data = chartsData.study_vs_grade.g3_means;
        chartInstances.studyVsGrade.data.datasets[1].data = chartsData.study_vs_grade.g1_means;
        chartInstances.studyVsGrade.update();
    }

    // 3. Grade Progression
    if (chartInstances.progression && chartsData.progression) {
        chartInstances.progression.data.datasets[0].data = chartsData.progression.top_performers;
        chartInstances.progression.data.datasets[1].data = chartsData.progression.avg_performers;
        chartInstances.progression.data.datasets[2].data = chartsData.progression.risk_performers;
        chartInstances.progression.update();
    }

    // 4. Absences Impact Scatter
    if (chartInstances.absences && chartsData.absences_scatter) {
        chartInstances.absences.data.datasets[0].data = chartsData.absences_scatter;
        chartInstances.absences.update();
    }

    // 5. Study Habits Radar
    if (chartInstances.radar && chartsData.radar_profile) {
        chartInstances.radar.data.datasets[0].data = chartsData.radar_profile.high_achievers;
        chartInstances.radar.data.datasets[1].data = chartsData.radar_profile.at_risk_group;
        chartInstances.radar.update();
    }

    // 6. Failures Impact
    if (chartInstances.failures && chartsData.failures_impact) {
        chartInstances.failures.data.datasets[0].data = chartsData.failures_impact.means;
        chartInstances.failures.update();
    }
}

function initStudentTable() {
    renderTable();

    const filterSelect = document.getElementById('tableFilterSubject');
    filterSelect.onchange = (e) => {
        const val = e.target.value;
        if (val === 'high') {
            filteredRecords = studentRecords.filter(s => s.g3 >= 14);
        } else if (val === 'average') {
            filteredRecords = studentRecords.filter(s => s.g3 >= 10 && s.g3 < 14);
        } else if (val === 'at-risk') {
            filteredRecords = studentRecords.filter(s => s.g3 < 10);
        } else {
            filteredRecords = [...studentRecords];
        }
        currentPage = 1;
        renderTable();
    };

    document.getElementById('prevPageBtn').onclick = () => {
        if (currentPage > 1) {
            currentPage--;
            renderTable();
        }
    };

    document.getElementById('nextPageBtn').onclick = () => {
        const maxPage = Math.ceil(filteredRecords.length / pageSize);
        if (currentPage < maxPage) {
            currentPage++;
            renderTable();
        }
    };
}

function renderTable() {
    const tbody = document.getElementById('studentsTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';

    const startIdx = (currentPage - 1) * pageSize;
    const endIdx = startIdx + pageSize;
    const pageRecords = filteredRecords.slice(startIdx, endIdx);

    if (pageRecords.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 20px;">No matching student records found</td></tr>`;
        document.getElementById('tableRecordCount').textContent = 'Showing 0 records';
        return;
    }

    pageRecords.forEach(s => {
        const tr = document.createElement('tr');

        let statusPill = '';
        if (s.g3 >= 14) {
            statusPill = `<span class="status-pill pass"><i class="fa-solid fa-circle-check"></i> Distinction</span>`;
        } else if (s.g3 >= 10) {
            statusPill = `<span class="status-pill average"><i class="fa-solid fa-circle-info"></i> Passing</span>`;
        } else {
            statusPill = `<span class="status-pill risk"><i class="fa-solid fa-triangle-exclamation"></i> At Risk</span>`;
        }

        tr.innerHTML = `
            <td><strong>${s.id}</strong></td>
            <td>${s.age} yrs</td>
            <td>${s.studyTime}</td>
            <td>${s.absences} days</td>
            <td>${s.g1} / 20</td>
            <td>${s.g2} / 20</td>
            <td><strong>${s.g3} / 20</strong></td>
            <td>${statusPill}</td>
        `;
        tbody.appendChild(tr);
    });

    const totalPages = Math.ceil(filteredRecords.length / pageSize) || 1;
    document.getElementById('currentPageNum').textContent = `Page ${currentPage} of ${totalPages}`;
    document.getElementById('tableRecordCount').textContent = `Showing ${startIdx + 1} to ${Math.min(endIdx, filteredRecords.length)} of ${filteredRecords.length} records`;

    document.getElementById('prevPageBtn').disabled = currentPage === 1;
    document.getElementById('nextPageBtn').disabled = currentPage >= totalPages;
}

function initGlobalSearch() {
    const searchInput = document.getElementById('globalSearchInput');
    if (!searchInput) return;

    searchInput.addEventListener('input', (e) => {
        const term = e.target.value.toLowerCase().trim();
        if (term === '') {
            filteredRecords = [...studentRecords];
        } else {
            filteredRecords = studentRecords.filter(s =>
                s.id.toLowerCase().includes(term) ||
                s.studyTime.toLowerCase().includes(term) ||
                s.g3.toString().includes(term)
            );
        }
        currentPage = 1;
        renderTable();
    });
}

/* ==========================================================================
   3. API INTEGRATION: MODEL INSIGHTS & BENCHMARKS
   ========================================================================== */
async function fetchModelInsights() {
    try {
        const response = await fetch('/api/model-insights');
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        const result = await response.json();

        if (result.status === 'success' && result.data) {
            updateModelInsightsUI(result.data);
        }
    } catch (err) {
        console.error('Failed to fetch /api/model-insights:', err);
    }
}

function updateModelInsightsUI(metricsData) {
    if (!metricsData) return;

    // Update banner alert text to confirm live evaluation
    const banner = document.querySelector('.alert-banner.warning-banner div');
    if (banner) {
        banner.innerHTML = `<strong>Live Benchmark Metrics:</strong> Evaluated on <em>student.csv</em> (80/20 train/test split, random_state=42). <strong>Selected Best Model:</strong> ${metricsData.selected_model} (R² = ${metricsData.selected_metrics.r2}).`;
    }

    // Update the 4 Model Cards
    const models = metricsData.models || {};
    const modelCards = document.querySelectorAll('.model-card');

    modelCards.forEach(card => {
        const titleElem = card.querySelector('h3');
        if (!titleElem) return;
        const titleText = titleElem.textContent.trim();

        let modelKey = null;
        if (titleText.includes('Random Forest')) modelKey = 'Random Forest Regressor';
        else if (titleText.includes('Gradient Boosting')) modelKey = 'Gradient Boosting Regressor';
        else if (titleText.includes('Linear Regression')) modelKey = 'Linear Regression';
        else if (titleText.includes('Lasso')) modelKey = 'Lasso Regression';

        if (modelKey && models[modelKey]) {
            const m = models[modelKey];
            const rows = card.querySelectorAll('.metric-row');
            if (rows.length >= 3) {
                // Row 1: R² Score
                const r2ValElem = rows[0].querySelector('.value');
                if (r2ValElem) r2ValElem.textContent = m.r2.toFixed(4);

                // Progress Bar fill
                const progressFill = card.querySelector('.progress-fill');
                if (progressFill) progressFill.style.width = `${Math.max(0, Math.min(100, m.r2 * 100))}%`;

                // Row 2: MAE
                const maeValElem = rows[1].querySelector('.value');
                if (maeValElem) maeValElem.textContent = m.mae.toFixed(4);

                // Row 3: RMSE
                const rmseValElem = rows[2].querySelector('.value');
                if (rmseValElem) rmseValElem.textContent = m.rmse.toFixed(4);
            }
        }
    });

    // Update Feature Importance Chart with genuine values
    if (chartInstances.featureImp && metricsData.feature_importances) {
        const topFeatures = metricsData.feature_importances.slice(0, 7);
        const labels = topFeatures.map(f => {
            if (f.feature === 'G2') return 'Period 2 Score (G2)';
            if (f.feature === 'G1') return 'Period 1 Score (G1)';
            if (f.feature === 'absences') return 'Cumulative Absences';
            if (f.feature === 'studytime') return 'Weekly Study Time';
            if (f.feature === 'failures') return 'Past Failures History';
            if (f.feature === 'age') return 'Student Age';
            return f.feature;
        });
        const weights = topFeatures.map(f => f.importance_percentage);

        chartInstances.featureImp.data.labels = labels;
        chartInstances.featureImp.data.datasets[0].data = weights;
        chartInstances.featureImp.update();
    }
}

/* ==========================================================================
   4. CHART.JS VISUALIZATIONS INITIALIZATION
   ========================================================================== */
function initCharts() {
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.color = '#64748b';

    // 1. Grade Distribution Chart
    const ctxGradeDist = document.getElementById('gradeDistributionChart');
    if (ctxGradeDist) {
        chartInstances.gradeDist = new Chart(ctxGradeDist, {
            type: 'bar',
            data: {
                labels: ['Fail (<10)', 'Average (10-12)', 'Good (13-15)', 'Distinction (16-20)'],
                datasets: [{
                    label: 'Number of Students',
                    data: [130, 142, 85, 38],
                    backgroundColor: [
                        'rgba(239, 68, 68, 0.85)',
                        'rgba(245, 158, 11, 0.85)',
                        'rgba(59, 130, 246, 0.85)',
                        'rgba(16, 185, 129, 0.85)'
                    ],
                    borderColor: ['#ef4444', '#f59e0b', '#3b82f6', '#10b981'],
                    borderWidth: 1.5,
                    borderRadius: 8,
                    barThickness: 38
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, grid: { color: '#f1f5f9' }, title: { display: true, text: 'Student Count' } },
                    x: { grid: { display: false } }
                }
            }
        });
    }

    // 2. Study Time vs Grade Chart
    const ctxStudyVsGrade = document.getElementById('studyVsGradeChart');
    if (ctxStudyVsGrade) {
        chartInstances.studyVsGrade = new Chart(ctxStudyVsGrade, {
            type: 'bar',
            data: {
                labels: ['< 2 Hours', '2 - 5 Hours', '5 - 10 Hours', '> 10 Hours'],
                datasets: [
                    {
                        label: 'Average Final Score (G3)',
                        data: [9.8, 11.4, 13.6, 14.8],
                        backgroundColor: 'rgba(30, 58, 138, 0.85)',
                        borderColor: '#1e3a8a',
                        borderWidth: 1.5,
                        borderRadius: 6
                    },
                    {
                        label: 'Period 1 Avg (G1)',
                        data: [9.2, 10.9, 12.8, 14.1],
                        backgroundColor: 'rgba(59, 130, 246, 0.5)',
                        borderColor: '#3b82f6',
                        borderWidth: 1.5,
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'top' } },
                scales: {
                    y: { beginAtZero: true, max: 20, grid: { color: '#f1f5f9' }, title: { display: true, text: 'Score out of 20' } },
                    x: { grid: { display: false } }
                }
            }
        });
    }

    // 3. Grade Progression Line Chart
    const ctxProgression = document.getElementById('gradeProgressionChart');
    if (ctxProgression) {
        chartInstances.progression = new Chart(ctxProgression, {
            type: 'line',
            data: {
                labels: ['Period 1 (G1)', 'Period 2 (G2)', 'Final Grade (G3)'],
                datasets: [
                    {
                        label: 'Top Performers (G3 ≥ 15)',
                        data: [15.8, 16.4, 16.9],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 6
                    },
                    {
                        label: 'Average Performers (10 ≤ G3 < 15)',
                        data: [11.2, 11.5, 11.8],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 6
                    },
                    {
                        label: 'At-Risk Performers (G3 < 10)',
                        data: [8.4, 7.9, 7.1],
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'top' } },
                scales: {
                    y: { min: 0, max: 20, grid: { color: '#f1f5f9' }, title: { display: true, text: 'Average Score' } },
                    x: { grid: { display: false } }
                }
            }
        });
    }

    // 4. Absences Impact Scatter Analysis
    const ctxAbsences = document.getElementById('absencesAnalysisChart');
    if (ctxAbsences) {
        chartInstances.absences = new Chart(ctxAbsences, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Students (Absences vs G3)',
                    data: [{ x: 4, y: 13 }, { x: 12, y: 8 }, { x: 2, y: 17 }],
                    backgroundColor: 'rgba(245, 158, 11, 0.8)',
                    borderColor: '#f59e0b',
                    pointRadius: 6,
                    pointHoverRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `Absences: ${ctx.raw.x} days, Final Grade: ${ctx.raw.y}/20`
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: { display: true, text: 'Cumulative Absences (Days)' },
                        grid: { color: '#f1f5f9' }
                    },
                    y: {
                        min: 0,
                        max: 20,
                        title: { display: true, text: 'Final Score (G3)' },
                        grid: { color: '#f1f5f9' }
                    }
                }
            }
        });
    }

    // 5. Study Habits Radar Chart
    const ctxRadar = document.getElementById('studyHabitsRadarChart');
    if (ctxRadar) {
        chartInstances.radar = new Chart(ctxRadar, {
            type: 'radar',
            data: {
                labels: ['Weekly Study Time', 'G1 Score', 'G2 Score', 'Attendance Rate', 'Extra Tutoring', 'Higher Edu Desire'],
                datasets: [
                    {
                        label: 'High Achievers (G3 ≥ 15)',
                        data: [85, 82, 88, 92, 60, 95],
                        backgroundColor: 'rgba(16, 185, 129, 0.25)',
                        borderColor: '#10b981',
                        pointBackgroundColor: '#10b981'
                    },
                    {
                        label: 'At-Risk Group (G3 < 10)',
                        data: [42, 36, 32, 58, 28, 62],
                        backgroundColor: 'rgba(239, 68, 68, 0.25)',
                        borderColor: '#ef4444',
                        pointBackgroundColor: '#ef4444'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: '#e2e8f0' },
                        grid: { color: '#f1f5f9' },
                        suggestedMin: 0,
                        suggestedMax: 100
                    }
                }
            }
        });
    }

    // 6. Failures Impact Bar Chart
    const ctxFailures = document.getElementById('failuresImpactChart');
    if (ctxFailures) {
        chartInstances.failures = new Chart(ctxFailures, {
            type: 'bar',
            data: {
                labels: ['0 Failures', '1 Failure', '2 Failures', '3+ Failures'],
                datasets: [{
                    label: 'Avg Final Score',
                    data: [12.6, 9.2, 7.8, 5.6],
                    backgroundColor: ['#1e3a8a', '#3b82f6', '#f59e0b', '#ef4444'],
                    borderRadius: 6,
                    barThickness: 34
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { min: 0, max: 20, grid: { color: '#f1f5f9' }, title: { display: true, text: 'Average Score G3' } },
                    x: { grid: { display: false } }
                }
            }
        });
    }

    // 7. Feature Importance Chart
    const ctxFeatureImp = document.getElementById('featureImportanceChart');
    if (ctxFeatureImp) {
        chartInstances.featureImp = new Chart(ctxFeatureImp, {
            type: 'bar',
            data: {
                labels: ['G2', 'absences', 'age', 'G1', 'studytime', 'failures'],
                datasets: [{
                    label: 'Relative Feature Importance Weight (%)',
                    data: [79.9, 11.6, 3.6, 2.3, 1.2, 0.7],
                    backgroundColor: 'rgba(30, 58, 138, 0.85)',
                    borderColor: '#1e3a8a',
                    borderWidth: 1,
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `Importance Weight: ${ctx.raw}%`
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        title: { display: true, text: 'Importance Weight (%)' },
                        grid: { color: '#f1f5f9' }
                    },
                    y: { grid: { display: false } }
                }
            }
        });
    }
}

/* ==========================================================================
   5. PREDICTION ENGINE: CONNECTED TO FLASK POST /api/predict
   ========================================================================== */
function initPredictionEngine() {
    const form = document.getElementById('predictionForm');
    const resetBtn = document.getElementById('resetFormBtn');

    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (validateForm()) {
            await executeMLPrediction();
        }
    });

    resetBtn.addEventListener('click', () => {
        form.reset();
        clearErrors();
        // Reset prediction UI to initial state
        resetPredictionUI();
    });

    const inputs = form.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('input', () => {
            clearFieldError(input);
        });
    });
}

function validateForm() {
    let isValid = true;
    clearErrors();

    const ageInput = document.getElementById('studentAge');
    const ageVal = parseInt(ageInput.value, 10);
    if (isNaN(ageVal) || ageVal < 15 || ageVal > 22) {
        showFieldError(ageInput, 'ageError', 'Age must be between 15 and 22 years.');
        isValid = false;
    }

    const absInput = document.getElementById('absences');
    const absVal = parseInt(absInput.value, 10);
    if (isNaN(absVal) || absVal < 0 || absVal > 30) {
        showFieldError(absInput, 'absencesError', 'Absences must be between 0 and 30 days.');
        isValid = false;
    }

    const g1Input = document.getElementById('g1Grade');
    const g1Val = parseFloat(g1Input.value);
    if (isNaN(g1Val) || g1Val < 0 || g1Val > 20) {
        showFieldError(g1Input, 'g1Error', 'Period 1 score must be between 0 and 20.');
        isValid = false;
    }

    const g2Input = document.getElementById('g2Grade');
    const g2Val = parseFloat(g2Input.value);
    if (isNaN(g2Val) || g2Val < 0 || g2Val > 20) {
        showFieldError(g2Input, 'g2Error', 'Period 2 score must be between 0 and 20.');
        isValid = false;
    }

    return isValid;
}

function showFieldError(inputElem, errorSpanId, message) {
    inputElem.classList.add('invalid');
    const span = document.getElementById(errorSpanId);
    if (span) span.textContent = message;
}

function clearFieldError(inputElem) {
    inputElem.classList.remove('invalid');
    const errorSpan = inputElem.parentElement.parentElement.querySelector('.error-message');
    if (errorSpan) errorSpan.textContent = '';
}

function clearErrors() {
    document.querySelectorAll('.error-message').forEach(span => span.textContent = '');
    document.querySelectorAll('input, select').forEach(input => input.classList.remove('invalid'));
}

/**
 * Execute real machine learning inference by calling Flask POST /api/predict
 */
async function executeMLPrediction() {
    const predictBtn = document.getElementById('predictBtn');
    const originalBtnHtml = predictBtn.innerHTML;

    // 1. Build payload matching pipeline feature names and types
    const payload = {
        age: parseInt(document.getElementById('studentAge').value, 10),
        studytime: parseInt(document.getElementById('studyTime').value, 10),
        absences: parseInt(document.getElementById('absences').value, 10),
        failures: parseInt(document.getElementById('failures').value, 10),
        G1: parseFloat(document.getElementById('g1Grade').value),
        G2: parseFloat(document.getElementById('g2Grade').value),
        schoolsup: document.getElementById('schoolsup').value,
        higher: document.getElementById('higher').value
    };

    // 2. Set loading state on button
    predictBtn.disabled = true;
    predictBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> <span>Evaluating Model...</span>`;

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (response.ok && data.status === 'success') {
            displayPredictionResult(data.prediction);
        } else {
            // Handle backend validation or processing error
            if (data.errors) {
                for (const [field, msg] of Object.entries(data.errors)) {
                    const elem = document.querySelector(`[name="${field}"]`) || document.getElementById(field);
                    if (elem) elem.classList.add('invalid');
                }
            }
            alert(`Prediction error: ${data.message || 'Server rejected request'}`);
        }
    } catch (err) {
        console.error('Inference API call failed:', err);
        alert('Could not connect to Flask prediction endpoint. Ensure app.py is running on port 5000.');
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = originalBtnHtml;
    }
}

function displayPredictionResult(prediction) {
    if (!prediction) return;

    // 1. Update score value and percentage
    document.getElementById('predictedGradeVal').textContent = prediction.score.toFixed(1);
    document.getElementById('predictedPercentVal').textContent = `${prediction.percentage.toFixed(1)}% Equivalent Score`;

    // 2. Update Conic Gradient Ring
    const scoreRing = document.getElementById('scoreRing');
    const colorHex = prediction.tier === 'good' ? '#10b981' : (prediction.tier === 'average' ? '#3b82f6' : '#ef4444');
    scoreRing.style.background = `conic-gradient(${colorHex} 0% ${prediction.percentage}%, var(--border-color) ${prediction.percentage}% 100%)`;

    // 3. Update Tier Pill
    const tierPill = document.getElementById('tierPill');
    tierPill.textContent = prediction.tier_label;
    tierPill.className = `tier-pill tier-${prediction.tier}`;

    // 4. Update Status Badge
    const statusBadge = document.getElementById('predictionStatusBadge');
    if (statusBadge) {
        statusBadge.textContent = `Live Pipeline (${prediction.model_used})`;
        statusBadge.className = 'badge badge-navy';
    }

    // 5. Update Dynamic Drivers List
    const driversList = document.getElementById('driversList');
    driversList.innerHTML = '';

    if (prediction.factors && Array.isArray(prediction.factors)) {
        prediction.factors.forEach(f => {
            const isPos = f.type === 'positive';
            const li = document.createElement('li');
            li.className = `driver-item ${isPos ? 'positive' : 'negative'}`;
            li.innerHTML = `
                <i class="fa-solid fa-${isPos ? 'circle-plus' : 'circle-minus'}"></i>
                <span>${f.text}</span>
            `;
            driversList.appendChild(li);
        });
    }

    // 6. Update Recommendation
    const recText = document.getElementById('recommendationText');
    if (recText) {
        recText.textContent = prediction.recommendation;
    }

    // 7. Update Callout Disclaimer
    const callout = document.querySelector('.callout-disclaimer div');
    if (callout) {
        callout.innerHTML = `<strong>Live Machine Learning Output:</strong> Predicted using the fitted <strong>${prediction.model_used}</strong> pipeline (student_model.pkl). Connected to Flask REST backend.`;
    }
}

function resetPredictionUI() {
    document.getElementById('predictedGradeVal').textContent = '--';
    document.getElementById('predictedPercentVal').textContent = 'Awaiting Input';
    const scoreRing = document.getElementById('scoreRing');
    scoreRing.style.background = `conic-gradient(var(--border-color) 0% 100%)`;
    const tierPill = document.getElementById('tierPill');
    tierPill.textContent = 'Awaiting Prediction';
    tierPill.className = 'tier-pill tier-average';
}
