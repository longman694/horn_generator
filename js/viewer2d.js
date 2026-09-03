/**
 * Horn 2D Profile Viewer Manager using Chart.js
 * Renders profile contour graphs and HCD transition curves.
 */

class Horn2DViewer {
    constructor(canvasId, transitionCanvasId) {
        this.canvas = document.getElementById(canvasId);
        this.transitionCanvas = document.getElementById(transitionCanvasId);
        this.chart = null;
        this.transitionChart = null;
    }

    updateChart(pointsData, isHCD = false, isMorph = false) {
        if (!this.canvas) return;

        const ctx = this.canvas.getContext('2d');
        const datasets = [];
        const points = Array.isArray(pointsData) ? pointsData : (pointsData.pointsMorphed || pointsData.points || []);

        // Check if pointsData is Morphing result object
        if (isMorph || (pointsData && pointsData.pointsMajor && pointsData.pointsMinor)) {
            const raw = pointsData.rawPoints || [];
            const major = pointsData.pointsMajor || [];
            const minor = pointsData.pointsMinor || [];
            const corner = pointsData.pointsCorner || [];

            datasets.push({
                label: 'Base Horn Reference y (mm)',
                data: raw.map(p => ({ x: p.x, y: p.y })),
                borderColor: '#94a3b8',
                borderDash: [4, 4],
                borderWidth: 2,
                fill: false,
                pointRadius: 2
            });

            datasets.push({
                label: 'Major Axis (ϕ=0°)',
                data: major.map(p => ({ x: p.x, y: p.y })),
                borderColor: '#f43f5e',
                backgroundColor: 'rgba(244, 63, 94, 0.1)',
                borderWidth: 2.5,
                fill: false,
                pointRadius: 3
            });

            datasets.push({
                label: 'Minor Axis (ϕ=90°)',
                data: minor.map(p => ({ x: p.x, y: p.y })),
                borderColor: '#38bdf8',
                backgroundColor: 'rgba(56, 189, 248, 0.1)',
                borderWidth: 2.5,
                fill: false,
                pointRadius: 3
            });

            if (corner && corner.length > 0) {
                datasets.push({
                    label: 'Corner Axis (ϕ=45°)',
                    data: corner.map(p => ({ x: p.x, y: p.y })),
                    borderColor: '#f59e0b',
                    borderDash: [2, 2],
                    borderWidth: 2,
                    fill: false,
                    pointRadius: 2
                });
            }
        } else if (isHCD) {
            datasets.push({
                label: 'Circular Reference Radius y (mm)',
                data: points.map(p => ({ x: p.x, y: p.y })),
                borderColor: '#94a3b8',
                borderDash: [5, 5],
                borderWidth: 2,
                fill: false,
                pointRadius: 2
            });

            datasets.push({
                label: 'HCD Semi-Minor Radius b (mm)',
                data: points.map(p => ({ x: p.x, y: p.b })),
                borderColor: '#38bdf8',
                backgroundColor: 'rgba(56, 189, 248, 0.15)',
                borderWidth: 2.5,
                fill: false,
                pointRadius: 3
            });

            datasets.push({
                label: 'HCD Semi-Major Radius a (mm)',
                data: points.map(p => ({ x: p.x, y: p.a })),
                borderColor: '#f43f5e',
                backgroundColor: 'rgba(244, 63, 94, 0.15)',
                borderWidth: 2.5,
                fill: false,
                pointRadius: 3
            });
        } else {
            datasets.push({
                label: 'Horn Profile Radius y (mm)',
                data: points.map(p => ({ x: p.x, y: p.y })),
                borderColor: '#38bdf8',
                backgroundColor: 'rgba(56, 189, 248, 0.1)',
                borderWidth: 2.5,
                fill: true,
                pointRadius: 3,
                pointHoverRadius: 6,
                tension: 0.2
            });
        }

        if (this.chart) {
            this.chart.destroy();
        }

        this.chart = new Chart(ctx, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    title: {
                        display: true,
                        text: isHCD ? '2D HCD Horn Profile Comparison' : (isMorph ? '2D Morphed Horn Profile Comparison' : '2D Horn Profile Contour'),
                        color: '#f8fafc',
                        font: { size: 16, weight: 'bold', family: 'Inter, system-ui, sans-serif' }
                    },
                    legend: {
                        labels: { color: '#94a3b8', font: { size: 12 } }
                    },
                    tooltip: {
                        backgroundColor: '#1e293b',
                        titleColor: '#f8fafc',
                        bodyColor: '#38bdf8',
                        borderColor: '#334155',
                        borderWidth: 1,
                        callbacks: {
                            label: function (context) {
                                return `${context.dataset.label}: ${context.parsed.y.toFixed(2)} mm (at x = ${context.parsed.x.toFixed(2)} mm)`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: 'Axial Length x (mm)', color: '#94a3b8' },
                        grid: { color: '#334155' },
                        ticks: { color: '#94a3b8' },
                        min: 0
                    },
                    y: {
                        title: { display: true, text: 'Radius / Radius Components (mm)', color: '#94a3b8' },
                        grid: { color: '#334155' },
                        ticks: { color: '#94a3b8' },
                        min: 0
                    }
                }
            }
        });

        // Update Transition Chart if HCD enabled
        const transitionCard = this.transitionCanvas ? (this.transitionCanvas.closest ? this.transitionCanvas.closest('.viewport-card') : this.transitionCanvas.parentElement.parentElement) : null;

        if (isHCD && this.transitionCanvas) {
            if (transitionCard) transitionCard.style.display = 'flex';
            this.transitionCanvas.parentElement.style.display = 'block';
            const tCtx = this.transitionCanvas.getContext('2d');

            if (this.transitionChart) {
                this.transitionChart.destroy();
            }

            this.transitionChart = new Chart(tCtx, {
                type: 'line',
                data: {
                    datasets: [{
                        label: 'Mouth Aspect Ratio (a / b)',
                        data: points.map((p, idx) => ({ x: p.x, y: p.mouthRatio })),
                        borderColor: '#eab308',
                        backgroundColor: 'rgba(234, 179, 8, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        pointRadius: 3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'HCD Mouth Aspect Ratio Transition Curve',
                            color: '#f8fafc',
                            font: { size: 14, weight: 'bold' }
                        },
                        legend: { labels: { color: '#94a3b8' } }
                    },
                    scales: {
                        x: {
                            type: 'linear',
                            title: { display: true, text: 'Axial Length x (mm)', color: '#94a3b8' },
                            grid: { color: '#334155' },
                            ticks: { color: '#94a3b8' }
                        },
                        y: {
                            title: { display: true, text: 'Aspect Ratio', color: '#94a3b8' },
                            grid: { color: '#334155' },
                            ticks: { color: '#94a3b8' },
                            min: 1
                        }
                    }
                }
            });
        } else if (this.transitionCanvas) {
            if (transitionCard) transitionCard.style.display = 'none';
            this.transitionCanvas.parentElement.style.display = 'none';
        }
    }
}

window.Horn2DViewer = Horn2DViewer;
