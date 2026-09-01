/**
 * Horn Generator Coordinator App Logic
 * Wires UI inputs to calculation engine, 2D/3D viewports, and exporter tools.
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const hornTypeSelect = document.getElementById('horn-type');
    const throatRInput = document.getElementById('throat-r');
    const cutoffFInput = document.getElementById('cutoff-f');
    const cutoffGroup = document.getElementById('cutoff-group');
    const osseFcBadge = document.getElementById('osse-fc-badge');
    const osseFcVal = document.getElementById('osse-fc-val');

    // OS-SE Inputs
    const osseParams = document.getElementById('osse-params');
    const lengthInput = document.getElementById('length');
    const alphaInput = document.getElementById('alpha');
    const alpha0Input = document.getElementById('alpha-0');
    const kFactorInput = document.getElementById('k-factor');
    const sRatioInput = document.getElementById('s-ratio');
    const qCoeffInput = document.getElementById('q-coeff');
    const nExponentInput = document.getElementById('n-exponent');
    const numPointsOsseInput = document.getElementById('num-points-osse');

    // Tractrix Inputs
    const tractrixParams = document.getElementById('tractrix-params');
    const numPointsTractrixInput = document.getElementById('num-points-tractrix');

    // Spherical Inputs
    const sphericalParams = document.getElementById('spherical-params');
    const scaleSphericalInput = document.getElementById('scale-spherical');
    const foldSphericalCheck = document.getElementById('fold-spherical');
    const foldBackSphericalCheck = document.getElementById('fold-back-spherical');

    // Exponential Inputs
    const exponentialParams = document.getElementById('exponential-params');
    const scaleExpInput = document.getElementById('scale-exp');

    // HCD Inputs
    const enableHCDCheck = document.getElementById('enable-hcd');
    const hcdParams = document.getElementById('hcd-params');
    const mouthRatioInput = document.getElementById('mouth-ratio');
    const hcdModeSelect = document.getElementById('hcd-mode');
    const hcdAccInput = document.getElementById('hcd-acc');

    // Modeling Inputs
    const wallThicknessInput = document.getElementById('wall-thickness');

    // Data Table
    const tableHead = document.getElementById('table-head');
    const tableBody = document.getElementById('table-body');

    // Export Buttons
    const btnExportDXF = document.getElementById('btn-export-dxf');
    const btnExportSTL = document.getElementById('btn-export-stl');
    const btnExportOBJ = document.getElementById('btn-export-obj');
    const btnExportSCAD = document.getElementById('btn-export-scad');
    const btnExportCSV = document.getElementById('btn-export-csv');

    // 3D Viewport Controls Buttons
    const btnToggleWireframe = document.getElementById('btn-toggle-wireframe');
    const btnToggleCutaway = document.getElementById('btn-toggle-cutaway');
    const btnToggleRotate = document.getElementById('btn-toggle-rotate');
    const btnResetCam = document.getElementById('btn-reset-cam');

    // --- Initialize Viewers ---
    const viewer3D = new Horn3DViewer('viewport-3d-container');
    const viewer2D = new Horn2DViewer('chart-2d', 'chart-hcd-transition');

    let currentPoints = [];
    let currentIsHCD = false;

    // --- Dynamic Form Visibility Update ---
    function updateFormVisibility() {
        const type = hornTypeSelect.value;

        // Hide all parameter groups initially
        osseParams.style.display = 'none';
        tractrixParams.style.display = 'none';
        sphericalParams.style.display = 'none';
        exponentialParams.style.display = 'none';

        if (type === 'OS-SE') {
            cutoffGroup.style.display = 'none';
            osseFcBadge.style.display = 'flex';
            osseParams.style.display = 'block';
        } else {
            cutoffGroup.style.display = 'block';
            osseFcBadge.style.display = 'none';

            if (type === 'Tractrix') tractrixParams.style.display = 'block';
            else if (type === 'Spherical') sphericalParams.style.display = 'block';
            else if (type === 'Exponential') exponentialParams.style.display = 'block';
        }

        // HCD Params Visibility
        hcdParams.style.display = enableHCDCheck.checked ? 'block' : 'none';

        // DXF Export availability (Only available in circular mode for full DXF spline)
        if (enableHCDCheck.checked) {
            btnExportDXF.disabled = true;
            btnExportDXF.title = 'DXF export is available for circular 2D profiles.';
        } else {
            btnExportDXF.disabled = false;
            btnExportDXF.title = '';
        }
    }

    // --- Main Update Function ---
    function updateHorn() {
        updateFormVisibility();

        const type = hornTypeSelect.value;
        const throatR = parseFloat(throatRInput.value) || 15.0;
        const cutoffF = parseFloat(cutoffFInput.value) || 1000.0;
        const wallThickness = parseFloat(wallThicknessInput.value) || 2.0;

        let result = { points: [] };

        try {
            if (type === 'OS-SE') {
                const length = parseFloat(lengthInput.value) || 10;
                const alpha = parseFloat(alphaInput.value) || 45;
                const alpha0 = parseFloat(alpha0Input.value) || 0;
                const k = parseFloat(kFactorInput.value) || 1.0;
                const s = parseFloat(sRatioInput.value) || 0.8;
                const q = parseFloat(qCoeffInput.value) || 0.998;
                const n = parseFloat(nExponentInput.value) || 5;
                const numPoints = parseInt(numPointsOsseInput.value) || 20;

                result = HornMath.generateOSSEHorn(throatR, length, alpha, alpha0, k, s, q, n, numPoints);

                if (result.calculatedFc) {
                    osseFcVal.textContent = `${Math.round(result.calculatedFc).toLocaleString()} Hz`;
                }
            } else if (type === 'Tractrix') {
                const numPoints = parseInt(numPointsTractrixInput.value) || 20;
                result = HornMath.generateTractrixHorn(throatR, cutoffF, numPoints);
            } else if (type === 'Spherical') {
                const scale = parseFloat(scaleSphericalInput.value) || 4.0;
                const fold = foldSphericalCheck.checked;
                const foldBack = foldBackSphericalCheck.checked;
                result = HornMath.generateSphericalHorn(throatR, cutoffF, scale, fold, foldBack);
            } else if (type === 'Exponential') {
                const scale = parseFloat(scaleExpInput.value) || 4.0;
                result = HornMath.generateExponentialHorn(throatR, cutoffF, scale);
            }
        } catch (err) {
            console.warn("Horn Math Warning:", err.message);
            alert(err.message);
            return;
        }

        let finalPoints = result.points;
        const isHCD = enableHCDCheck.checked;
        currentIsHCD = isHCD;

        if (isHCD && finalPoints.length > 0) {
            const mouthRatio = parseFloat(mouthRatioInput.value) || 1.7;
            const mode = hcdModeSelect.value;
            const acc = parseFloat(hcdAccInput.value) || 1.0;

            const hcdResult = HornMath.generateHCDHorn(finalPoints, mouthRatio, mode, acc);
            finalPoints = hcdResult.points;
        }

        currentPoints = finalPoints;

        // Update 2D Chart
        viewer2D.updateChart(finalPoints, isHCD);

        // Update 3D Mesh Viewport
        viewer3D.updateMesh(finalPoints, isHCD, wallThickness);

        // Populate Table
        renderTable(finalPoints, isHCD);
    }

    // --- Render Data Table ---
    function renderTable(points, isHCD) {
        if (isHCD) {
            tableHead.innerHTML = `
                <th>#</th>
                <th>x (mm)</th>
                <th>y (mm)</th>
                <th>Semi-Major a (mm)</th>
                <th>Semi-Minor b (mm)</th>
                <th>Mouth Ratio</th>
            `;
        } else {
            tableHead.innerHTML = `
                <th>#</th>
                <th>x (mm)</th>
                <th>y (mm)</th>
            `;
        }

        let bodyHtml = '';
        points.forEach((p, idx) => {
            if (isHCD) {
                bodyHtml += `
                    <tr>
                        <td>${idx + 1}</td>
                        <td>${p.x.toFixed(3)}</td>
                        <td>${p.y.toFixed(3)}</td>
                        <td>${p.a.toFixed(3)}</td>
                        <td>${p.b.toFixed(3)}</td>
                        <td>${p.mouthRatio.toFixed(3)}</td>
                    </tr>
                `;
            } else {
                bodyHtml += `
                    <tr>
                        <td>${idx + 1}</td>
                        <td>${p.x.toFixed(3)}</td>
                        <td>${p.y.toFixed(3)}</td>
                    </tr>
                `;
            }
        });
        tableBody.innerHTML = bodyHtml;
    }

    // --- Event Listeners for Auto-Update ---
    const allInputs = document.querySelectorAll('input, select');
    allInputs.forEach(input => {
        input.addEventListener('change', () => updateHorn());
        if (input.type === 'number' || input.type === 'range') {
            input.addEventListener('input', () => updateHorn());
        }
    });

    // --- 3D Viewport Controls Event Listeners ---
    let isWireframe = false;
    btnToggleWireframe.addEventListener('click', () => {
        isWireframe = !isWireframe;
        viewer3D.setWireframe(isWireframe);
        btnToggleWireframe.classList.toggle('btn-primary', isWireframe);
    });

    let isCutaway = false;
    btnToggleCutaway.addEventListener('click', () => {
        isCutaway = !isCutaway;
        viewer3D.setCutaway(isCutaway);
        btnToggleCutaway.classList.toggle('btn-primary', isCutaway);
    });

    let isRotating = false;
    btnToggleRotate.addEventListener('click', () => {
        isRotating = !isRotating;
        viewer3D.setAutoRotate(isRotating);
        btnToggleRotate.classList.toggle('btn-primary', isRotating);
    });

    btnResetCam.addEventListener('click', () => {
        viewer3D.resetCamera();
    });

    // --- Export File Event Handlers ---
    btnExportDXF.addEventListener('click', () => {
        if (currentPoints.length === 0) return;
        const dxf = HornExporters.generateDXF(currentPoints);
        const fileName = `${hornTypeSelect.value}_Profile.dxf`;
        HornExporters.downloadFile(dxf, fileName, 'image/vnd.dxf');
    });

    btnExportSTL.addEventListener('click', () => {
        if (currentPoints.length === 0) return;
        const wallThickness = parseFloat(wallThicknessInput.value) || 2.0;
        const stlBuffer = HornExporters.generateSTL(currentPoints, currentIsHCD, wallThickness);
        const fileName = `${hornTypeSelect.value}${currentIsHCD ? '_HCD' : ''}_Model.stl`;
        HornExporters.downloadFile(stlBuffer, fileName, 'model/stl');
    });

    btnExportOBJ.addEventListener('click', () => {
        if (currentPoints.length === 0) return;
        const wallThickness = parseFloat(wallThicknessInput.value) || 2.0;
        const objText = HornExporters.generateOBJ(currentPoints, currentIsHCD, wallThickness);
        const fileName = `${hornTypeSelect.value}${currentIsHCD ? '_HCD' : ''}_Model.obj`;
        HornExporters.downloadFile(objText, fileName, 'text/plain');
    });

    btnExportSCAD.addEventListener('click', () => {
        if (currentPoints.length === 0) return;
        const wallThickness = parseFloat(wallThicknessInput.value) || 2.0;
        const scadText = HornExporters.generateOpenSCAD(currentPoints, currentIsHCD, wallThickness);
        const fileName = `${hornTypeSelect.value}${currentIsHCD ? '_HCD' : ''}_Script.scad`;
        HornExporters.downloadFile(scadText, fileName, 'text/x-scad');
    });

    btnExportCSV.addEventListener('click', () => {
        if (currentPoints.length === 0) return;
        const csvText = HornExporters.generateCSV(currentPoints, currentIsHCD);
        const fileName = `${hornTypeSelect.value}${currentIsHCD ? '_HCD' : ''}_Data.csv`;
        HornExporters.downloadFile(csvText, fileName, 'text/csv');
    });

    // Initial Trigger
    updateHorn();
});
