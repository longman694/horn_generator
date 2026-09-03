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

    // Surface Morphing Inputs
    const osseMorphSection = document.getElementById('osse-morph-section');
    const morphTargetShapeSelect = document.getElementById('morph-target-shape');
    const morphOptionsDiv = document.getElementById('morph-options');
    const morphTargetWidthInput = document.getElementById('morph-target-width');
    const labelMorphWidth = document.getElementById('label-morph-width');
    const morphTargetHeightInput = document.getElementById('morph-target-height');
    const groupMorphHeight = document.getElementById('group-morph-height');
    const morphCornerRadiusInput = document.getElementById('morph-corner-radius');
    const groupMorphCorner = document.getElementById('group-morph-corner');
    const morphFixedPartInput = document.getElementById('morph-fixed-part');
    const morphRateInput = document.getElementById('morph-rate');
    const morphAllowShrinkageCheck = document.getElementById('morph-allow-shrinkage');

    // HCD Inputs
    const hcdSection = document.getElementById('hcd-section');
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
    let currentResult = null;
    let currentIsHCD = false;
    let currentIsMorph = false;

    // --- Dynamic Form Visibility Update ---
    function updateFormVisibility() {
        const type = hornTypeSelect.value;

        // Hide all type-specific parameter groups initially
        osseParams.style.display = 'none';
        tractrixParams.style.display = 'none';
        sphericalParams.style.display = 'none';
        exponentialParams.style.display = 'none';

        if (type === 'OS-SE') {
            cutoffGroup.style.display = 'none';
            osseFcBadge.style.display = 'flex';
            osseParams.style.display = 'block';

            // OS-SE: Morphing is available, HCD is hidden
            if (osseMorphSection) osseMorphSection.style.display = 'block';
            if (hcdSection) hcdSection.style.display = 'none';
            if (enableHCDCheck) enableHCDCheck.checked = false;
            if (hcdParams) hcdParams.style.display = 'none';

            const morphShape = morphTargetShapeSelect ? morphTargetShapeSelect.value : 'none';
            if (morphShape !== 'none') {
                if (morphOptionsDiv) {
                    morphOptionsDiv.style.display = 'block';

                    if (morphShape === 'circle') {
                        if (groupMorphHeight) groupMorphHeight.style.display = 'none';
                        if (groupMorphCorner) groupMorphCorner.style.display = 'none';
                        if (labelMorphWidth) labelMorphWidth.textContent = 'Target Diameter (mm)';
                    } else if (morphShape === 'ellipse') {
                        if (groupMorphHeight) groupMorphHeight.style.display = 'block';
                        if (groupMorphCorner) groupMorphCorner.style.display = 'none';
                        if (labelMorphWidth) labelMorphWidth.textContent = 'Target Width (mm)';
                    } else {
                        // rectangle
                        if (groupMorphHeight) groupMorphHeight.style.display = 'block';
                        if (groupMorphCorner) groupMorphCorner.style.display = 'block';
                        if (labelMorphWidth) labelMorphWidth.textContent = 'Target Width (mm)';
                    }
                }
            } else {
                if (morphOptionsDiv) morphOptionsDiv.style.display = 'none';
            }

            // DXF Export availability
            if (morphShape !== 'none') {
                btnExportDXF.disabled = true;
                btnExportDXF.title = 'DXF export is for circular 2D profiles. Use STL/OBJ for 3D morphed models.';
            } else {
                btnExportDXF.disabled = false;
                btnExportDXF.title = '';
            }
        } else {
            cutoffGroup.style.display = 'block';
            osseFcBadge.style.display = 'none';

            if (type === 'Tractrix') tractrixParams.style.display = 'block';
            else if (type === 'Spherical') sphericalParams.style.display = 'block';
            else if (type === 'Exponential') exponentialParams.style.display = 'block';

            // Other types: HCD is available, Morphing is hidden
            if (osseMorphSection) osseMorphSection.style.display = 'none';
            if (morphOptionsDiv) morphOptionsDiv.style.display = 'none';
            if (morphTargetShapeSelect) morphTargetShapeSelect.value = 'none';

            if (hcdSection) hcdSection.style.display = 'block';
            const isHCDActive = Boolean(enableHCDCheck && enableHCDCheck.checked);
            if (hcdParams) hcdParams.style.display = isHCDActive ? 'block' : 'none';

            // DXF Export availability
            if (isHCDActive) {
                btnExportDXF.disabled = true;
                btnExportDXF.title = 'DXF export is for circular 2D profiles. Use STL/OBJ for 3D HCD loft.';
            } else {
                btnExportDXF.disabled = false;
                btnExportDXF.title = '';
            }
        }
    }

    // --- Main Update Function ---
    function updateHorn() {
        updateFormVisibility();

        const type = hornTypeSelect.value;
        const throatR = parseFloat(throatRInput.value) || 15.0;
        const cutoffF = parseFloat(cutoffFInput.value) || 1000.0;
        const wallThickness = parseFloat(wallThicknessInput.value) || 2.0;

        let baseResult = { points: [] };

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

                baseResult = HornMath.generateOSSEHorn(throatR, length, alpha, alpha0, k, s, q, n, numPoints);

                if (baseResult.calculatedFc) {
                    osseFcVal.textContent = `${Math.round(baseResult.calculatedFc).toLocaleString()} Hz`;
                }
            } else if (type === 'Tractrix') {
                const numPoints = parseInt(numPointsTractrixInput.value) || 20;
                baseResult = HornMath.generateTractrixHorn(throatR, cutoffF, numPoints);
            } else if (type === 'Spherical') {
                const scale = parseFloat(scaleSphericalInput.value) || 4.0;
                const fold = foldSphericalCheck.checked;
                const foldBack = foldBackSphericalCheck.checked;
                baseResult = HornMath.generateSphericalHorn(throatR, cutoffF, scale, fold, foldBack);
            } else if (type === 'Exponential') {
                const scale = parseFloat(scaleExpInput.value) || 4.0;
                baseResult = HornMath.generateExponentialHorn(throatR, cutoffF, scale);
            }
        } catch (err) {
            console.warn("Horn Math Warning:", err.message);
            alert(err.message);
            return;
        }

        const basePoints = baseResult.points || [];
        const isHCD = (type !== 'OS-SE') && Boolean(enableHCDCheck && enableHCDCheck.checked);
        const targetShape = (type === 'OS-SE' && morphTargetShapeSelect) ? morphTargetShapeSelect.value : 'none';
        const isMorph = (type === 'OS-SE') && (targetShape !== 'none');

        let finalPoints = basePoints;
        let chartData = basePoints;

        if (isHCD && basePoints.length > 0) {
            const mouthRatio = parseFloat(mouthRatioInput.value) || 1.7;
            const mode = hcdModeSelect.value;
            const acc = parseFloat(hcdAccInput.value) || 1.0;

            const hcdResult = HornMath.generateHCDHorn(basePoints, mouthRatio, mode, acc);
            finalPoints = hcdResult.points;
            chartData = finalPoints;
        } else if (isMorph && basePoints.length > 0) {
            let targetWidth = parseFloat(morphTargetWidthInput.value) || 300;
            let targetHeight = parseFloat(morphTargetHeightInput.value) || 200;
            if (targetShape === 'circle') {
                targetHeight = targetWidth;
            }
            const cornerRadius = parseFloat(morphCornerRadiusInput.value) || 20;
            const fixedPart = parseFloat(morphFixedPartInput.value) || 0.0;
            const morphRate = parseFloat(morphRateInput.value) || 3.0;
            const allowShrinkage = morphAllowShrinkageCheck ? morphAllowShrinkageCheck.checked : false;

            const morphResult = HornMath.applySurfaceMorphing(
                basePoints, targetShape, targetWidth, targetHeight, cornerRadius,
                fixedPart, morphRate, allowShrinkage, 96
            );
            if (baseResult.calculatedFc) {
                morphResult.calculatedFc = baseResult.calculatedFc;
            }
            finalPoints = morphResult.pointsMorphed;
            chartData = morphResult;
        }

        currentPoints = finalPoints;
        currentResult = chartData;
        currentIsHCD = isHCD;
        currentIsMorph = isMorph;

        // Update 2D Chart
        viewer2D.updateChart(chartData, isHCD, isMorph);

        // Update 3D Mesh Viewport
        viewer3D.updateMesh(finalPoints, isHCD, wallThickness);

        // Populate Table
        renderTable(finalPoints, isHCD, isMorph);
    }

    // --- Render Data Table ---
    function renderTable(points, isHCD, isMorph) {
        if (isMorph) {
            tableHead.innerHTML = `
                <th>#</th>
                <th>x (mm)</th>
                <th>Major a (mm)</th>
                <th>Minor b (mm)</th>
                <th>Corner r (mm)</th>
            `;
        } else if (isHCD) {
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
            if (isMorph) {
                bodyHtml += `
                    <tr>
                        <td>${idx + 1}</td>
                        <td>${p.x.toFixed(3)}</td>
                        <td>${(p.a || p.y).toFixed(3)}</td>
                        <td>${(p.b || p.y).toFixed(3)}</td>
                        <td>${(p.corner || p.y).toFixed(3)}</td>
                    </tr>
                `;
            } else if (isHCD) {
                bodyHtml += `
                    <tr>
                        <td>${idx + 1}</td>
                        <td>${p.x.toFixed(3)}</td>
                        <td>${p.y.toFixed(3)}</td>
                        <td>${p.a.toFixed(3)}</td>
                        <td>${p.b.toFixed(3)}</td>
                        <td>${(p.mouthRatio || 1.0).toFixed(3)}</td>
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
    if (enableHCDCheck) {
        enableHCDCheck.addEventListener('change', () => {
            if (enableHCDCheck.checked && morphTargetShapeSelect) {
                morphTargetShapeSelect.value = 'none';
            }
            updateHorn();
        });
    }

    if (morphTargetShapeSelect) {
        morphTargetShapeSelect.addEventListener('change', () => {
            if (morphTargetShapeSelect.value !== 'none' && enableHCDCheck) {
                enableHCDCheck.checked = false;
            }
            updateHorn();
        });
    }

    const allInputs = document.querySelectorAll('input, select');
    allInputs.forEach(input => {
        if (input.id !== 'enable-hcd' && input.id !== 'morph-target-shape') {
            input.addEventListener('change', () => updateHorn());
        }
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
        HornExporters.downloadFile(dxf, fileName, 'application/dxf');
    });

    btnExportSTL.addEventListener('click', () => {
        if (currentPoints.length === 0) return;
        const wallThickness = parseFloat(wallThicknessInput.value) || 2.0;
        const stlBuffer = HornExporters.generateSTL(currentPoints, currentIsHCD, wallThickness);
        const suffix = currentIsHCD ? '_HCD' : (currentIsMorph ? `_Morphed_${morphTargetShapeSelect.value}` : '');
        const fileName = `${hornTypeSelect.value}${suffix}_Model.stl`;
        HornExporters.downloadFile(stlBuffer, fileName, 'model/stl');
    });

    btnExportOBJ.addEventListener('click', () => {
        if (currentPoints.length === 0) return;
        const wallThickness = parseFloat(wallThicknessInput.value) || 2.0;
        const objText = HornExporters.generateOBJ(currentPoints, currentIsHCD, wallThickness);
        const suffix = currentIsHCD ? '_HCD' : (currentIsMorph ? `_Morphed_${morphTargetShapeSelect.value}` : '');
        const fileName = `${hornTypeSelect.value}${suffix}_Model.obj`;
        HornExporters.downloadFile(objText, fileName, 'text/plain');
    });

    btnExportSCAD.addEventListener('click', () => {
        if (currentPoints.length === 0) return;
        const wallThickness = parseFloat(wallThicknessInput.value) || 2.0;
        const scadText = HornExporters.generateOpenSCAD(currentPoints, currentIsHCD, wallThickness);
        const suffix = currentIsHCD ? '_HCD' : (currentIsMorph ? `_Morphed_${morphTargetShapeSelect.value}` : '');
        const fileName = `${hornTypeSelect.value}${suffix}_Script.scad`;
        HornExporters.downloadFile(scadText, fileName, 'text/x-scad');
    });

    btnExportCSV.addEventListener('click', () => {
        if (currentPoints.length === 0) return;
        const csvText = HornExporters.generateCSV(currentPoints, currentIsHCD, currentIsMorph);
        const suffix = currentIsHCD ? '_HCD' : (currentIsMorph ? `_Morphed_${morphTargetShapeSelect.value}` : '');
        const fileName = `${hornTypeSelect.value}${suffix}_Data.csv`;
        HornExporters.downloadFile(csvText, fileName, 'text/csv');
    });

    // Initial Trigger
    updateHorn();
});
