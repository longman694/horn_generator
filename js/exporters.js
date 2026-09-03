/**
 * Horn Exporter Library in Pure JavaScript
 * Provides client-side export functions for DXF, STL, OBJ, OpenSCAD, and CSV files.
 */

// Helper to trigger browser file download from string or ArrayBuffer
function downloadFile(content, fileName, mimeType) {
    if (!content) {
        console.error("downloadFile called with empty content.");
        return;
    }
    const blob = content instanceof Blob ? content : new Blob([content], { type: mimeType || 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    // Revoke URL after 10 seconds to allow async browser download to complete cleanly
    setTimeout(() => URL.revokeObjectURL(url), 10000);
}

/**
 * Generate CSV text from points data
 */
function generateCSV(points, isHCD = false, isMorph = false) {
    let csv = "";
    if (isMorph) {
        csv = "x (mm),y (mm),major_a (mm),minor_b (mm),corner_r (mm)\n";
        for (const p of points) {
            csv += `${p.x.toFixed(4)},${p.y.toFixed(4)},${(p.a || p.y).toFixed(4)},${(p.b || p.y).toFixed(4)},${(p.corner || p.y).toFixed(4)}\n`;
        }
    } else if (isHCD) {
        csv = "x (mm),y (mm),a (mm),b (mm),mouth_ratio\n";
        for (const p of points) {
            csv += `${p.x.toFixed(4)},${p.y.toFixed(4)},${p.a.toFixed(4)},${p.b.toFixed(4)},${(p.mouthRatio || 1.0).toFixed(4)}\n`;
        }
    } else {
        csv = "x (mm),y (mm)\n";
        for (const p of points) {
            csv += `${p.x.toFixed(4)},${p.y.toFixed(4)}\n`;
        }
    }
    return csv;
}

/**
 * Generate DXF string for 2D profile (AutoCAD R12 / AC1009 Specification)
 * DXF R12 is the universal CAD interchange standard supported natively by Fusion 360,
 * AutoCAD, SolidWorks, FreeCAD, Rhino, and Illustrator without requiring complex R2000
 * object handles or dictionary table boilerplate.
 */
function generateDXF(points) {
    if (!points || points.length === 0) return "";

    let minX = 0, minY = 0;
    let maxX = 0, maxY = 0;
    for (const p of points) {
        if (p.x > maxX) maxX = p.x;
        if (p.y > maxY) maxY = p.y;
    }

    let dxf = "0\nSECTION\n2\nHEADER\n";
    dxf += "9\n$ACADVER\n1\nAC1009\n"; // AutoCAD R12 format
    dxf += "9\n$INSUNITS\n70\n4\n";    // 4 = Millimeters
    dxf += `9\n$EXTMIN\n10\n${minX.toFixed(4)}\n20\n${minY.toFixed(4)}\n30\n0.0000\n`;
    dxf += `9\n$EXTMAX\n10\n${maxX.toFixed(4)}\n20\n${maxY.toFixed(4)}\n30\n0.0000\n`;
    dxf += "0\nENDSEC\n";

    // --- TABLES SECTION ---
    dxf += "0\nSECTION\n2\nTABLES\n";

    // LTYPE Table (BYBLOCK, BYLAYER, CONTINUOUS)
    dxf += "0\nTABLE\n2\nLTYPE\n70\n3\n";
    dxf += "0\nLTYPE\n2\nBYBLOCK\n70\n0\n3\n\n72\n65\n73\n0\n40\n0.0\n";
    dxf += "0\nLTYPE\n2\nBYLAYER\n70\n0\n3\n\n72\n65\n73\n0\n40\n0.0\n";
    dxf += "0\nLTYPE\n2\nCONTINUOUS\n70\n0\n3\nSolid line\n72\n65\n73\n0\n40\n0.0\n";
    dxf += "0\nENDTAB\n";

    // LAYER Table (Layer '0' and Layer 'CENTERLINE')
    dxf += "0\nTABLE\n2\nLAYER\n70\n2\n";
    dxf += "0\nLAYER\n2\n0\n70\n0\n62\n7\n6\nCONTINUOUS\n";
    dxf += "0\nLAYER\n2\nCENTERLINE\n70\n0\n62\n1\n6\nCONTINUOUS\n";
    dxf += "0\nENDTAB\n";

    dxf += "0\nENDSEC\n";

    // --- BLOCKS SECTION ---
    dxf += "0\nSECTION\n2\nBLOCKS\n0\nENDSEC\n";

    // --- ENTITIES SECTION ---
    dxf += "0\nSECTION\n2\nENTITIES\n";

    // Write POLYLINE entity (R12 Standard with VERTEX sub-entities)
    dxf += "0\nPOLYLINE\n";
    dxf += "8\n0\n";        // Layer 0
    dxf += "66\n1\n";        // Vertices follow flag (REQUIRED in R12)
    dxf += "70\n0\n";        // Open polyline
    dxf += "10\n0.0\n20\n0.0\n30\n0.0\n"; // Header dummy origin

    for (const p of points) {
        dxf += "0\nVERTEX\n";
        dxf += "8\n0\n";     // Layer 0
        dxf += `10\n${p.x.toFixed(4)}\n`;
        dxf += `20\n${p.y.toFixed(4)}\n`;
        dxf += "30\n0.0000\n";
    }

    dxf += "0\nSEQEND\n";
    dxf += "8\n0\n";

    // Add X-axis centerline
    dxf += "0\nLINE\n";
    dxf += "8\nCENTERLINE\n";
    dxf += "10\n0.0000\n20\n0.0000\n30\n0.0000\n";
    dxf += `11\n${maxX.toFixed(4)}\n21\n0.0000\n31\n0.0000\n`;

    dxf += "0\nENDSEC\n";
    dxf += "0\nEOF\n";

    return dxf;
}

/**
 * Generate OpenSCAD (.scad) code
 */
function generateOpenSCAD(points, isHCD = false, wallThickness = 2.0) {
    let scad = `// Horn Generator OpenSCAD Script\n`;
    scad += `// Generated: ${new Date().toISOString()}\n\n`;
    scad += `$fn = 60;\n`;
    scad += `wall_thickness = ${wallThickness};\n\n`;

    if (!isHCD) {
        scad += `// 2D Profile Points [x, y]\nprofile_points = [\n`;
        for (let i = 0; i < points.length; i++) {
            const p = points[i];
            scad += `  [${p.x.toFixed(3)}, ${p.y.toFixed(3)}]${i < points.length - 1 ? ',' : ''}\n`;
        }
        scad += `];\n\n`;

        scad += `module horn_solid() {\n`;
        scad += `    rotate_extrude()\n`;
        scad += `    polygon(points = concat(\n`;
        scad += `        profile_points,\n`;
        scad += `        [for (i = [len(profile_points)-1 : -1 : 0]) [profile_points[i][0], profile_points[i][1] + wall_thickness]]\n`;
        scad += `    ));\n`;
        scad += `}\n\n`;
        scad += `horn_solid();\n`;
    } else {
        scad += `// HCD Elliptical Loft Points [x, a, b]\nhcd_points = [\n`;
        for (let i = 0; i < points.length; i++) {
            const p = points[i];
            scad += `  [${p.x.toFixed(3)}, ${p.a.toFixed(3)}, ${p.b.toFixed(3)}]${i < points.length - 1 ? ',' : ''}\n`;
        }
        scad += `];\n\n`;
        scad += `echo("Use STL/OBJ export for complex 3D HCD loft geometries.");\n`;
    }

    return scad;
}

/**
 * Helper to build 3D mesh vertices and indices for solid Horn with wall thickness
 */
/**
 * Helper to compute radius at exact angle theta for any horn point p
 */
function getRadiusAtAngle(p, theta, isHCD) {
    if (isHCD && p.a !== undefined && p.b !== undefined) {
        const a = p.a;
        const b = p.b;
        const u = Math.cos(theta);
        const v = Math.sin(theta);
        const denom = Math.sqrt(Math.pow(b * u, 2) + Math.pow(a * v, 2));
        return denom > 1e-9 ? (a * b) / denom : a;
    }

    if (p.radii && p.radii.length > 0) {
        const numA = p.radii.length;
        let normTheta = theta % (2 * Math.PI);
        if (normTheta < 0) normTheta += 2 * Math.PI;
        const angleIdx = Math.round((normTheta / (2 * Math.PI)) * numA) % numA;
        return p.radii[angleIdx];
    }

    if (p.morphParams && p.morphParams.targetShape && p.morphParams.targetShape !== 'none') {
        const m = p.morphParams;
        const calcFn = typeof calculateTargetMouthRadius === 'function' 
            ? calculateTargetMouthRadius 
            : (typeof HornMath !== 'undefined' ? HornMath.calculateTargetMouthRadius : null);

        if (calcFn) {
            const rM = calcFn(theta, m.targetShape, m.targetWidth, m.targetHeight, m.cornerRadius);
            if (p.x < m.fixedPart) {
                return p.y;
            } else {
                const progress = (p.x - m.fixedPart) / Math.max(1e-6, m.length - m.fixedPart);
                const blend = Math.pow(progress, m.morphRate);
                return p.y + blend * (rM - m.rawMouthR);
            }
        }
    }
    
    return p.y;
}

/**
 * Helper to build 3D mesh vertices and indices for solid Horn with wall thickness
 */
function buildHornMeshGeometry(points, isHCD = false, wallThickness = 2.0, numRadial = 96) {
    const vertices = [];
    const triangles = [];

    const numPoints = points.length;
    const numRot = numRadial;

    // Outer and Inner ring vertices
    // Index layout:
    // Inner surface: ring i (0 to numPoints-1), angle j (0 to numRot-1) -> index: i * numRot + j
    // Outer surface: ring i (0 to numPoints-1), angle j (0 to numRot-1) -> index: numPoints * numRot + i * numRot + j

    const outerOffset = numPoints * numRot;

    // Generate Inner Surface Vertices
    for (let i = 0; i < numPoints; i++) {
        const p = points[i];

        for (let j = 0; j < numRot; j++) {
            const theta = (j * 2 * Math.PI) / numRot;
            const rInner = getRadiusAtAngle(p, theta, isHCD);

            // Three.js Coordinate Alignment:
            // X-axis: Axial horn length (throat -> mouth)
            // Y-axis: Vertical height (sin theta)
            // Z-axis: Horizontal width (cos theta)
            const x = p.x;
            const y = rInner * Math.sin(theta);
            const z = rInner * Math.cos(theta);
            vertices.push(x, y, z);
        }
    }

    // Generate Outer Surface Vertices (offset by wallThickness)
    for (let i = 0; i < numPoints; i++) {
        const p = points[i];

        for (let j = 0; j < numRot; j++) {
            const theta = (j * 2 * Math.PI) / numRot;
            const rInner = getRadiusAtAngle(p, theta, isHCD);
            const rOuter = rInner + wallThickness;

            const x = p.x;
            const y = rOuter * Math.sin(theta);
            const z = rOuter * Math.cos(theta);
            vertices.push(x, y, z);
        }
    }

    // Generate Quads/Triangles for Inner Tube
    for (let i = 0; i < numPoints - 1; i++) {
        for (let j = 0; j < numRot; j++) {
            const nextJ = (j + 1) % numRot;
            const idx1 = i * numRot + j;
            const idx2 = i * numRot + nextJ;
            const idx3 = (i + 1) * numRot + nextJ;
            const idx4 = (i + 1) * numRot + j;

            // Inner surface faces (facing inwards)
            triangles.push([idx1, idx3, idx2]);
            triangles.push([idx1, idx4, idx3]);
        }
    }

    // Generate Quads/Triangles for Outer Tube
    for (let i = 0; i < numPoints - 1; i++) {
        for (let j = 0; j < numRot; j++) {
            const nextJ = (j + 1) % numRot;
            const idx1 = outerOffset + i * numRot + j;
            const idx2 = outerOffset + i * numRot + nextJ;
            const idx3 = outerOffset + (i + 1) * numRot + nextJ;
            const idx4 = outerOffset + (i + 1) * numRot + j;

            // Outer surface faces (facing outwards)
            triangles.push([idx1, idx2, idx3]);
            triangles.push([idx1, idx3, idx4]);
        }
    }

    // Throat Rim Cap (i = 0)
    for (let j = 0; j < numRot; j++) {
        const nextJ = (j + 1) % numRot;
        const in1 = 0 * numRot + j;
        const in2 = 0 * numRot + nextJ;
        const out1 = outerOffset + 0 * numRot + j;
        const out2 = outerOffset + 0 * numRot + nextJ;

        triangles.push([in1, out2, in2]);
        triangles.push([in1, out1, out2]);
    }

    // Mouth Rim Cap (i = numPoints - 1)
    const lastI = numPoints - 1;
    for (let j = 0; j < numRot; j++) {
        const nextJ = (j + 1) % numRot;
        const in1 = lastI * numRot + j;
        const in2 = lastI * numRot + nextJ;
        const out1 = outerOffset + lastI * numRot + j;
        const out2 = outerOffset + lastI * numRot + nextJ;

        triangles.push([in1, in2, out2]);
        triangles.push([in1, out2, out1]);
    }

    return { vertices, triangles };
}

/**
 * Generate 3D STL File (Binary Format)
 */
function generateSTL(points, isHCD = false, wallThickness = 2.0) {
    const { vertices, triangles } = buildHornMeshGeometry(points, isHCD, wallThickness);

    const bufferLength = 80 + 4 + triangles.length * 50;
    const buffer = new ArrayBuffer(bufferLength);
    const view = new DataView(buffer);

    // 80-byte header
    const headerStr = "Horn Generator 3D STL Model";
    for (let i = 0; i < 80; i++) {
        view.setUint8(i, i < headerStr.length ? headerStr.charCodeAt(i) : 0);
    }

    // Number of triangles
    view.setUint32(80, triangles.length, true);

    let offset = 84;
    for (let t = 0; t < triangles.length; t++) {
        const tri = triangles[t];
        const p1 = [vertices[tri[0] * 3], vertices[tri[0] * 3 + 1], vertices[tri[0] * 3 + 2]];
        const p2 = [vertices[tri[1] * 3], vertices[tri[1] * 3 + 1], vertices[tri[1] * 3 + 2]];
        const p3 = [vertices[tri[2] * 3], vertices[tri[2] * 3 + 1], vertices[tri[2] * 3 + 2]];

        // Normal calculation
        const u = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        const v = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]];
        const nx = u[1] * v[2] - u[2] * v[1];
        const ny = u[2] * v[0] - u[0] * v[2];
        const nz = u[0] * v[1] - u[1] * v[0];
        const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;

        view.setFloat32(offset, nx / len, true);
        view.setFloat32(offset + 4, ny / len, true);
        view.setFloat32(offset + 8, nz / len, true);

        offset += 12;

        // Vertices
        view.setFloat32(offset, p1[0], true);
        view.setFloat32(offset + 4, p1[1], true);
        view.setFloat32(offset + 8, p1[2], true);
        offset += 12;

        view.setFloat32(offset, p2[0], true);
        view.setFloat32(offset + 4, p2[1], true);
        view.setFloat32(offset + 8, p2[2], true);
        offset += 12;

        view.setFloat32(offset, p3[0], true);
        view.setFloat32(offset + 4, p3[1], true);
        view.setFloat32(offset + 8, p3[2], true);
        offset += 12;

        // Attribute byte count
        view.setUint16(offset, 0, true);
        offset += 2;
    }

    return buffer;
}

/**
 * Generate 3D Wavefront OBJ File
 */
function generateOBJ(points, isHCD = false, wallThickness = 2.0) {
    const { vertices, triangles } = buildHornMeshGeometry(points, isHCD, wallThickness);

    let obj = "# Horn Generator 3D OBJ Model\n";

    for (let i = 0; i < vertices.length; i += 3) {
        obj += `v ${vertices[i].toFixed(4)} ${vertices[i + 1].toFixed(4)} ${vertices[i + 2].toFixed(4)}\n`;
    }

    for (const tri of triangles) {
        // OBJ indices are 1-based
        obj += `f ${tri[0] + 1} ${tri[1] + 1} ${tri[2] + 1}\n`;
    }

    return obj;
}

// Export for browser global and modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        downloadFile,
        generateCSV,
        generateDXF,
        generateOpenSCAD,
        generateSTL,
        generateOBJ,
        buildHornMeshGeometry
    };
} else {
    window.HornExporters = {
        downloadFile,
        generateCSV,
        generateDXF,
        generateOpenSCAD,
        generateSTL,
        generateOBJ,
        buildHornMeshGeometry
    };
}
