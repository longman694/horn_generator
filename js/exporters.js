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

        if (wallThickness <= 0) {
            scad += `module horn_surface() {\n`;
            scad += `    // Single face profile (zero thickness surface)\n`;
            scad += `    rotate_extrude()\n`;
            scad += `    polygon(points = profile_points);\n`;
            scad += `}\n\n`;
            scad += `horn_surface();\n`;
        } else {
            scad += `module horn_solid() {\n`;
            scad += `    rotate_extrude()\n`;
            scad += `    polygon(points = concat(\n`;
            scad += `        profile_points,\n`;
            scad += `        [for (i = [len(profile_points)-1 : -1 : 0]) [profile_points[i][0], profile_points[i][1] + wall_thickness]]\n`;
            scad += `    ));\n`;
            scad += `}\n\n`;
            scad += `horn_solid();\n`;
        }
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
 * Helper to build 3D mesh vertices and indices for solid Horn with wall thickness,
 * or single face horn surface if wallThickness <= 0.
 */
function buildHornMeshGeometry(points, isHCD = false, wallThickness = 2.0, numRadial = 96, isQuarter = false) {
    const vertices = [];
    const triangles = [];

    const numPoints = points.length;
    const numRot = isQuarter ? Math.max(12, Math.round(numRadial / 4)) : numRadial;
    const maxTheta = isQuarter ? (Math.PI / 2) : (2 * Math.PI);
    const stride = isQuarter ? (numRot + 1) : numRot;
    const isSingleFace = (wallThickness <= 0);

    // Outer and Inner ring vertices
    // Inner surface: ring i (0 to numPoints-1), angle j (0 to stride-1)
    // Outer surface (if solid): offset = numPoints * stride
    const outerOffset = numPoints * stride;

    // Generate Surface Vertices (Inner/Primary profile)
    for (let i = 0; i < numPoints; i++) {
        const p = points[i];
        const jCount = isQuarter ? (numRot + 1) : numRot;

        for (let j = 0; j < jCount; j++) {
            const theta = (j * maxTheta) / numRot;
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

    if (!isSingleFace) {
        // Generate Outer Surface Vertices (offset by wallThickness)
        for (let i = 0; i < numPoints; i++) {
            const p = points[i];
            const jCount = isQuarter ? (numRot + 1) : numRot;

            for (let j = 0; j < jCount; j++) {
                const theta = (j * maxTheta) / numRot;
                const rInner = getRadiusAtAngle(p, theta, isHCD);
                const rOuter = rInner + wallThickness;

                const x = p.x;
                const y = rOuter * Math.sin(theta);
                const z = rOuter * Math.cos(theta);
                vertices.push(x, y, z);
            }
        }
    }

    // Generate Quads/Triangles for Inner Tube / Surface
    for (let i = 0; i < numPoints - 1; i++) {
        for (let j = 0; j < numRot; j++) {
            const nextJ = isQuarter ? (j + 1) : ((j + 1) % numRot);
            const idx1 = i * stride + j;
            const idx2 = i * stride + nextJ;
            const idx3 = (i + 1) * stride + nextJ;
            const idx4 = (i + 1) * stride + j;

            if (isSingleFace) {
                // Outward facing normals for single surface
                triangles.push([idx1, idx2, idx3]);
                triangles.push([idx1, idx3, idx4]);
            } else {
                // Inner surface faces (facing inwards towards bore)
                triangles.push([idx1, idx3, idx2]);
                triangles.push([idx1, idx4, idx3]);
            }
        }
    }

    if (!isSingleFace) {
        // Generate Quads/Triangles for Outer Tube
        for (let i = 0; i < numPoints - 1; i++) {
            for (let j = 0; j < numRot; j++) {
                const nextJ = isQuarter ? (j + 1) : ((j + 1) % numRot);
                const idx1 = outerOffset + i * stride + j;
                const idx2 = outerOffset + i * stride + nextJ;
                const idx3 = outerOffset + (i + 1) * stride + nextJ;
                const idx4 = outerOffset + (i + 1) * stride + j;

                // Outer surface faces (facing outwards)
                triangles.push([idx1, idx2, idx3]);
                triangles.push([idx1, idx3, idx4]);
            }
        }

        // Throat Rim Cap (i = 0)
        for (let j = 0; j < numRot; j++) {
            const nextJ = isQuarter ? (j + 1) : ((j + 1) % numRot);
            const in1 = 0 * stride + j;
            const in2 = 0 * stride + nextJ;
            const out1 = outerOffset + 0 * stride + j;
            const out2 = outerOffset + 0 * stride + nextJ;

            triangles.push([in1, out2, in2]);
            triangles.push([in1, out1, out2]);
        }

        // Mouth Rim Cap (i = numPoints - 1)
        const lastI = numPoints - 1;
        for (let j = 0; j < numRot; j++) {
            const nextJ = isQuarter ? (j + 1) : ((j + 1) % numRot);
            const in1 = lastI * stride + j;
            const in2 = lastI * stride + nextJ;
            const out1 = outerOffset + lastI * stride + j;
            const out2 = outerOffset + lastI * stride + nextJ;

            triangles.push([in1, in2, out2]);
            triangles.push([in1, out2, out1]);
        }
    }

    return { vertices, triangles };
}

/**
 * Generate 3D STL File (Binary Format)
 */
function generateSTL(points, isHCD = false, wallThickness = 2.0, numRadial = 96, isQuarter = false) {
    const { vertices, triangles } = buildHornMeshGeometry(points, isHCD, wallThickness, numRadial, isQuarter);

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

/**
 * Generate 3D STL for circular throat driving diaphragm cap at x = 0 (Binary Format)
 * Normal vector points in +X (down the horn into the radiation field).
 * Rim vertices match buildHornMeshGeometry(points, ..., 0, numRadial) at ring 0.
 */
/**
 * Generate 3D STL for circular or elliptical throat driving diaphragm cap at z = -length (Binary Format)
 * Normal vector points in +Z (down the horn into the interior acoustic domain).
 * Rim vertices match horn bore at ring 0.
 */
function generateThroatCapSTL(throatRadius = 15.0, length = 0.0, numRadial = 96, isQuarter = false, firstPoint = null, isHCD = false) {
    const numTri = isQuarter ? Math.max(12, Math.round(numRadial / 4)) : numRadial;
    const maxTheta = isQuarter ? (Math.PI / 2) : (2 * Math.PI);
    const bufferLength = 80 + 4 + numTri * 50;
    const buffer = new ArrayBuffer(bufferLength);
    const view = new DataView(buffer);

    // 80-byte header
    const headerStr = isQuarter ? "Horn Generator Quarter Throat Diaphragm STL" : "Horn Generator Throat Diaphragm STL";
    for (let i = 0; i < 80; i++) {
        view.setUint8(i, i < headerStr.length ? headerStr.charCodeAt(i) : 0);
    }

    // Number of triangles
    view.setUint32(80, numTri, true);

    let offset = 84;
    const center = [0.0, 0.0, -length];

    for (let j = 0; j < numTri; j++) {
        const theta1 = (j * maxTheta) / numTri;
        const theta2 = ((j + 1) * maxTheta) / numTri;

        const r1 = firstPoint ? getRadiusAtAngle(firstPoint, theta1, isHCD) : throatRadius;
        const r2 = firstPoint ? getRadiusAtAngle(firstPoint, theta2, isHCD) : throatRadius;

        // ABEC Standard Coordinates:
        // X: Horizontal width (cos), Y: Vertical height (sin), Z: -length
        const p1 = [r1 * Math.cos(theta1), r1 * Math.sin(theta1), -length];
        const p2 = [r2 * Math.cos(theta2), r2 * Math.sin(theta2), -length];

        // Normal points in +Z: cross product (p1 - center) x (p2 - center)
        const nx = 0.0, ny = 0.0, nz = 1.0;

        view.setFloat32(offset, nx, true);
        view.setFloat32(offset + 4, ny, true);
        view.setFloat32(offset + 8, nz, true);
        offset += 12;

        // Center
        view.setFloat32(offset, center[0], true);
        view.setFloat32(offset + 4, center[1], true);
        view.setFloat32(offset + 8, center[2], true);
        offset += 12;

        // p1 (theta1)
        view.setFloat32(offset, p1[0], true);
        view.setFloat32(offset + 4, p1[1], true);
        view.setFloat32(offset + 8, p1[2], true);
        offset += 12;

        // p2 (theta2)
        view.setFloat32(offset, p2[0], true);
        view.setFloat32(offset + 4, p2[1], true);
        view.setFloat32(offset + 8, p2[2], true);
        offset += 12;

        view.setUint16(offset, 0, true);
        offset += 2;
    }

    return buffer;
}

/**
 * Generate 3D STL for mouth aperture interface cap at z = 0 (origin plane, Binary Format)
 * Normal vector points in -Z (back into Subdomain 1 / interior cavity) per ABEC interface convention.
 * Rim vertices match horn mouth rim at ring N-1.
 */
function generateMouthInterfaceSTL(mouthRadius = 50.0, numRadial = 96, isQuarter = false, lastPoint = null, isHCD = false) {
    const numTri = isQuarter ? Math.max(12, Math.round(numRadial / 4)) : numRadial;
    const maxTheta = isQuarter ? (Math.PI / 2) : (2 * Math.PI);
    const bufferLength = 80 + 4 + numTri * 50;
    const buffer = new ArrayBuffer(bufferLength);
    const view = new DataView(buffer);

    const headerStr = isQuarter ? "Horn Generator Quarter Mouth Interface STL" : "Horn Generator Mouth Interface STL";
    for (let i = 0; i < 80; i++) {
        view.setUint8(i, i < headerStr.length ? headerStr.charCodeAt(i) : 0);
    }

    view.setUint32(80, numTri, true);

    let offset = 84;
    const center = [0.0, 0.0, 0.0];

    for (let j = 0; j < numTri; j++) {
        const theta1 = (j * maxTheta) / numTri;
        const theta2 = ((j + 1) * maxTheta) / numTri;

        const r1 = lastPoint ? getRadiusAtAngle(lastPoint, theta1, isHCD) : mouthRadius;
        const r2 = lastPoint ? getRadiusAtAngle(lastPoint, theta2, isHCD) : mouthRadius;

        // ABEC Standard Coordinates:
        // X: Horizontal width (cos), Y: Vertical height (sin), Z: 0.0
        const p1 = [r1 * Math.cos(theta1), r1 * Math.sin(theta1), 0.0];
        const p2 = [r2 * Math.cos(theta2), r2 * Math.sin(theta2), 0.0];

        // Normal points in -Z: cross product (p2 - center) x (p1 - center)
        const nx = 0.0, ny = 0.0, nz = -1.0;

        view.setFloat32(offset, nx, true);
        view.setFloat32(offset + 4, ny, true);
        view.setFloat32(offset + 8, nz, true);
        offset += 12;

        // Center
        view.setFloat32(offset, center[0], true);
        view.setFloat32(offset + 4, center[1], true);
        view.setFloat32(offset + 8, center[2], true);
        offset += 12;

        // p2 (theta2)
        view.setFloat32(offset, p2[0], true);
        view.setFloat32(offset + 4, p2[1], true);
        view.setFloat32(offset + 8, p2[2], true);
        offset += 12;

        // p1 (theta1)
        view.setFloat32(offset, p1[0], true);
        view.setFloat32(offset + 4, p1[1], true);
        view.setFloat32(offset + 8, p1[2], true);
        offset += 12;

        view.setUint16(offset, 0, true);
        offset += 2;
    }

    return buffer;
}

/**
 * Generate 3D STL for single-face Horn wall mesh aligned with ABEC Z-axis (Binary Format)
 * Mouth sits on origin plane z = 0, throat sits recessed at z = -length.
 * Normals point inward toward the central acoustic domain axis.
 */
function generateHornBEMSTL(points, isHCD = false, numRadial = 96, isQuarter = false) {
    const numPoints = points.length;
    const totalLen = numPoints > 0 ? points[numPoints - 1].x : 0.0;
    const numRot = isQuarter ? Math.max(12, Math.round(numRadial / 4)) : numRadial;
    const maxTheta = isQuarter ? (Math.PI / 2) : (2 * Math.PI);
    const stride = isQuarter ? (numRot + 1) : numRot;

    const vertices = [];
    for (let i = 0; i < numPoints; i++) {
        const p = points[i];
        // Shift z so mouth is at z = 0 and throat is at z = -totalLen
        const zVal = p.x - totalLen;
        const jCount = isQuarter ? (numRot + 1) : numRot;

        for (let j = 0; j < jCount; j++) {
            const theta = (j * maxTheta) / numRot;
            const rInner = getRadiusAtAngle(p, theta, isHCD);

            // ABEC Coordinate Alignment:
            // X: Horizontal width (cos theta)
            // Y: Vertical height (sin theta)
            // Z: Axial length (throat -totalLen -> mouth 0)
            const x = rInner * Math.cos(theta);
            const y = rInner * Math.sin(theta);
            const z = zVal;
            vertices.push(x, y, z);
        }
    }

    const triangles = [];
    for (let i = 0; i < numPoints - 1; i++) {
        for (let j = 0; j < numRot; j++) {
            const nextJ = isQuarter ? (j + 1) : ((j + 1) % numRot);
            const idx1 = i * stride + j;
            const idx2 = i * stride + nextJ;
            const idx3 = (i + 1) * stride + nextJ;
            const idx4 = (i + 1) * stride + j;

            // Inward-pointing normal winding
            triangles.push([idx1, idx3, idx2]);
            triangles.push([idx1, idx4, idx3]);
        }
    }

    const bufferLength = 80 + 4 + triangles.length * 50;
    const buffer = new ArrayBuffer(bufferLength);
    const view = new DataView(buffer);

    const headerStr = isQuarter ? "Horn Generator Quarter Face BEM STL" : "Horn Generator Single Face BEM STL";
    for (let i = 0; i < 80; i++) {
        view.setUint8(i, i < headerStr.length ? headerStr.charCodeAt(i) : 0);
    }
    view.setUint32(80, triangles.length, true);

    let offset = 84;
    for (let t = 0; t < triangles.length; t++) {
        const tri = triangles[t];
        const p1 = [vertices[tri[0] * 3], vertices[tri[0] * 3 + 1], vertices[tri[0] * 3 + 2]];
        const p2 = [vertices[tri[1] * 3], vertices[tri[1] * 3 + 1], vertices[tri[1] * 3 + 2]];
        const p3 = [vertices[tri[2] * 3], vertices[tri[2] * 3 + 1], vertices[tri[2] * 3 + 2]];

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

        view.setUint16(offset, 0, true);
        offset += 2;
    }

    return buffer;
}

/**
 * Generate ABEC / AKABAK 3 BEM Project Scripts
 */
function generateABECProjectScripts(hornParams = {}, hornName = "Horn") {
    const hornType = hornParams.hornType || "OS-SE";
    const throatR = hornParams.throatR !== undefined ? hornParams.throatR : 15.0;
    const mouthR = hornParams.mouthR !== undefined ? hornParams.mouthR : 50.0;
    const length = hornParams.length !== undefined ? hornParams.length : 50.0;
    const cutoffF = hornParams.cutoffF !== undefined ? hornParams.cutoffF : 1000.0;
    const f1 = hornParams.f1 || Math.max(100, Math.round(cutoffF * 0.5));
    const f2 = hornParams.f2 || 20000;
    const numFreq = hornParams.numFreq || (hornParams.symmetry === 'quarter' ? 30 : 48);
    const distance = hornParams.distance || 1.0;
    const isQuarter = hornParams.symmetry === 'quarter';
    const symLine = isQuarter ? "  Sym=xy\n" : "";

    const projectAbec = `// Master ABEC 3 Project Definition File
// Compatible with ABEC 3 and AKABAK 3 (Tools -> Import ABEC Project...)

[Project]
Scriptname_InfoFile=README.txt
[Solving]
Scriptname_Solving=solving.txt
[DirectSound]
Scriptname_DirectSound=
[LEScript]
Scriptname_LEScript=
[Observation]
C0=observation.txt
[MeshFiles]
C0=Horn.stl,M1
C1=Throat.stl,M2
C2=Interface.stl,M3
`;

    const solvingTxt = `// ABEC / AKABAK 3 Solving Script
// Boundary Element Method (BEM) Simulation with Infinite Baffle (2*pi steradians)
// Origin Plane Alignment: Mouth sits at z = 0, Throat recessed at z = -${length.toFixed(2)}mm
${isQuarter ? '// Quarter-Symmetric Simulation (Sym=xy): Dual symmetry across X=0 and Y=0 planes (8x-16x BEM speedup)\n' : ''}
Control_Solver
  f1=${f1}; f2=${f2}; NumFrequencies=${numFreq}
  Abscissa=log; Dim=3D; MeshFrequency=${f2}
${symLine}
MeshFile_Properties
  MeshFileAlias="M1"; Scale=1mm

MeshFile_Properties
  MeshFileAlias="M2"; Scale=1mm

MeshFile_Properties
  MeshFileAlias="M3"; Scale=1mm

// Subdomain 1: Enclosed Horn Interior Volume
SubDomain_Properties
  SubDomain=1; ElType=Interior

// Subdomain 2: Exterior Radiation Half-Space in front of Infinite Baffle (z = 0)
SubDomain_Properties
  SubDomain=2; ElType=Exterior; IBPlane=z; IBOffset=0mm

// Horn Wall Boundary (Rigid sound-hard boundary: vn = 0)
Elements "Horn_Wall"
  Subdomain=1; MeshFileAlias="M1"
  101 Mesh Include ALL

// Mouth Interface (Couples interior Subdomain 1 to exterior Subdomain 2)
Elements "Mouth_Interface"
  Subdomain=1,2; MeshFileAlias="M3"
  301 Mesh Include ALL

// Throat Driving Diaphragm (Acoustic Velocity excitation at z = -${length.toFixed(2)}mm)
Elements "Throat_Diaphragm"
  Subdomain=1; MeshFileAlias="M2"
  201 Mesh Include ALL

Driving "Throat_Diaphragm"
  RefElements="Throat_Diaphragm"
  DrvGroup=1001
  DrvWeight=1.0
  Direction=z
  1  201  RefElements="Throat_Diaphragm"  Weight=1.0
`;

    const observationTxt = `// ABEC / AKABAK 3 Observation Script
// Far-field Directivity & Acoustic Radiation Impedance (Front Half-Space)

Driving_Values
  DrvType=Velocity; Value=1.0
  1  DrvGroup=1001  Weight=1.0  Delay=0.0

// Throat Radiation Impedance (Real & Imaginary acoustic loading)
Radiation_Impedance
  GraphHeader="RadImp"
  BodeType=Complex
  RadImpType=Normalized
  Range_min=0; Range_max=2
  1  1001  1001  ID=1001

// Horizontal Directivity Sonogram (-90 to +90 deg in horizontal X-Z plane, On-axis = +Z)
BE_Spectrum
  PlotType=Polar
  GraphHeader="Directivity_Hor"
  BodeType=LeveldB
  Range_min=-45; Range_max=5
  PolarRange=-90,90,91
  BasePlane=zx
  Farfield=true
  Distance=${distance}m
  1  Inclination=0  DrvGroups=1001  ID=101

// Vertical Directivity Sonogram (-90 to +90 deg in vertical Y-Z plane, On-axis = +Z)
BE_Spectrum
  PlotType=Polar
  GraphHeader="Directivity_Ver"
  BodeType=LeveldB
  Range_min=-45; Range_max=5
  PolarRange=-90,90,91
  BasePlane=zy
  Farfield=true
  Distance=${distance}m
  1  Inclination=0  DrvGroups=1001  ID=102
`;

    const readmeTxt = `========================================================================
AKABAK 3 / ABEC - Horn BEM Directivity Simulation Package
========================================================================
Generated by:    Horn Profile Generator (Pure HTML & JavaScript)
Horn Type:       ${hornType}
Symmetry:        ${isQuarter ? 'Quarter-Symmetric (Sym=xy) - 8x-16x BEM speedup' : 'Full 360 deg Mesh (Standard)'}
Throat Radius:   ${throatR} mm (Throat Diameter: ${(throatR * 2).toFixed(2)} mm at z = -${length.toFixed(2)} mm)
Mouth Radius:    ${mouthR} mm (Mouth Diameter: ${(mouthR * 2).toFixed(2)} mm at z = 0.00 mm)
Axial Length:    ${length} mm
Cutoff Freq:     ${cutoffF} Hz
Frequency Sweep: ${numFreq} log points (${f1} Hz to ${f2} Hz)
Driving Source:  Ideal Plane-Wave Diaphragm (Velocity = 1.0 m/s)
Acoustic Domain: Infinite Baffle on z = 0 (2*pi steradians Half-Space)

Simulation Files:
-----------------
- Project.abec     : Master ABEC project definition
- solving.txt      : BEM physics, Subdomains (1:Interior, 2:Exterior), Infinite Baffle & Boundaries
- observation.txt  : Far-field polar directivity arcs (Hor X-Z / Ver Y-Z) & RadImp
- Horn.stl         : Horn surface mesh (sound-hard boundary, z = -${length.toFixed(2)}mm to 0.00mm)
- Throat.stl       : Planar driving diaphragm cap at z = -${length.toFixed(2)}mm
- Interface.stl    : Planar mouth aperture interface mesh at z = 0.00mm
- README.txt       : Quick-start execution guide

Instructions to Run in AKABAK 3:
---------------------------------
1. Launch AKABAK (e.g., C:\\Program Files\\RDTeam\\AKABAK\\AKABAK.exe).
2. Select menu: Tools -> Import ABEC Project...
3. Browse and select "Project.abec" from this extracted folder.
4. Click "Open", then click "Start Import".
5. Once verified, click "Apply" to build the AKABAK 3 simulation model.
${isQuarter ? '   * Notice: In the AKABAK 3D viewport, the horn will automatically appear\\n     mirrored across X and Y symmetry planes as a complete horn!\\n' : ''}6. Press F5 (or click Calculate) to run the BEM frequency sweep.
7. In VACS, inspect the generated graphs:
   * "Directivity_Hor" : Horizontal Directivity Isobar Sonogram (-90 deg to +90 deg)
   * "Directivity_Ver" : Vertical Directivity Isobar Sonogram (-90 deg to +90 deg)
   * "RadImp"          : Throat Radiation Resistance & Reactance
========================================================================
`;

    return { projectAbec, solvingTxt, observationTxt, readmeTxt };
}

/**
 * Client-Side ZIP Export for AKABAK 3 / ABEC Simulation Package
 */
function exportAKABAKZip(points, isHCD = false, isMorph = false, hornParams = {}, hornName = "Horn") {
    if (typeof JSZip === 'undefined') {
        alert("JSZip library is required to bundle the simulation package. Please check network connection.");
        return;
    }

    const isQuarter = hornParams.symmetry === 'quarter';
    const totalLen = points.length > 0 ? points[points.length - 1].x : (hornParams.length || 50.0);
    const mouthPoint = points.length > 0 ? points[points.length - 1] : null;
    const mouthR = mouthPoint ? mouthPoint.y : 50.0;
    const throatR = hornParams.throatR !== undefined ? hornParams.throatR : (points[0] ? points[0].y : 15.0);

    hornParams.length = totalLen;
    hornParams.mouthR = mouthR;
    hornParams.throatR = throatR;

    // 1. Generate single-face Horn STL aligned with Z-axis (mouth at z=0, throat at z=-totalLen)
    const hornStlBuffer = generateHornBEMSTL(points, isHCD, 96, isQuarter);

    // 2. Generate flat Throat driving diaphragm cap at z = -totalLen
    const throatStlBuffer = generateThroatCapSTL(throatR, totalLen, 96, isQuarter, points[0], isHCD);

    // 3. Generate flat Mouth interface cap at z = 0
    const interfaceStlBuffer = generateMouthInterfaceSTL(mouthR, 96, isQuarter, mouthPoint, isHCD);

    // 4. Generate simulation scripts
    const scripts = generateABECProjectScripts(hornParams, hornName);

    // 5. Bundle into ZIP
    const zip = new JSZip();
    zip.file("Horn.stl", hornStlBuffer);
    zip.file("Throat.stl", throatStlBuffer);
    zip.file("Interface.stl", interfaceStlBuffer);
    zip.file("Project.abec", scripts.projectAbec);
    zip.file("solving.txt", scripts.solvingTxt);
    zip.file("observation.txt", scripts.observationTxt);
    zip.file("README.txt", scripts.readmeTxt);

    const symSuffix = isQuarter ? "_QuarterSym" : "";
    const zipFileName = `${hornName}${symSuffix}_AKABAK_Simulation.zip`;

    zip.generateAsync({ type: "blob" }).then(function(blob) {
        downloadFile(blob, zipFileName, "application/zip");
    }).catch(function(err) {
        console.error("Failed to generate AKABAK ZIP archive:", err);
        alert("Failed to create ZIP file: " + err.message);
    });
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
        buildHornMeshGeometry,
        generateHornBEMSTL,
        generateThroatCapSTL,
        generateMouthInterfaceSTL,
        generateABECProjectScripts,
        exportAKABAKZip
    };
} else {
    window.HornExporters = {
        downloadFile,
        generateCSV,
        generateDXF,
        generateOpenSCAD,
        generateSTL,
        generateOBJ,
        buildHornMeshGeometry,
        generateHornBEMSTL,
        generateThroatCapSTL,
        generateMouthInterfaceSTL,
        generateABECProjectScripts,
        exportAKABAKZip
    };
}
