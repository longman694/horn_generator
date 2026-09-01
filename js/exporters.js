/**
 * Horn Exporter Library in Pure JavaScript
 * Provides client-side export functions for DXF, STL, OBJ, OpenSCAD, and CSV files.
 */

// Helper to trigger browser file download from string or ArrayBuffer
function downloadFile(content, fileName, mimeType) {
    const blob = content instanceof Blob ? content : new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 100);
}

/**
 * Generate CSV text from points data
 */
function generateCSV(points, isHCD = false) {
    let csv = isHCD
        ? "x (mm),y (mm),a (mm),b (mm),mouth_ratio\n"
        : "x (mm),y (mm)\n";

    for (const p of points) {
        if (isHCD) {
            csv += `${p.x.toFixed(4)},${p.y.toFixed(4)},${p.a.toFixed(4)},${p.b.toFixed(4)},${p.mouthRatio.toFixed(4)}\n`;
        } else {
            csv += `${p.x.toFixed(4)},${p.y.toFixed(4)}\n`;
        }
    }
    return csv;
}

/**
 * Generate DXF string for 2D profile (Fully Compliant AutoCAD 2000 / R2000 AC1015)
 * Includes required HEADER $HANDSEED, $INSUNITS, TABLES (LTYPE, LAYER, BLOCK_RECORD),
 * BLOCKS (*Model_Space, *Paper_Space), and OBJECTS dictionary for strict CAD software like AutoCAD & Fusion 360.
 */
function generateDXF(points) {
    if (!points || points.length === 0) return "";

    let dxf = "0\nSECTION\n2\nHEADER\n";
    dxf += "9\n$ACADVER\n1\nAC1015\n"; // AutoCAD 2000
    dxf += "9\n$HANDSEED\n5\nFFFF\n";  // Handle seed
    dxf += "9\n$INSUNITS\n70\n4\n";    // 4 = Millimeters
    dxf += "0\nENDSEC\n";

    // --- TABLES SECTION ---
    dxf += "0\nSECTION\n2\nTABLES\n";

    // VPORT Table
    dxf += "0\nTABLE\n2\nVPORT\n70\n0\n0\nENDTAB\n";

    // LTYPE Table
    dxf += "0\nTABLE\n2\nLTYPE\n70\n1\n";
    dxf += "0\nLTYPE\n2\nCONTINUOUS\n70\n0\n3\nSolid line\n73\n0\n40\n0.0\n";
    dxf += "0\nENDTAB\n";

    // LAYER Table (defining Layer '0' and Layer 'CENTERLINE')
    dxf += "0\nTABLE\n2\nLAYER\n70\n2\n";
    dxf += "0\nLAYER\n2\n0\n70\n0\n62\n7\n6\nCONTINUOUS\n";
    dxf += "0\nLAYER\n2\nCENTERLINE\n70\n0\n62\n1\n6\nCONTINUOUS\n";
    dxf += "0\nENDTAB\n";

    // STYLE, VIEW, UCS, APPID, DIMSTYLE Tables
    dxf += "0\nTABLE\n2\nSTYLE\n70\n0\n0\nENDTAB\n";
    dxf += "0\nTABLE\n2\nVIEW\n70\n0\n0\nENDTAB\n";
    dxf += "0\nTABLE\n2\nUCS\n70\n0\n0\nENDTAB\n";
    dxf += "0\nTABLE\n2\nAPPID\n70\n1\n0\nAPPID\n2\nACAD\n70\n0\n0\nENDTAB\n";
    dxf += "0\nTABLE\n2\nDIMSTYLE\n70\n0\n0\nENDTAB\n";

    // BLOCK_RECORD Table (*Model_Space & *Paper_Space)
    dxf += "0\nTABLE\n2\nBLOCK_RECORD\n70\n2\n";
    dxf += "0\nBLOCK_RECORD\n2\n*Model_Space\n";
    dxf += "0\nBLOCK_RECORD\n2\n*Paper_Space\n";
    dxf += "0\nENDTAB\n";

    dxf += "0\nENDSEC\n";

    // --- BLOCKS SECTION ---
    dxf += "0\nSECTION\n2\nBLOCKS\n";
    dxf += "0\nBLOCK\n2\n*Model_Space\n70\n0\n10\n0.0\n20\n0.0\n30\n0.0\n0\nENDBLK\n";
    dxf += "0\nBLOCK\n2\n*Paper_Space\n70\n0\n10\n0.0\n20\n0.0\n30\n0.0\n0\nENDBLK\n";
    dxf += "0\nENDSEC\n";

    // --- ENTITIES SECTION ---
    dxf += "0\nSECTION\n2\nENTITIES\n";

    // Write LWPOLYLINE entity for horn 2D profile
    dxf += "0\nLWPOLYLINE\n";
    dxf += "5\n30\n";                  // Entity handle
    dxf += "100\nAcDbEntity\n";        // Subclass marker
    dxf += "8\n0\n";                   // Layer 0
    dxf += "100\nAcDbPolyline\n";      // Subclass marker
    dxf += `90\n${points.length}\n`;   // Vertices count
    dxf += "70\n0\n";                  // Open polyline flag

    for (const p of points) {
        dxf += `10\n${p.x.toFixed(4)}\n`;
        dxf += `20\n${p.y.toFixed(4)}\n`;
    }

    // Add X-axis centerline from x=0 to x=max_x
    const maxX = points[points.length - 1].x;
    dxf += "0\nLINE\n";
    dxf += "5\n31\n";
    dxf += "100\nAcDbEntity\n";
    dxf += "8\nCENTERLINE\n";
    dxf += "100\nAcDbLine\n";
    dxf += "10\n0.0000\n20\n0.0000\n30\n0.0000\n";
    dxf += `11\n${maxX.toFixed(4)}\n21\n0.0000\n31\n0.0000\n`;

    dxf += "0\nENDSEC\n";

    // --- OBJECTS SECTION ---
    dxf += "0\nSECTION\n2\nOBJECTS\n";
    dxf += "0\nDICTIONARY\n5\nC\n100\nAcDbDictionary\n281\n1\n";
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
function buildHornMeshGeometry(points, isHCD = false, wallThickness = 2.0, numRadial = 48) {
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
        const a = isHCD ? p.a : p.y;
        const b = isHCD ? p.b : p.y;

        for (let j = 0; j < numRot; j++) {
            const theta = (j * 2 * Math.PI) / numRot;
            const x = p.x;
            const y = a * Math.cos(theta);
            const z = b * Math.sin(theta);
            vertices.push(x, y, z);
        }
    }

    // Generate Outer Surface Vertices (offset by wallThickness)
    for (let i = 0; i < numPoints; i++) {
        const p = points[i];
        const aOuter = (isHCD ? p.a : p.y) + wallThickness;
        const bOuter = (isHCD ? p.b : p.y) + wallThickness;

        for (let j = 0; j < numRot; j++) {
            const theta = (j * 2 * Math.PI) / numRot;
            const x = p.x;
            const y = aOuter * Math.cos(theta);
            const z = bOuter * Math.sin(theta);
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
