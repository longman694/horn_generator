#!/usr/bin/env python3
"""
AKABAK 3 / ABEC BEM Simulation Package Exporter
Generates full 3D BEM simulation files for AKABAK 3 / ABEC:
- Horn.stl (Sound-hard boundary surface mesh)
- Throat.stl (Planar velocity-driven diaphragm cap at x=0)
- Project.abec (Master ABEC project definition)
- solving.txt (BEM solver, physics & boundary conditions)
- observation.txt (Far-field horizontal/vertical directivity sonograms & RadImp)
- README.txt (Step-by-step execution guide)
"""

import argparse
import math
import os
import struct
import zipfile
import numpy as np
import pandas as pd

from lib import (
    generate_osse_horn,
    generate_tractrix_horn,
    generate_spherical_horn,
    generate_exponential_horn,
    generate_hcd_horn,
    calculate_target_mouth_radius,
    generate_osse_morphed_horn
)


def write_binary_stl(header_str: str, triangles: list) -> bytes:
    """Pack triangle list into standard 3D binary STL format."""
    header = header_str.encode('ascii')[:80].ljust(80, b'\0')
    num_tri = len(triangles)
    parts = [header, struct.pack('<I', num_tri)]
    for n, p1, p2, p3 in triangles:
        parts.append(struct.pack('<3f3f3f3fH',
            n[0], n[1], n[2],
            p1[0], p1[1], p1[2],
            p2[0], p2[1], p2[2],
            p3[0], p3[1], p3[2],
            0
        ))
    return b''.join(parts)


def calc_normal(p1, p2, p3):
    """Compute normalized cross product (p2 - p1) x (p3 - p1)."""
    u = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    v = (p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2])
    nx = u[1] * v[2] - u[2] * v[1]
    ny = u[2] * v[0] - u[0] * v[2]
    nz = u[0] * v[1] - u[1] * v[0]
    length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return (nx / length, ny / length, nz / length)


def generate_throat_stl(throat_r: float, length: float = 0.0, num_radial: int = 96, is_quarter: bool = False, first_row=None, is_hcd: bool = False) -> bytes:
    """
    Generate planar circular or elliptical driving diaphragm mesh at z = -length.
    Normals point in +Z (down the horn into the interior acoustic domain).
    Rim vertices match the horn throat ring vertices exactly.
    """
    triangles = []
    center = (0.0, 0.0, -length)
    num_tri = max(12, round(num_radial / 4)) if is_quarter else num_radial
    max_theta = (math.pi / 2) if is_quarter else (2 * math.pi)

    for j in range(num_tri):
        t1 = (j * max_theta) / num_tri
        t2 = ((j + 1) * max_theta) / num_tri

        r1 = get_radius_at_angle(first_row, t1, is_hcd) if first_row is not None else throat_r
        r2 = get_radius_at_angle(first_row, t2, is_hcd) if first_row is not None else throat_r

        # ABEC Standard Coordinate Convention:
        # Z: Axial propagation (throat at -length, mouth at 0)
        # X: Horizontal width (cos)
        # Y: Vertical height (sin)
        p1 = (r1 * math.cos(t1), r1 * math.sin(t1), -length)
        p2 = (r2 * math.cos(t2), r2 * math.sin(t2), -length)

        # Normal points in +Z: cross product (p1 - center) x (p2 - center)
        n = (0.0, 0.0, 1.0)
        triangles.append((n, center, p1, p2))

    header = "Horn Generator Quarter Throat Diaphragm STL" if is_quarter else "Horn Generator Throat Diaphragm STL"
    return write_binary_stl(header, triangles)


def generate_interface_stl(mouth_r: float, num_radial: int = 96, is_quarter: bool = False, last_row=None, is_hcd: bool = False) -> bytes:
    """
    Generate planar mouth interface aperture cap mesh at z = 0 (origin plane).
    Normals point in -Z (back into Subdomain 1 / interior cavity) per ABEC interface convention.
    Rim vertices match the horn mouth ring vertices exactly.
    """
    triangles = []
    center = (0.0, 0.0, 0.0)
    num_tri = max(12, round(num_radial / 4)) if is_quarter else num_radial
    max_theta = (math.pi / 2) if is_quarter else (2 * math.pi)

    for j in range(num_tri):
        t1 = (j * max_theta) / num_tri
        t2 = ((j + 1) * max_theta) / num_tri

        r1 = get_radius_at_angle(last_row, t1, is_hcd) if last_row is not None else mouth_r
        r2 = get_radius_at_angle(last_row, t2, is_hcd) if last_row is not None else mouth_r

        p1 = (r1 * math.cos(t1), r1 * math.sin(t1), 0.0)
        p2 = (r2 * math.cos(t2), r2 * math.sin(t2), 0.0)

        # Normal points in -Z: cross product (p2 - center) x (p1 - center)
        n = (0.0, 0.0, -1.0)
        triangles.append((n, center, p2, p1))

    header = "Horn Generator Quarter Mouth Interface STL" if is_quarter else "Horn Generator Mouth Interface STL"
    return write_binary_stl(header, triangles)


def get_radius_at_angle(row, theta, is_hcd=False):
    """Calculate the profile surface radius at polar angle theta."""
    if row is None:
        return 15.0
    if is_hcd:
        a = row['a'] if 'a' in row else row.get('a (mm)', None)
        b = row['b'] if 'b' in row else row.get('b (mm)', None)
        if a is not None and b is not None:
            u = math.cos(theta)
            v = math.sin(theta)
            denom = math.sqrt((b * u) ** 2 + (a * v) ** 2)
            return (a * b) / denom if denom > 1e-9 else a

    if 'y (mm)' in row:
        return row['y (mm)']
    if 'y' in row:
        return row['y']
    return 15.0


def generate_horn_stl(data, is_hcd: bool = False, num_radial: int = 96, is_quarter: bool = False) -> bytes:
    """
    Generate single-face horn surface mesh (wallThickness = 0).
    Mouth positioned at origin plane z = 0, throat positioned at z = -length.
    X: Horizontal width (cos), Y: Vertical height (sin), Z: Axial propagation.
    Normals oriented inwards toward the interior acoustic domain.
    Supports DataFrame (circular, HCD) and dict with r_matrix (morphed OS-SE).
    """
    triangles = []
    num_rot = max(12, round(num_radial / 4)) if is_quarter else num_radial
    max_theta = (math.pi / 2) if is_quarter else (2 * math.pi)
    stride = (num_rot + 1) if is_quarter else num_rot

    if isinstance(data, dict) and 'r_matrix' in data:
        raw_z = data['z']
        r_matrix = data['r_matrix']
        phi = data['phi_angles']
        num_z = len(raw_z)
        total_len = float(raw_z[-1])

        verts = []
        for i in range(num_z):
            # Shift z so mouth is at z = 0 and throat is at z = -total_len
            z_val = float(raw_z[i]) - total_len
            for j in range(stride):
                t = (j * max_theta) / num_rot
                norm_t = t % (2 * math.pi)
                phi_idx = (norm_t / (2 * math.pi)) * len(phi)
                i0 = int(phi_idx) % len(phi)
                i1 = (i0 + 1) % len(phi)
                frac = phi_idx - int(phi_idx)
                r = (1.0 - frac) * r_matrix[i, i0] + frac * r_matrix[i, i1]
                verts.append((r * math.cos(t), r * math.sin(t), z_val))

        for i in range(num_z - 1):
            for j in range(num_rot):
                next_j = (j + 1) if is_quarter else ((j + 1) % num_rot)
                idx1 = i * stride + j
                idx2 = i * stride + next_j
                idx3 = (i + 1) * stride + next_j
                idx4 = (i + 1) * stride + j

                v1 = verts[idx1]
                v2 = verts[idx2]
                v3 = verts[idx3]
                v4 = verts[idx4]

                # Inward-pointing normal winding
                n1 = calc_normal(v1, v3, v2)
                triangles.append((n1, v1, v3, v2))

                n2 = calc_normal(v1, v4, v3)
                triangles.append((n2, v1, v4, v3))
    else:
        df = data
        num_pts = len(df)
        total_len = float(df['x (mm)'].iloc[-1])
        verts = []

        for i in range(num_pts):
            row = df.iloc[i]
            # Shift z so mouth is at z = 0 and throat is at z = -total_len
            z_val = float(row['x (mm)']) - total_len
            for j in range(stride):
                t = (j * max_theta) / num_rot
                r = get_radius_at_angle(row, t, is_hcd)
                verts.append((r * math.cos(t), r * math.sin(t), z_val))

        for i in range(num_pts - 1):
            for j in range(num_rot):
                next_j = (j + 1) if is_quarter else ((j + 1) % num_rot)
                idx1 = i * stride + j
                idx2 = i * stride + next_j
                idx3 = (i + 1) * stride + next_j
                idx4 = (i + 1) * stride + j

                v1 = verts[idx1]
                v2 = verts[idx2]
                v3 = verts[idx3]
                v4 = verts[idx4]

                # Inward-pointing normal winding
                n1 = calc_normal(v1, v3, v2)
                triangles.append((n1, v1, v3, v2))

                n2 = calc_normal(v1, v4, v3)
                triangles.append((n2, v1, v4, v3))

    header = "Horn Generator Quarter Face Mesh STL" if is_quarter else "Horn Generator Single Face Mesh STL"
    return write_binary_stl(header, triangles)


def generate_abec_scripts(horn_params: dict) -> dict:
    """Generate script files for ABEC 3 / AKABAK 3 BEM simulation with Infinite Baffle."""
    horn_type = horn_params.get("horn_type", "OS-SE")
    throat_r = horn_params.get("throat_r", 15.0)
    mouth_r = horn_params.get("mouth_r", 50.0)
    length = horn_params.get("length", 50.0)
    cutoff_f = horn_params.get("cutoff_f", 1000.0)
    f1 = horn_params.get("f1", max(100, int(round(cutoff_f * 0.5))))
    f2 = horn_params.get("f2", 20000)
    symmetry = horn_params.get("symmetry", "quarter")
    is_quarter = (symmetry == "quarter")
    num_freq = horn_params.get("num_freq", 30 if is_quarter else 48)
    distance = horn_params.get("distance", 1.0)
    sym_line = "  Sym=xy\n" if is_quarter else ""

    project_abec = """// Master ABEC 3 Project Definition File
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
"""

    solving_txt = f"""// ABEC / AKABAK 3 Solving Script
// Boundary Element Method (BEM) Simulation with Infinite Baffle (2*pi steradians)
// Origin Plane Alignment: Mouth sits at z = 0, Throat recessed at z = -{length:.2f}mm
{"// Quarter-Symmetric Simulation (Sym=xy): Dual symmetry across X=0 and Y=0 planes (8x-16x BEM speedup)\n" if is_quarter else ""}
Control_Solver
  f1={f1}; f2={f2}; NumFrequencies={num_freq}
  Abscissa=log; Dim=3D; MeshFrequency={f2}
{sym_line}
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

// Throat Driving Diaphragm (Acoustic Velocity excitation at z = -{length:.2f}mm)
Elements "Throat_Diaphragm"
  Subdomain=1; MeshFileAlias="M2"
  201 Mesh Include ALL

Driving "Throat_Diaphragm"
  RefElements="Throat_Diaphragm"
  DrvGroup=1001
  DrvWeight=1.0
  Direction=z
  1  201  RefElements="Throat_Diaphragm"  Weight=1.0
"""

    observation_txt = f"""// ABEC / AKABAK 3 Observation Script
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
  Distance={distance}m
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
  Distance={distance}m
  1  Inclination=0  DrvGroups=1001  ID=102
"""

    readme_txt = f"""========================================================================
AKABAK 3 / ABEC - Horn BEM Directivity Simulation Package
========================================================================
Generated by:    Horn Profile Generator
Horn Type:       {horn_type}
Symmetry:        {"Quarter-Symmetric (Sym=xy) - 8x-16x BEM speedup" if is_quarter else "Full 360 deg Mesh (Standard)"}
Throat Radius:   {throat_r} mm (Throat Diameter: {throat_r * 2:.2f} mm at z = -{length:.2f} mm)
Mouth Radius:    {mouth_r} mm (Mouth Diameter: {mouth_r * 2:.2f} mm at z = 0.00 mm)
Axial Length:    {length} mm
Cutoff Freq:     {cutoff_f} Hz
Frequency Sweep: {num_freq} log points ({f1} Hz to {f2} Hz)
Driving Source:  Ideal Plane-Wave Diaphragm (Velocity = 1.0 m/s)
Acoustic Domain: Infinite Baffle on z = 0 (2*pi steradians Half-Space)

Simulation Files:
-----------------
- Project.abec     : Master ABEC project definition
- solving.txt      : BEM physics, Subdomains (1:Interior, 2:Exterior), Infinite Baffle & Boundaries
- observation.txt  : Far-field polar directivity arcs (Hor X-Z / Ver Y-Z) & RadImp
- Horn.stl         : Horn surface mesh (sound-hard boundary, z = -{length:.2f}mm to 0.00mm)
- Throat.stl       : Planar driving diaphragm cap at z = -{length:.2f}mm
- Interface.stl    : Planar mouth aperture interface mesh at z = 0.00mm
- README.txt       : Quick-start execution guide

Instructions to Run in AKABAK 3:
---------------------------------
1. Launch AKABAK (e.g., C:\\Program Files\\RDTeam\\AKABAK\\AKABAK.exe).
2. Select menu: Tools -> Import ABEC Project...
3. Browse and select "Project.abec" from this folder.
4. Click "Open", then click "Start Import".
5. Once verified, click "Apply" to build the AKABAK 3 simulation model.
{"   * Notice: In the AKABAK 3D viewport, the horn will automatically appear\\n     mirrored across X and Y symmetry planes as a complete horn!\\n" if is_quarter else ""}6. Press F5 (or click Calculate) to run the BEM frequency sweep.
7. In VACS, inspect the generated graphs:
   * "Directivity_Hor" : Horizontal Directivity Isobar Sonogram (-90 deg to +90 deg)
   * "Directivity_Ver" : Vertical Directivity Isobar Sonogram (-90 deg to +90 deg)
   * "RadImp"          : Throat Radiation Resistance & Reactance
========================================================================
"""

    return {
        "Project.abec": project_abec,
        "solving.txt": solving_txt,
        "observation.txt": observation_txt,
        "README.txt": readme_txt
    }


def main():
    parser = argparse.ArgumentParser(description="Export AKABAK 3 / ABEC BEM Simulation Package")
    parser.add_argument("--type", choices=["OS-SE", "Tractrix", "Spherical", "Exponential"], default="OS-SE", help="Horn Profile Type")
    parser.add_argument("--throat", type=float, default=15.0, help="Throat radius in mm")
    parser.add_argument("--fc", type=float, default=1000.0, help="Cutoff frequency in Hz")
    parser.add_argument("--length", type=float, default=50.0, help="Axial length in mm (OS-SE)")
    parser.add_argument("--alpha", type=float, default=45.0, help="Coverage angle alpha in degrees (OS-SE)")
    parser.add_argument("--alpha0", type=float, default=0.0, help="Throat angle alpha0 in degrees (OS-SE)")
    parser.add_argument("--k", type=float, default=1.0, help="Expansion factor k (OS-SE)")
    parser.add_argument("--s", type=float, default=0.8, help="Flare factor s (OS-SE)")
    parser.add_argument("--q", type=float, default=0.998, help="Truncation coeff q (OS-SE)")
    parser.add_argument("--n", type=float, default=5.0, help="Superellipse exponent n (OS-SE)")
    parser.add_argument("--points", type=int, default=30, help="Number of axial points")
    parser.add_argument("--radial-segments", type=int, default=96, help="Number of radial mesh divisions")

    # Symmetry & BEM Solver Arguments
    parser.add_argument("--symmetry", choices=["quarter", "full"], default="quarter", help="BEM symmetry mode (quarter: Sym=yz with 8x-16x speedup, full: full 360 deg)")
    parser.add_argument("--f1", type=float, default=200.0, help="BEM start frequency in Hz")
    parser.add_argument("--f2", type=float, default=20000.0, help="BEM end frequency in Hz")
    parser.add_argument("--num-freq", type=int, default=30, help="Number of frequency sweep points")
    parser.add_argument("--distance", type=float, default=1.0, help="Far-field observation distance in meters")

    # Surface Morphing Arguments
    parser.add_argument("--morph", choices=["none", "rectangle", "ellipse"], default="none", help="Ath Surface Morphing target shape")
    parser.add_argument("--morph-width", type=float, default=300.0, help="Target mouth width (mm)")
    parser.add_argument("--morph-height", type=float, default=200.0, help="Target mouth height (mm)")
    parser.add_argument("--morph-corner", type=float, default=20.0, help="Corner radius (mm)")
    parser.add_argument("--morph-rate", type=float, default=3.0, help="Morph rate gamma")
    parser.add_argument("--morph-fixed", type=float, default=0.0, help="Fixed throat portion (0.0 to 0.9)")
    parser.add_argument("--morph-shrinkage", action="store_true", help="Allow mouth shrinkage")

    # HCD Arguments
    parser.add_argument("--hcd", action="store_true", help="Enable Hybrid Constant Directivity (HCD)")
    parser.add_argument("--mouth-ratio", type=float, default=1.7, help="HCD mouth ratio")
    parser.add_argument("--hcd-mode", choices=["linear", "para", "exp", "log", "hyper", "logistic"], default="linear", help="HCD expansion curve")
    parser.add_argument("--hcd-acc", type=float, default=1.0, help="HCD accelerate factor")

    # Output Arguments
    parser.add_argument("--out", type=str, default="./akabak_sim", help="Output directory path")
    parser.add_argument("--zip", action="store_true", help="Also package output files into a .zip archive")

    args = parser.parse_args()

    # 1. Generate base profile
    is_morph = (args.type == "OS-SE" and args.morph != "none")
    is_hcd = (args.type != "OS-SE" and args.hcd)
    is_quarter = (args.symmetry == "quarter")
    first_row = None

    if args.type == "OS-SE":
        df_base = generate_osse_horn(
            args.throat, args.length, args.alpha, args.alpha0,
            args.k, args.s, args.q, args.n,
            num_points=args.points, plot=False
        )
        if is_morph:
            df_morphed = generate_osse_morphed_horn(
                args.throat, args.length, args.alpha, args.alpha0,
                args.k, args.s, args.q, args.n,
                target_shape=args.morph,
                target_width=args.morph_width,
                target_height=args.morph_height,
                corner_radius=args.morph_corner,
                fixed_part=args.morph_fixed,
                morph_rate=args.morph_rate,
                allow_shrinkage=args.morph_shrinkage,
                num_points=args.points,
                num_angles=args.radial_segments
            )
            model_data = df_morphed
            actual_length = float(df_morphed['z'][-1])
        else:
            model_data = df_base
            actual_length = float(df_base.iloc[-1]['x (mm)'])
            first_row = df_base.iloc[0]
    elif args.type == "Tractrix":
        df = generate_tractrix_horn(args.throat, args.fc, num_points=args.points, plot=False)
        model_data = df
        actual_length = float(df.iloc[-1]['x (mm)'])
        first_row = df.iloc[0]
    elif args.type == "Spherical":
        df = generate_spherical_horn(args.throat, args.fc, scale=4.0, fold=False, fold_back=True, plot=False)
        model_data = df
        actual_length = float(df.iloc[-1]['x (mm)'])
        first_row = df.iloc[0]
    elif args.type == "Exponential":
        df = generate_exponential_horn(args.throat, args.fc, scale=4.0, plot=False)
        model_data = df
        actual_length = float(df.iloc[-1]['x (mm)'])
        first_row = df.iloc[0]

    if is_hcd and args.type != "OS-SE":
        res = generate_hcd_horn(model_data, mouth_ratio=args.mouth_ratio, mode=args.hcd_mode, acc=args.hcd_acc, plot=False)
        df = res[0] if isinstance(res, (tuple, list)) else res
        model_data = df
        actual_length = float(df.iloc[-1]['x (mm)'])
        first_row = df.iloc[0]

    # Extract last_row for mouth interface
    if isinstance(model_data, dict) and 'r_matrix' in model_data:
        last_row = model_data['r_matrix'][-1, :]
        mouth_r = float(model_data['r_matrix'][-1, 0])
        first_row = model_data['r_matrix'][0, :]
    else:
        last_row = model_data.iloc[-1]
        mouth_r = float(last_row['y (mm)'] if 'y (mm)' in last_row else last_row.get('y', 50.0))

    horn_params = {
        "horn_type": args.type,
        "throat_r": args.throat,
        "mouth_r": mouth_r,
        "length": actual_length,
        "cutoff_f": args.fc,
        "f1": args.f1,
        "f2": args.f2,
        "num_freq": args.num_freq,
        "distance": args.distance,
        "symmetry": args.symmetry
    }

    # 2. Generate STL meshes
    horn_stl_bytes = generate_horn_stl(model_data, is_hcd=is_hcd, num_radial=args.radial_segments, is_quarter=is_quarter)
    throat_stl_bytes = generate_throat_stl(args.throat, length=actual_length, num_radial=args.radial_segments, is_quarter=is_quarter, first_row=first_row, is_hcd=is_hcd)
    interface_stl_bytes = generate_interface_stl(mouth_r, num_radial=args.radial_segments, is_quarter=is_quarter, last_row=last_row, is_hcd=is_hcd)

    # 3. Generate simulation scripts
    scripts = generate_abec_scripts(horn_params)

    # 4. Write to disk
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "Horn.stl"), "wb") as f:
        f.write(horn_stl_bytes)

    with open(os.path.join(args.out, "Throat.stl"), "wb") as f:
        f.write(throat_stl_bytes)

    with open(os.path.join(args.out, "Interface.stl"), "wb") as f:
        f.write(interface_stl_bytes)

    for filename, content in scripts.items():
        with open(os.path.join(args.out, filename), "w", encoding="utf-8") as f:
            f.write(content)

    sym_label = "Quarter-Symmetric (Sym=xy)" if is_quarter else "Full 360 deg Mesh"
    print(f"[OK] Exported AKABAK simulation package ({sym_label}) to: {args.out}")
    print(f"     - Horn.stl      : {len(horn_stl_bytes):,} bytes (z = -{actual_length:.2f}mm to 0.00mm)")
    print(f"     - Throat.stl    : {len(throat_stl_bytes):,} bytes (z = -{actual_length:.2f}mm)")
    print(f"     - Interface.stl : {len(interface_stl_bytes):,} bytes (z = 0.00mm origin plane)")
    for filename in scripts:
        print(f"     - {filename}")

    # 5. Optional ZIP archive
    if args.zip:
        sym_suffix = "_QuarterSym" if is_quarter else ""
        zip_path = os.path.join(args.out, f"{args.type}{sym_suffix}_AKABAK_Simulation.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(os.path.join(args.out, "Horn.stl"), "Horn.stl")
            zf.write(os.path.join(args.out, "Throat.stl"), "Throat.stl")
            zf.write(os.path.join(args.out, "Interface.stl"), "Interface.stl")
            for filename in scripts:
                zf.write(os.path.join(args.out, filename), filename)
        print(f"[OK] Created ZIP archive: {zip_path}")

    os._exit(0)


if __name__ == "__main__":
    main()
