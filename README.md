# Horn Generator

A parametric acoustic horn profile generator built in **pure HTML5, Vanilla CSS, and JavaScript**.

Calculates exact 2D and 3D acoustic horn contours for loudspeaker design and provides real-time 3D WebGL preview alongside CAD vector (DXF) and 3D print mesh (STL, OBJ, OpenSCAD) exporters.

---

## ✨ Features

- **Multiple Horn Profile Types**:
  - **OS-SE (Oblate Spheroidal with Superellipse Termination)**: Custom coverage angle $\alpha$, throat angle $\alpha_0$, expansion factor $k$, superellipse aspect ratio $s$, truncation $q$, and exponent $n$. Auto-calculates cutoff frequency $f_c$.
  - **Tractrix**: Classic tractrix contour calculated from throat radius and cutoff frequency.
  - **Spherical**: Spherical wave flare with optional backward folding and fold-back filtering.
  - **Exponential**: Exponential area expansion $S = \pi r_0^2 e^{m x}$ with wave ratio $k_{\text{rm}} \le 1.0$ cutoff.

- **HCD (Hybrid Constant Directivity) Mode**:
  - Transforms circular cross-sections into elliptical lofts ($a_i$ semi-major, $b_i$ semi-minor).
  - Supports 6 transformation curves: `Linear`, `Parabolic`, `Exponential`, `Logarithmic`, `Hyperbolic`, and `Logistic`.

- **Interactive 3D WebGL Viewport (Three.js)**:
  - Drag to rotate, scroll to zoom, right-click to pan.
  - Real-time 3D solid wall thickness modeling.
  - **Shaded Surface**, **Wireframe Structure**, and **Half-Section Cutaway** view modes to inspect internal throat-to-mouth expansion.

- **2D Contour Visualization (Chart.js)**:
  - Responsive 2D plots for axial length $x$ (mm) vs radius $y$ (mm).
  - HCD multi-curve comparison ($y$, semi-minor $b$, semi-major $a$) and mouth aspect ratio transition curves.

- **Client-Side CAD & 3D Print Exporters**:
  - 📐 **DXF**: 2D polyline/spline vector format for CAD revolve operations (AutoCAD, SolidWorks, Fusion 360, FreeCAD).
  - 🖨️ **STL**: 3D binary mesh file ready for 3D printing in slicers (PrusaSlicer, Cura, Bambu Studio).
  - 📦 **OBJ**: Wavefront 3D mesh model format.
  - 📜 **OpenSCAD**: Parametric `.scad` script.
  - 📊 **CSV**: Export coordinate table data ($x, y, a, b, \text{Mouth Ratio}$).

---

## 🚀 Quick Start / Usage

No installation or build steps are required. Simply open `index.html` in any web browser!

### Running via Local HTTP Server:

```shell
# Using Python
python -m http.server 8080

# Or using Node npx
npx serve .
```

Then visit `http://localhost:8080` in your web browser.

---

## 🛠️ Tech Stack

- **Core**: HTML5, Vanilla CSS3, JavaScript (ES6)
- **3D Graphics**: [Three.js](https://threejs.org/) (WebGL & OrbitControls)
- **2D Graphing**: [Chart.js](https://www.chartjs.org/)

---

## 📄 License

MIT License.

