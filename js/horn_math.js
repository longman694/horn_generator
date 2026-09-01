/**
 * Horn Generator Math Library in Pure JavaScript
 * Provides functions for generating OS-SE, Tractrix, Spherical, Exponential, and HCD horn profiles.
 */

// --- Cubic Spline Interpolation Implementation ---
class CubicSpline {
    constructor(x, y) {
        this.x = x.slice();
        this.y = y.slice();
        const n = x.length - 1;
        if (n < 1) throw new Error("CubicSpline requires at least 2 points.");

        const h = new Array(n);
        for (let i = 0; i < n; i++) {
            h[i] = x[i + 1] - x[i];
        }

        const alpha = new Array(n);
        alpha[0] = 0;
        for (let i = 1; i < n; i++) {
            alpha[i] = (3 / h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (y[i] - y[i - 1]);
        }

        const l = new Array(n + 1);
        const mu = new Array(n + 1);
        const z = new Array(n + 1);

        l[0] = 1;
        mu[0] = 0;
        z[0] = 0;

        for (let i = 1; i < n; i++) {
            l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        l[n] = 1;
        z[n] = 0;

        this.c = new Array(n + 1);
        this.b = new Array(n);
        this.d = new Array(n);
        this.a = y.slice(0, n);

        this.c[n] = 0;
        for (let j = n - 1; j >= 0; j--) {
            this.c[j] = z[j] - mu[j] * this.c[j + 1];
            this.b[j] = (y[j + 1] - y[j]) / h[j] - (h[j] * (this.c[j + 1] + 2 * this.c[j])) / 3;
            this.d[j] = (this.c[j + 1] - this.c[j]) / (3 * h[j]);
        }
    }

    eval(val) {
        const x = this.x;
        const n = x.length - 1;

        if (val <= x[0]) return this.y[0];
        if (val >= x[n]) return this.y[n];

        // Binary search for interval
        let low = 0;
        let high = n - 1;
        let i = 0;
        while (low <= high) {
            const mid = Math.floor((low + high) / 2);
            if (x[mid] <= val && val <= x[mid + 1]) {
                i = mid;
                break;
            } else if (x[mid] > val) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }

        const dx = val - x[i];
        return this.a[i] + this.b[i] * dx + this.c[i] * dx * dx + this.d[i] * dx * dx * dx;
    }
}

// --- Array Helpers ---
function linspace(start, stop, num) {
    if (num <= 1) return [start];
    const arr = new Array(num);
    const step = (stop - start) / (num - 1);
    for (let i = 0; i < num; i++) {
        arr[i] = start + i * step;
    }
    return arr;
}

function arange(start, stop, step) {
    const arr = [];
    for (let val = start; val < stop; val += step) {
        arr.push(val);
    }
    return arr;
}

// --- Horn Generation Functions ---

/**
 * OS-SE Horn Profile Generator
 */
function generateOSSEHorn(throatRadius, length, alpha = 45, alpha0 = 0, k = 1.0, s = 0.8, q = 0.998, n = 5, numPoints = 10) {
    const alphaRad = (alpha * Math.PI) / 180;
    const alpha0Rad = (alpha0 * Math.PI) / 180;

    const z = linspace(0, length, numPoints);
    const points = [];

    const tanAlpha = Math.tan(alphaRad);
    const tanAlpha0 = Math.tan(alpha0Rad);

    for (let i = 0; i < numPoints; i++) {
        const zi = z[i];
        const term1 = Math.sqrt(
            Math.pow(k * throatRadius, 2) +
            2 * k * throatRadius * zi * tanAlpha0 +
            Math.pow(zi * tanAlpha, 2)
        );
        const term2 = throatRadius * (1 - k);
        const rGOS = term1 + term2;

        const ratio = (q * zi) / length;
        const innerPow = Math.pow(ratio, n);
        const termBracket = Math.pow(Math.max(0, 1 - innerPow), 1 / n);
        const rTERM = ((s * length) / q) * (1 - termBracket);

        const rOSSE = rGOS + rTERM;
        points.push({ x: zi, y: rOSSE });
    }

    const throatRadiusM = throatRadius / 1000.0;
    const calculatedFc = (44.0 * Math.sin(alphaRad)) / throatRadiusM;

    return {
        points,
        calculatedFc
    };
}

/**
 * Tractrix Horn Profile Generator
 */
function generateTractrixHorn(throatRadius, cutoffFreq, numPoints = 10) {
    const throatRadiusM = throatRadius / 1000.0;
    const c = 343.0; // speed of sound m/s
    const a = c / (2 * Math.PI * cutoffFreq);

    if (throatRadiusM >= a) {
        throw new Error(`Throat radius (${throatRadius}mm) must be smaller than mouth radius (${(a * 1000).toFixed(1)}mm) for cutoff ${cutoffFreq}Hz.`);
    }

    const yM = linspace(throatRadiusM, a, numPoints);
    const xM = [];

    for (let i = 0; i < numPoints; i++) {
        const yVal = yM[i];
        const sqrtTerm = Math.sqrt(Math.max(0, a * a - yVal * yVal));
        const xVal = a * Math.log((a + sqrtTerm) / yVal) - sqrtTerm;
        xM.push(xVal);
    }

    const maxX = Math.max(...xM);
    const points = [];

    for (let i = 0; i < numPoints; i++) {
        const xFlipped = Math.abs(xM[i] - maxX) * 1000;
        const yMm = yM[i] * 1000;
        points.push({ x: xFlipped, y: yMm });
    }

    // Sort by x ascending
    points.sort((p1, p2) => p1.x - p2.x);

    return { points };
}

/**
 * Spherical Horn Profile Generator
 */
function generateSphericalHorn(throatRadius, cutoffFreq, scale = 4.0, fold = false, foldBack = true) {
    const throatRadiusM = throatRadius / 1000.0;
    const scaleM = scale / 1000.0;
    const c = 343.0;
    const r0 = c / Math.PI / cutoffFreq;

    if (r0 * r0 < throatRadiusM * throatRadiusM) {
        throw new Error(`Cutoff frequency (${cutoffFreq}Hz) is too high for throat radius (${throatRadius}mm).`);
    }

    const h0 = r0 - Math.sqrt(r0 * r0 - throatRadiusM * throatRadiusM);
    const flareRate = (4 * Math.PI * cutoffFreq) / c;

    const xArr = arange(0, 1.0, scaleM);
    const rawData = [];

    for (let i = 0; i < xArr.length; i++) {
        const x = xArr[i];
        const h = h0 * Math.exp(flareRate * x);
        const xh = x - h + h0;
        const sArea = 2 * Math.PI * r0 * h;
        rawData.push({ x, h, xh, sArea });
    }

    let filtered = rawData;

    if (!fold) {
        let maxXh = -Infinity;
        let maxXAtMaxXh = 0;
        for (const item of rawData) {
            if (item.xh > maxXh) {
                maxXh = item.xh;
                maxXAtMaxXh = item.x;
            }
        }
        filtered = rawData.filter(d => d.x <= maxXAtMaxXh);
    } else {
        filtered = rawData.filter(d => (d.sArea / Math.PI - d.h * d.h) >= 0);
        if (!foldBack) {
            filtered = filtered.filter(d => d.xh >= 0);
        }
    }

    const points = filtered.map(d => {
        const yVal = Math.sqrt(Math.max(0, d.sArea / Math.PI - d.h * d.h));
        return {
            x: d.xh * 1000,
            y: yVal * 1000
        };
    });

    return { points };
}

/**
 * Exponential Horn Profile Generator
 */
function generateExponentialHorn(throatRadius, cutoffFreq, scale = 4.0) {
    const throatRadiusM = throatRadius / 1000.0;
    const scaleM = scale / 1000.0;
    const c = 343.0;

    const wavelength = c / cutoffFreq;
    const growthFactor = (4 * Math.PI) / wavelength;

    const xArr = arange(0, 1.0, scaleM);
    const points = [];

    for (let i = 0; i < xArr.length; i++) {
        const x = xArr[i];
        const sArea = Math.pow(throatRadiusM, 2) * Math.PI * Math.exp(growthFactor * x);
        const r = Math.sqrt(sArea / Math.PI);
        const cir = 2 * Math.PI * r;
        const krm = cir / wavelength;

        if (krm <= 1.0) {
            points.push({
                x: x * 1000,
                y: r * 1000,
                krm: krm
            });
        }
    }

    return { points };
}

/**
 * Hybrid Constant Directivity (HCD) Horn Generator
 */
function generateHCDHorn(originPoints, mouthRatio = 1.7, mode = 'linear', acc = 1.0) {
    if (!originPoints || originPoints.length === 0) {
        throw new Error("Origin profile is empty.");
    }

    const orgMouthRatio = mouthRatio;
    let targetMouthRatio = mouthRatio;
    if (acc > 1.0) {
        targetMouthRatio *= acc;
    }

    const xVals = originPoints.map(p => p.x);
    const maxX = Math.max(...xVals);

    // Compute transformation mouth ratio for each point
    const rawMouthRatios = originPoints.map(p => {
        const normX = maxX > 0 ? p.x / maxX : 0;
        let yTrans = 1.0;

        switch (mode) {
            case 'linear':
                yTrans = 1 + normX * (targetMouthRatio - 1);
                break;
            case 'para':
                yTrans = Math.pow(normX, 2) * (targetMouthRatio - 1) + 1;
                break;
            case 'exp':
                yTrans = Math.sqrt(normX) * (targetMouthRatio - 1) + 1;
                break;
            case 'log':
                yTrans = Math.log10(9 * normX + 1) * (targetMouthRatio - 1) + 1;
                break;
            case 'hyper':
                yTrans = (Math.sqrt(Math.pow(normX + 1, 2) - 1) / Math.sqrt(3)) * (targetMouthRatio - 1) + 1;
                break;
            case 'logistic':
                yTrans = (1 / (1 + Math.exp(11 * (0.5 - normX)))) * (targetMouthRatio - 1) + 1;
                break;
            default:
                yTrans = 1 + normX * (targetMouthRatio - 1);
        }

        return Math.min(yTrans, orgMouthRatio);
    });

    // Find first index where mouth ratio reaches orgMouthRatio and clamp remaining
    let firstMaxIdx = -1;
    for (let i = 0; i < rawMouthRatios.length; i++) {
        if (rawMouthRatios[i] >= orgMouthRatio) {
            firstMaxIdx = i;
            break;
        }
    }

    const finalMouthRatios = rawMouthRatios.map((mr, idx) => {
        if (firstMaxIdx !== -1 && idx > firstMaxIdx) {
            return orgMouthRatio;
        }
        return mr;
    });

    const hcdPoints = originPoints.map((p, idx) => {
        const area = Math.PI * Math.pow(p.y, 2);
        const mr = finalMouthRatios[idx];
        const b = Math.sqrt(area / Math.PI / mr);
        const a = b * mr;
        return {
            x: p.x,
            y: p.y,
            a: a,
            b: b,
            mouthRatio: mr
        };
    });

    return { points: hcdPoints };
}

// Export for ES modules and browser global
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        CubicSpline,
        generateOSSEHorn,
        generateTractrixHorn,
        generateSphericalHorn,
        generateExponentialHorn,
        generateHCDHorn
    };
} else {
    window.HornMath = {
        CubicSpline,
        generateOSSEHorn,
        generateTractrixHorn,
        generateSphericalHorn,
        generateExponentialHorn,
        generateHCDHorn
    };
}
