# HELIO-FORGE AI — Dashboard Blueprint
## Solar Flare Intelligence & Evolution Dashboard

> **Theme:** NASA + ISRO Mission Control  
> **Stack:** React · Next.js · Three.js · React Three Fiber · FastAPI · PyTorch  
> **Mission:** Transform raw Aditya-L1 solar observations into an interactive 3D visualization of the Sun, AI predictions, and flare evolution.

---

## Table of Contents

1. [Project Vision](#1-project-vision)
2. [Overall Data Flow Architecture](#2-overall-data-flow-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Design System](#4-design-system)
5. [Page-by-Page Specification](#5-page-by-page-specification)
   - [Page 1 — Mission Control Landing](#page-1--mission-control-landing)
   - [Page 2 — 3D Interactive Sun](#page-2--3d-interactive-sun)
   - [Page 3 — Solar Evolution Timeline](#page-3--solar-evolution-timeline)
   - [Page 4 — Live AI Prediction](#page-4--live-ai-prediction)
   - [Page 5 — Solar Intensity Visualization](#page-5--solar-intensity-visualization)
   - [Page 6 — Processing Timeline](#page-6--processing-timeline)
   - [Page 7 — Signal Analysis](#page-7--signal-analysis)
   - [Page 8 — Feature Dashboard 32 Features](#page-8--feature-dashboard-32-features)
   - [Page 9 — AI Explanation Panel](#page-9--ai-explanation-panel)
   - [Page 10 — Confidence Meter](#page-10--confidence-meter)
   - [Page 11 — Solar Risk Indicator](#page-11--solar-risk-indicator)
   - [Page 12 — Dataset Explorer FITS Upload](#page-12--dataset-explorer-fits-upload)
   - [Page 13 — Performance Metrics](#page-13--performance-metrics)
   - [Page 14 — Active Region Detection](#page-14--active-region-detection)
   - [Page 15 — Flare Evolution Animation](#page-15--flare-evolution-animation)
   - [Page 16 — Future Prediction Mode](#page-16--future-prediction-mode)
6. [Backend API Specification](#6-backend-api-specification)
7. [Frontend Project Structure](#7-frontend-project-structure)
8. [Component Architecture](#8-component-architecture)
9. [Global Layout](#9-global-layout)
10. [Deployment Architecture](#10-deployment-architecture)
11. [Implementation Roadmap](#11-implementation-roadmap)

---

## 1. Project Vision

HELIO-FORGE AI is a full-stack space weather intelligence dashboard built on the trained `HelioForgeTCN` baseline model. It makes the model's predictions tangible, explainable, and visually stunning — designed to impress both researchers and mission control operators at first glance.

**What judges see in the first 5 seconds:**
- A 3D rotating Sun with glowing active regions colour-coded by flare prediction
- Real-time AI prediction card showing class, confidence, and risk level
- NASA/ISRO mission control aesthetic — dark, cinematic, professional

**What makes it scientifically credible:**
- All predictions come from the actual trained `HelioForgeTCN` weights (`best_macro_f1.pt`, Epoch 25)
- 32 engineered features displayed with physical labels
- Confusion matrix, ROC, and training curves from the real experiment
- Temporal evolution across sequential observation windows

---

## 2. Overall Data Flow Architecture

```
New Solar Observation (FITS / .pt tensor)
              │
              ▼
  ┌───────────────────────────┐
  │  Preprocessing Pipeline   │  (Python / FastAPI)
  │  - Read raw HEL1OS/SoLEXS │
  │  - MinMax normalise        │
  │  - Build (32, 512) tensor  │
  └───────────────────────────┘
              │
              ▼
  ┌───────────────────────────┐
  │  Feature Engineering (32) │
  │  - mean, variance         │
  │  - entropy, wavelet energy│
  │  - peak count, signal E   │
  │  - rolling stats          │
  └───────────────────────────┘
              │
              ▼
  ┌───────────────────────────┐
  │  HelioForgeTCN (PyTorch)  │
  │  - TCNEncoder (8 blocks)  │
  │  - ClassifierHead         │
  │  - Output: 5 logits       │
  └───────────────────────────┘
              │
     ┌────────┼──────────────────────┐
     ▼        ▼                      ▼
 Predicted  Softmax               Feature
  Class     Probabilities         Tensor
     │        │                      │
     └────────┴──────────────────────┘
                    │
                    ▼
         FastAPI REST Response
                    │
                    ▼
         Next.js React Dashboard
         ┌────────────────────────┐
         │  3D Sun (Three.js)     │
         │  Prediction Cards      │
         │  Evolution Timeline    │
         │  Feature Cards         │
         │  Signal Chart          │
         │  Confusion Matrix      │
         └────────────────────────┘
```

---

## 3. Technology Stack

### Frontend

| Layer | Technology | Purpose |
|---|---|---|
| Framework | **Next.js 14** (App Router) | Routing, SSR, API routes |
| UI | **React 18** | Component model |
| 3D Rendering | **Three.js** + **React Three Fiber** | Interactive 3D Sun |
| 3D Extras | **@react-three/drei** | OrbitControls, stars, bloom |
| Post-processing | **@react-three/postprocessing** | Bloom, god rays, depth of field |
| Charts | **Recharts** | Signal analysis, training curves |
| Charts (scientific) | **Plotly.js** | ROC curve, confusion matrix heatmap |
| Animations | **Framer Motion** | Page transitions, card entrance |
| Styling | **Tailwind CSS** + custom CSS variables | Design tokens, dark theme |
| State | **Zustand** | Global prediction state |
| HTTP Client | **Axios** | API calls to FastAPI backend |
| Icons | **Lucide React** | Mission control UI icons |

### Backend

| Layer | Technology | Purpose |
|---|---|---|
| API | **FastAPI** | REST endpoints, async inference |
| Model | **PyTorch** (`HelioForgeTCN`) | Flare classification |
| Data | **astropy** (`fits`) | FITS file parsing |
| Preprocessing | **NumPy** + **SciPy** | Feature engineering pipeline |
| CORS | **FastAPI CORS Middleware** | Frontend-Backend communication |
| Server | **Uvicorn** | ASGI server |

### Deployment

| Layer | Technology |
|---|---|
| Containerisation | Docker + Docker Compose |
| Cloud | AWS EC2 (existing instance) |
| Reverse Proxy | Nginx |
| SSL | Let's Encrypt (Certbot) |

---

## 4. Design System

### Colour Palette — NASA / ISRO Mission Control

```css
/* Background */
--bg-space:       #010409;   /* deep space black */
--bg-panel:       #0d1117;   /* dark panel */
--bg-card:        #161b22;   /* card surface */
--bg-card-hover:  #1c2128;   /* hover state */
--bg-border:      #21262d;   /* subtle border */

/* Accent */
--accent-solar:   #f97316;   /* solar orange — primary */
--accent-gold:    #d97706;   /* secondary accent */
--accent-blue:    #3b82f6;   /* data/chart accent */
--accent-glow:    #fb923c40; /* glow overlay */

/* Flare Class Colours */
--class-quiet:    #22c55e;   /* green  — Quiet */
--class-b:        #eab308;   /* yellow — B-class */
--class-c:        #f97316;   /* orange — C-class */
--class-m:        #ef4444;   /* red    — M-class */
--class-x:        #a855f7;   /* purple — X-class */

/* Text */
--text-primary:   #e6edf3;
--text-secondary: #8b949e;
--text-muted:     #484f58;

/* Risk Indicator */
--risk-low:      #22c55e;
--risk-medium:   #eab308;
--risk-high:     #f97316;
--risk-extreme:  #ef4444;
```

### Typography

```
Google Fonts:
  Space Grotesk  → headings / titles
  JetBrains Mono → all numeric / data values
  Inter          → body text / descriptions
```

### Flare Class Visual Mapping

| Class | Label | Hex | Emoji | Risk Level | Sun Glow |
|---|---|---|---|---|---|
| 0 | **Quiet** | `#22c55e` | 🟢 | LOW | Green soft glow |
| 1 | **B-class** | `#eab308` | 🟡 | LOW | Yellow soft glow |
| 2 | **C-class** | `#f97316` | 🟠 | MEDIUM | Orange glow |
| 3 | **M-class** | `#ef4444` | 🔴 | HIGH | Red glow + pulse |
| 4 | **X-class** | `#a855f7` | 🟣 | EXTREME | Purple bloom + particles |

---

## 5. Page-by-Page Specification

---

### Page 1 — Mission Control Landing

**Route:** `/`  
**Purpose:** First impression. Judges land here.

#### Layout

```
┌─────────────────────────────────────────────────────────────┐
│  ══ HELIO-FORGE AI ══        [ISRO Logo]    [NASA Logo]     │
│  AI-Powered Solar Flare Intelligence System                  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   [   3D ROTATING SUN    ]   │  📦 Observation              │
│                              │     OBS_20260728_0031         │
│   [  bloom + particle    ]   │                              │
│   [  corona glow effect  ]   │  ⏱ Window: 512 timesteps     │
│                              │                              │
│                              │  🔮 Prediction: M-Class      │
│                              │                              │
│                              │  📊 Confidence: 87%          │
│                              │                              │
│                              │  🟠 Solar Risk: HIGH         │
│                              │                              │
│                              │  ⚡ Processing: 78 ms        │
├──────────────────────────────────────────────────────────────┤
│  [Signal]  [Evolution]  [Features]  [Performance]  [Upload]  │
└──────────────────────────────────────────────────────────────┘
```

#### Key Elements

- **Background:** Animated star field (`THREE.Points`, procedural positions)
- **3D Sun:** Auto-rotating (`rotation.y += 0.003` per frame via `useFrame`)
- **Info Cards:** Glassmorphism — `backdrop-filter: blur(12px)`, semi-transparent borders
- **Nav Bar:** Sticky top, glowing HELIO-FORGE logo with subtle flicker animation
- **Scanline effect:** Thin CRT scanline CSS overlay for mission control atmosphere
- **Live UTC clock:** Top-right corner, updates every second

#### Animations

- Hero Sun enters with scale `0 → 1`, 1.2s ease-out
- Info cards slide in from right with stagger delay (0, 100ms, 200ms, 300ms…)
- Confidence bar animates from `0% → 87%` on load
- Risk indicator pulses every 2s when class is M or X

---

### Page 2 — 3D Interactive Sun

**Route:** `/sun`  
**Purpose:** The centrepiece — interactive 3D Sun with clickable active regions.

#### Three.js Scene Setup

```jsx
<Canvas camera={{ position: [0, 0, 3], fov: 45 }}>
  <Stars radius={100} depth={50} count={5000} factor={4} />
  <ambientLight intensity={0.1} />
  <pointLight position={[0, 0, 0]} intensity={2} color="#ff6a00" />
  <Sun />
  <ActiveRegions regions={regions} onSelect={onRegionClick} />
  <EffectComposer>
    <Bloom luminanceThreshold={0.2} intensity={1.5} />
    <GodRays sun={sunRef} />
  </EffectComposer>
  <OrbitControls enableZoom autoRotate autoRotateSpeed={0.4} />
</Canvas>
```

#### Sun Sphere Materials

```jsx
<mesh ref={sunRef}>
  <sphereGeometry args={[1, 64, 64]} />
  <meshStandardMaterial
    map={solarTexture}          // NASA SDO HMI continuum (public domain)
    normalMap={normalMap}
    emissiveMap={solarTexture}
    emissive="#ff4500"
    emissiveIntensity={0.3}
  />
</mesh>
```

#### Active Regions

Each active region is a clickable 3D marker placed in spherical coordinates:

```jsx
// lat/lon → Cartesian on the Sun surface (radius = 1.05)
<mesh
  position={sphericalToCartesian(lat, lon, 1.05)}
  onClick={() => onRegionClick(region)}
>
  <sphereGeometry args={[0.04, 16, 16]} />
  <meshStandardMaterial
    color={CLASS_COLORS[region.predicted_class]}
    emissive={CLASS_COLORS[region.predicted_class]}
    emissiveIntensity={1.5}
  />
</mesh>
```

A pulsing ring animates around the highest-severity region (expanding, fading loop).

#### Click Popup — Region Info Card

```
┌──────────────────────────────┐
│  Active Region #3            │
│  ─────────────────────────── │
│  Latitude:        -12.3°     │
│  Longitude:       +47.8°     │
│  Predicted Class: M-Class 🔴 │
│  Confidence:      84%        │
│  Intensity:       7,320 c/s  │
│  Observation:     12:32 UTC  │
└──────────────────────────────┘
```

#### Controls Panel (top-right overlay)

- Toggle: Active Regions on/off
- Toggle: Corona Layer visibility
- Slider: Rotation Speed
- Dropdown: Texture (Optical / AIA 171Å / AIA 304Å)

---

### Page 3 — Solar Evolution Timeline

**Route:** `/evolution`  
**Purpose:** Show how solar activity evolves across sequential observation windows.

#### Layout

```
t - 4 hrs        t - 3 hrs        t - 2 hrs        t - 1 hr         CURRENT
    ☀                ☀                ☀                ☀                ☀
[Quiet 🟢]      [B-class 🟡]     [B-class 🟡]     [C-class 🟠]    [M-class 🔴]
Conf: 91%        Conf: 78%        Conf: 82%        Conf: 73%        Conf: 87%

──────────────────────────── Time ───────────────────────────────────────────▶
```

#### Components

- **5 mini 3D Suns** rendered side by side, each with its class-specific glow colour
- Arrow connectors between frames with animated particle flow
- **Active frame** has bright border glow and is slightly enlarged
- Clicking any frame loads that observation's full details into all other pages
- **Auto-play mode:** Cycles forward through frames at 1.5s per step

#### Backend Data Required

```json
{
  "evolution_sequence": [
    { "timestamp": "2026-07-28T08:00Z", "class": 0, "label": "Quiet", "confidence": 0.91 },
    { "timestamp": "2026-07-28T09:00Z", "class": 1, "label": "B",     "confidence": 0.78 },
    { "timestamp": "2026-07-28T10:00Z", "class": 1, "label": "B",     "confidence": 0.82 },
    { "timestamp": "2026-07-28T11:00Z", "class": 2, "label": "C",     "confidence": 0.73 },
    { "timestamp": "2026-07-28T12:00Z", "class": 3, "label": "M",     "confidence": 0.87 }
  ]
}
```

---

### Page 4 — Live AI Prediction

**Route:** `/prediction`  
**Purpose:** Full detail view of the model's output — probabilities for all 5 classes.

#### Layout

```
┌─────────────────────────────────────────────────────────┐
│  AI Prediction Result                                   │
│                                                         │
│  Predicted Class:   M-Class  🔴                         │
│  Confidence:        87%                                 │
│                                                         │
│  ─── Class Probability Distribution ─────────────────── │
│                                                         │
│  Quiet  ░░░░░░░░░░░░░░░░░░  0.00                        │
│  B      ▓░░░░░░░░░░░░░░░░░  0.03                        │
│  C      ▓▓▓░░░░░░░░░░░░░░░  0.09                        │
│  M      ▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░  0.84  ← PREDICTED           │
│  X      ▓░░░░░░░░░░░░░░░░░  0.04                        │
│                                                         │
│  ─── Radar Chart ────────────────────────────────────── │
│  [Pentagon radar — 5 axes = 5 class probabilities]      │
└─────────────────────────────────────────────────────────┘
```

#### Visualisation Tabs

1. **Horizontal Bars** — probability per class with class colour coding
2. **Radar Chart** — 5-axis radar showing class probability distribution
3. **Donut Chart** — proportional breakdown of all 5 classes

#### API Response Shape

```json
{
  "predicted_class": 3,
  "predicted_label": "M",
  "confidence": 0.87,
  "probabilities": {
    "Quiet": 0.00,
    "B":     0.03,
    "C":     0.09,
    "M":     0.84,
    "X":     0.04
  }
}
```

---

### Page 5 — Solar Intensity Visualization

**Route:** `/intensity`  
**Purpose:** Display HEL1OS RGB intensity channels and a composite RGB Sun.

#### Layout

```
┌────────────────────────────────────────────────────────────┐
│  Solar Intensity — Composite RGB View                      │
├──────────────────────────────────────┬─────────────────────┤
│  Channel Analysis                    │  Composite Sun      │
│                                      │                     │
│  🔴 Red   (Hard X-ray)               │   [Three.js sphere] │
│  ████████████████░░░░  182 c/s       │  Emissive colour    │
│                                      │  computed from RGB  │
│  🟢 Green (Soft X-ray)               │  channel values     │
│  █████████████░░░░░░░  147 c/s       │                     │
│                                      │  Updates live when  │
│  🔵 Blue  (UV channel)               │  new obs is loaded  │
│  ██████████░░░░░░░░░░  103 c/s       │                     │
└──────────────────────────────────────┴─────────────────────┘
```

#### Composite RGB Sun

- Three.js sphere whose `emissive` colour is computed from the 3 channel intensities
- Raw counts → normalised RGB `[0, 255]` via MinMax scaling from training bounds
- `colour.lerp()` used for smooth transitions between observations

#### Channel Trend Lines

- Small Recharts sparkline per channel (512 timesteps)
- Shaded area under curve, prediction window highlighted with semi-transparent overlay
- Hover tooltip: exact value + timestep index

---

### Page 6 — Processing Timeline

**Route:** Embedded component (sidebar panel or step-by-step modal)  
**Purpose:** Show judges the full AI pipeline — each stage glows when active.

#### Visual

```
  ① Observation Loaded      ✅  12 ms
       │
  ② FITS Parsing            ✅   8 ms
       │
  ③ Preprocessing           ✅  15 ms
       │
  ④ Feature Engineering     ✅  22 ms
       │
  ⑤ TCN Inference           ⚙   78 ms   ← currently processing (pulsing)
       │
  ⑥ Softmax + Class         ──
       │
  ⑦ Result Displayed        ──
```

#### Behaviour

- Each stage animates sequentially, 200ms delay between steps
- Active stage: orange pulsing glow
- Completed stages: green with checkmark
- Failed stages: red with error message and hint text
- Total elapsed time shown at the bottom

---

### Page 7 — Signal Analysis

**Route:** `/signals`  
**Purpose:** Show the raw HEL1OS/SoLEXS time-series signal with the prediction window highlighted.

#### Layout

```
┌────────────────────────────────────────────────────────────┐
│  Raw Signal — Soft X-ray Intensity (SoLEXS Channel 0)     │
│                                                            │
│  Intensity │                          /\                   │
│  (c/s)     │                        /    \                 │
│            │          /\           /       \               │
│            │         /   \        /                        │
│            │________/     \______/___________________      │
│            └─────────────────────────────────────────      │
│            0        128       256       384      512        │
│                               Timestep                     │
│                                                            │
│  ████████████████████  ← Prediction Window (512 steps)     │
│                                                            │
│  ── Channel ──────────  [Soft X-ray ▾]                     │
└────────────────────────────────────────────────────────────┘
```

#### Features

- Channel selector dropdown (all 32 features available)
- Prediction window highlighted with semi-transparent orange `ReferenceArea`
- Peak markers as vertical dashed `ReferenceLine` elements
- Zoom in/out via mouse wheel
- Hover tooltip: exact value + timestep index
- Recharts `ComposedChart` with `Line` + `Area`

---

### Page 8 — Feature Dashboard 32 Features

**Route:** `/features`  
**Purpose:** Display all 32 engineered features as cards. Green = normal, red = anomalous.

#### Layout — 8 × 4 Grid

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  soft_mean  │  │  soft_std   │  │  entropy    │  │  wavelet_L3 │
│   182.4     │  │   51.2      │  │   3.72      │  │   91.2      │
│  c/s  🟢   │  │  units 🔴   │  │  bits  🟢   │  │  J     🟠  │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
   ... (all 32 features) ...
```

#### Card Status Logic

| Status | Condition | Colour |
|---|---|---|
| 🟢 Normal | Value within 1σ of training mean | Green border |
| 🟡 Elevated | Value between 1σ – 2σ | Yellow border |
| 🔴 Anomalous | Value beyond 2σ | Red border + glow |

Thresholds from `scaler_f32_w512.json` training statistics.

#### Interactivity

- Click any card → expands to show sparkline of that feature across 512 timesteps
- Sort by: Name / Value / Status
- Filter: All / Normal / Anomalous

#### Full 32 Feature Reference

| # | Feature Name | Physical Meaning |
|---|---|---|
| 0 | `soft_mean` | Mean soft X-ray count rate |
| 1 | `soft_std` | Standard deviation of soft X-ray |
| 2 | `soft_max` | Peak soft X-ray count |
| 3 | `soft_min` | Minimum soft X-ray count |
| 4 | `soft_range` | Dynamic range (max − min) |
| 5 | `soft_skew` | Signal asymmetry |
| 6 | `soft_kurtosis` | Peak sharpness / impulsiveness |
| 7 | `soft_energy` | Total integrated energy |
| 8 | `soft_entropy` | Shannon entropy of distribution |
| 9 | `soft_peak_count` | Number of detected peaks |
| 10 | `hard_mean` | Mean hard X-ray count rate |
| 11 | `hard_std` | Hard X-ray variability |
| 12 | `hard_max` | Peak hard X-ray count |
| 13 | `hard_energy` | Hard X-ray integrated energy |
| 14 | `ratio_hard_soft` | Hard/soft ratio (flare hardness) |
| 15 | `roll_mean_16` | 16-step rolling mean |
| 16 | `roll_std_16` | 16-step rolling std |
| 17 | `roll_mean_64` | 64-step rolling mean |
| 18 | `roll_std_64` | 64-step rolling std |
| 19 | `wavelet_energy_L1` | Wavelet energy level 1 (fine scale) |
| 20 | `wavelet_energy_L2` | Wavelet energy level 2 |
| 21 | `wavelet_energy_L3` | Wavelet energy level 3 |
| 22 | `wavelet_energy_L4` | Wavelet energy level 4 (coarse) |
| 23 | `spectral_entropy` | Frequency-domain entropy |
| 24 | `dominant_freq` | Dominant oscillation frequency |
| 25 | `rise_rate` | Rate of intensity increase |
| 26 | `decay_rate` | Rate of intensity decay |
| 27 | `delta_mean` | Change in mean across window halves |
| 28 | `log_energy` | Log-transformed total energy |
| 29 | `zero_crossing_rate` | Signal oscillation rate |
| 30 | `temporal_gradient` | First derivative mean |
| 31 | `channel_correlation` | HEL1OS ↔ SoLEXS correlation |

---

### Page 9 — AI Explanation Panel

**Route:** Embedded on landing + prediction pages  
**Purpose:** Make the model interpretable — show *why* it predicted what it did.

#### Layout

```
┌────────────────────────────────────────────────────────────┐
│  AI Explanation — Why M-Class?                            │
│  ────────────────────────────────────────────────────────  │
│  🔴 High wavelet energy at Level 3                        │
│     wavelet_L3 = 91.2  (threshold: > 80 for M-class)     │
│                                                            │
│  🔴 Increasing spectral entropy                           │
│     spectral_entropy = 3.72  (elevated — broadband        │
│     energy injection pattern)                             │
│                                                            │
│  🔴 Rapid energy growth                                   │
│     rise_rate = +1,240 c/s per timestep                   │
│                                                            │
│  🔴 High rolling variance                                 │
│     roll_std_64 = 51.2  (95th percentile of training)     │
│                                                            │
│  🟡 Peak count elevated                                    │
│     29 peaks detected — above B-class baseline            │
│  ────────────────────────────────────────────────────────  │
│  ℹ Future: SHAP value integration planned                 │
└────────────────────────────────────────────────────────────┘
```

#### Phase 1 Implementation — Rule-Based

Reasons generated from threshold rules in `explanation_rules.json`:

```json
{
  "rules": [
    {
      "feature": "wavelet_energy_L3",
      "class": "M",
      "condition": "> 80",
      "severity": "high",
      "reason": "High wavelet energy at Level 3"
    }
  ]
}
```

#### Phase 2 — SHAP Integration

```python
import shap
explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(X_instance)
# Top 5 features by |SHAP| → displayed as explanation reasons
```

---

### Page 10 — Confidence Meter

**Route:** Embedded widget on all primary pages  
**Purpose:** Instantly communicate model certainty.

#### Visual

```
AI Confidence

▓▓▓▓▓▓▓▓▓▓▓▓▓░░  87%

 Low       Med       High
  └─────────────────┘
           87%
```

#### Implementation

- Animated `<progress>` bar with CSS gradient fill
- Colour interpolation: green (< 60%) → yellow (60–80%) → orange (80–90%) → red (> 90%)
- Framer Motion counter animation: 0 → 87% on page load
- Tooltip: "Model is 87% confident this is an M-class event."

---

### Page 11 — Solar Risk Indicator

**Route:** Persistent sidebar widget on all pages  
**Purpose:** Most important operational single metric — visible at all times.

#### Visual

```
Solar Risk Level

  ● LOW
  ● MEDIUM
  ▶ HIGH         ← current (glowing + pulsing)
  ● EXTREME
```

#### Risk Class Mapping

| Risk Level | Triggered By | Colour |
|---|---|---|
| **LOW** | Quiet, B | `#22c55e` |
| **MEDIUM** | C | `#eab308` |
| **HIGH** | M | `#f97316` |
| **EXTREME** | X | `#ef4444` |

#### Behaviour

- M and X: risk indicator **pulses** every 1.5s using CSS keyframe animation
- Level transitions are animated with 200ms crossfade
- Shared via Zustand global store — updates instantly on any new prediction

---

### Page 12 — Dataset Explorer FITS Upload

**Route:** `/upload`  
**Purpose:** Allow users to upload a new FITS file and see live AI predictions.

#### Layout

```
┌────────────────────────────────────────────────────────────┐
│  Dataset Explorer                                          │
│  ────────────────────────────────────────────────────────  │
│                                                            │
│    [  Drag & Drop  FITS / .pt  file here  ]               │
│    [  or click to browse                  ]               │
│                                                            │
│  ────────────────────────────────────────────────────────  │
│  Processing Pipeline:                                      │
│  ✅  File loaded:       new_observation.fits               │
│  ⚙   Reading FITS...                         (animated)   │
│  ⚙   Extracting features...                  (animated)   │
│  ⚙   Running TCN inference...                (animated)   │
│  ✅  Prediction ready!                                     │
│  ────────────────────────────────────────────────────────  │
│  Result:  M-Class    Confidence: 87%    Risk: 🟠 HIGH     │
└────────────────────────────────────────────────────────────┘
```

#### Backend Endpoint

```
POST /api/predict
Content-Type: multipart/form-data
Body: { file: <FITS file> }

Response: { predicted_class, predicted_label, confidence,
            probabilities, features, processing_time_ms, signal, ... }
```

#### Live Updates via SSE

```
GET /api/stream/predict

event: pipeline_step
data: { "step": "FITS Parsing", "status": "running", "elapsed_ms": 8 }

event: pipeline_step
data: { "step": "Feature Engineering", "status": "done", "elapsed_ms": 22 }
```

---

### Page 13 — Performance Metrics

**Route:** `/performance`  
**Purpose:** Show research results — reproducible and audit-able.

#### Sub-sections

**13.1 Summary Cards**

```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Accuracy │  │ Macro F1 │  │  Recall  │  │ Precision│
│  89.41%  │  │  0.8514  │  │  0.8698  │  │  0.8488  │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
```

**13.2 Per-Class Table**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Quiet | 0.9485 | 1.0000 | 0.9735 | 92 |
| B | 0.8716 | 0.9627 | 0.9149 | 134 |
| C | 0.9775 | 0.7982 | 0.8788 | 109 |
| M | 0.8696 | 0.7547 | 0.8081 | 53 |
| X | 0.5769 | 0.8333 | 0.6818 | 18 |

**13.3 Confusion Matrix** — Plotly annotated heatmap (5×5, white → red scale)

**13.4 Training Curves** — Recharts dual-axis: train loss + val loss + Macro F1, LR reduction markers at Epochs 10 and 18, best checkpoint star at Epoch 25

**13.5 ROC Curves** — Plotly one-vs-rest curves per class with AUC annotation

**13.6 Precision-Recall Curves** — Plotly, AP per class annotated

---

### Page 14 — Active Region Detection

**Route:** Sidebar panel on `/sun` page  
**Purpose:** Enumerate all detected active regions with properties.

#### Layout

```
┌────────────────────────────────────────────────────────────┐
│  Active Regions  (5 detected)                              │
│  ────────────────────────────────────────────────────────  │
│  AR-001   Lat: +12.3°   Lon: -45.1°   C-class 🟠  81%    │
│  AR-002   Lat: -22.7°   Lon: +12.4°   B-class 🟡  76%    │
│  AR-003   Lat:  -5.1°   Lon: +67.3°   M-class 🔴  84%  ● │
│  AR-004   Lat: +30.0°   Lon: -78.2°   B-class 🟡  69%    │
│  AR-005   Lat:  +8.4°   Lon: +33.1°   C-class 🟠  72%    │
│  ────────────────────────────────────────────────────────  │
│  ● = Highest risk region (pulsing on 3D Sun)              │
└────────────────────────────────────────────────────────────┘
```

#### On Region Click

- 3D Sun camera auto-rotates to face the selected region
- Region info popup appears
- Signal chart updates to show that region's time-series
- Feature cards update to show that region's 32 features

---

### Page 15 — Flare Evolution Animation

**Route:** `/animation`  
**Purpose:** The most visually impressive page — animated flare lifecycle from Quiet to X.

#### Animation Sequence

```
STATE 1: Quiet
  Sun colour:   dim green glow
  Corona:       none
  Particles:    0

STATE 2: B-class
  Sun colour:   soft yellow glow
  Corona:       faint ring
  Particles:    10

STATE 3: C-class
  Sun colour:   orange glow
  Corona:       visible ring, expanding
  Particles:    50

STATE 4: M-class
  Sun colour:   red glow + bloom
  Corona:       expanding prominently
  Particles:    200  (CME begins)

STATE 5: X-class
  Sun colour:   purple/white bloom
  Corona:       full CME ejection
  Particles:    800  (streaming outward)
  Extras:       screen flash, geomagnetic storm ring
```

#### Per-State Configuration

```typescript
const FLARE_STATES = [
  { class: 'Quiet', color: '#22c55e', bloom: 0.3, particles: 0,   corona: 1.0 },
  { class: 'B',     color: '#eab308', bloom: 0.5, particles: 10,  corona: 1.1 },
  { class: 'C',     color: '#f97316', bloom: 0.8, particles: 50,  corona: 1.3 },
  { class: 'M',     color: '#ef4444', bloom: 1.5, particles: 200, corona: 1.6 },
  { class: 'X',     color: '#a855f7', bloom: 3.0, particles: 800, corona: 2.5 },
]
```

Particles use `THREE.Points` — initialised at Sun surface, velocity radially outward, opacity fades over distance.

#### Controls

- **Play / Pause** button
- **Speed** slider: 0.5× / 1× / 2×
- **Jump to class** buttons: Quiet | B | C | M | X
- **Loop** toggle

---

### Page 16 — Future Prediction Mode

**Route:** `/forecast`  
**Purpose:** Show predicted flare trajectory. Placeholder in Phase 1; backend forecasting added in Phase 2.

#### Layout

```
┌────────────────────────────────────────────────────────────┐
│  Solar Flare Trajectory Forecast                           │
│  ────────────────────────────────────────────────────────  │
│                                                            │
│  Current:    C-class 🟠  (now)                            │
│       │                                                    │
│       ▼  ─── 30 min ──▶   M-class 🔴                      │
│       │                                                    │
│       ▼  ─── 60 min ──▶   X-class 🟣   ← ⚠ ALERT         │
│                                                            │
│  [Animated glowing arrow trajectory]                      │
│                                                            │
│  ⚠ WARNING: X-class flare predicted in ~60 minutes        │
│     Trajectory confidence: 71%                            │
│                                                            │
│  Recommended Actions:                                      │
│  → Switch sensitive instruments to safe mode              │
│  → Alert ground station teams                             │
└────────────────────────────────────────────────────────────┘
```

> **Phase 1 Status:** Static UI with mock data.  
> **Phase 2:** Real multi-step forecasting model output.

---

## 6. Backend API Specification

### Base URL
```
http://<ec2-instance-ip>:8000/api
```

### Endpoints

#### `GET /api/health`
```json
{
  "status": "ok",
  "model": "HelioForgeTCN",
  "checkpoint": "best_macro_f1.pt",
  "epoch": 25,
  "macro_f1": 0.8514
}
```

#### `POST /api/predict`
```
Request:  multipart/form-data { file: <FITS or .pt> }

Response:
{
  "observation_id":     "OBS_20260728_0031",
  "predicted_class":    3,
  "predicted_label":    "M",
  "confidence":         0.87,
  "risk_level":         "HIGH",
  "probabilities":      { "Quiet": 0.00, "B": 0.03, "C": 0.09, "M": 0.84, "X": 0.04 },
  "features":           { "soft_mean": 182.4, "soft_std": 51.2, ... },
  "active_regions":     [ { "id": "AR-003", "lat": -5.1, "lon": 67.3, "class": 3 } ],
  "rgb_intensity":      { "red": 182, "green": 147, "blue": 103 },
  "processing_time_ms": 78,
  "signal":             [0.12, 0.15, ...]  // 512-length normalised array
}
```

#### `GET /api/demo`
Returns a pre-computed demo observation (no file needed). Same schema as `/api/predict`.

#### `GET /api/evolution`
```json
{
  "sequence": [
    { "timestamp": "...", "class": 0, "label": "Quiet", "confidence": 0.91 },
    { "timestamp": "...", "class": 1, "label": "B",     "confidence": 0.78 },
    { "timestamp": "...", "class": 1, "label": "B",     "confidence": 0.82 },
    { "timestamp": "...", "class": 2, "label": "C",     "confidence": 0.73 },
    { "timestamp": "...", "class": 3, "label": "M",     "confidence": 0.87 }
  ]
}
```

#### `GET /api/performance`
```json
{
  "accuracy": 0.8941,
  "macro_f1": 0.8514,
  "macro_precision": 0.8488,
  "macro_recall": 0.8698,
  "per_class": {
    "Quiet": { "precision": 0.9485, "recall": 1.0000, "f1": 0.9735, "support": 92 },
    "B":     { "precision": 0.8716, "recall": 0.9627, "f1": 0.9149, "support": 134 },
    "C":     { "precision": 0.9775, "recall": 0.7982, "f1": 0.8788, "support": 109 },
    "M":     { "precision": 0.8696, "recall": 0.7547, "f1": 0.8081, "support": 53 },
    "X":     { "precision": 0.5769, "recall": 0.8333, "f1": 0.6818, "support": 18 }
  },
  "confusion_matrix": [[92,0,0,0,0],[5,129,0,0,0],[0,19,87,3,0],[0,0,2,40,11],[0,0,0,3,15]],
  "training_history": [
    { "epoch": 1,  "train_loss": 1.4162, "val_loss": 2.6070, "val_f1": 0.5912, "lr": 1e-3 },
    { "epoch": 19, "train_loss": 0.0672, "val_loss": 0.9214, "val_f1": 0.8164, "lr": 2.5e-4 },
    { "epoch": 25, "train_loss": null,   "val_loss": 0.8234, "val_f1": 0.8714, "lr": 2.5e-4 }
  ]
}
```

#### `GET /api/stream/predict` — Server-Sent Events
```
event: pipeline_step
data: { "step": "FITS Parsing",        "status": "running", "elapsed_ms": 8  }

event: pipeline_step
data: { "step": "Feature Engineering", "status": "done",    "elapsed_ms": 22 }

event: pipeline_step
data: { "step": "TCN Inference",       "status": "done",    "elapsed_ms": 78 }

event: complete
data: { "prediction": { ... } }
```

---

### 6.1 Backend Implementation Bridge (Code Reuse Map)

To avoid duplicating logic, the FastAPI backend directly imports and reuses existing Python modules:

| Pipeline Stage | Existing Codebase Component | Backend Integration (`backend/`) |
|---|---|---|
| **FITS Ingestion & Preprocessing** | `src.pipeline.preprocessing.hel1os.process_hel1os`<br>`src.pipeline.preprocessing.solexs.process_solexs`<br>`src.pipeline.preprocessing.synchronization` | `backend/preprocessing.py`<br>Executes `process_hel1os()` & `process_solexs()` on uploaded FITS byte streams. |
| **Feature Extraction (32 Features)** | `src.features.rolling_feature_extractor.RollingFeatureExtractor`<br>`src.features.feature_selector.FeatureSelector` | `backend/preprocessing.py`<br>Extracts 32 rolling-window features per timestep from raw Soft/Hard X-ray signals. |
| **Tensor Construction ($W_t$)** | `src.pipeline.ingestion.multivariate_feature_window_generator.MultivariateFeatureWindowGenerator` | `backend/preprocessing.py`<br>Constructs `(1, 32, 512)` PyTorch input tensors normalized using `scaler_f32_w512.json`. |
| **TCN Model Inference** | `src.HPINA.models.baseline_tcn.model.HelioForgeTCN`<br>`src.HPINA.models.baseline_tcn.metrics.evaluate` | `backend/inference.py`<br>Loads `best_macro_f1.pt` weights and executes model forward pass `logits = model(x)`. |
| **Evaluation Metrics & Performance** | `docs/model/TEST_EVALUATION_REPORT.md`<br>`src.HPINA.models.baseline_tcn.metrics` | `backend/main.py`<br>Serves pre-computed metrics and confusion matrix for `/api/performance`. |

```
                 UPLOAD: new_observation.fits
                              │
                              ▼
               ┌──────────────────────────────┐
               │   FastAPI POST /api/predict  │
               └──────────────┬───────────────┘
                              │
                              ▼
 ┌──────────────────────────────────────────────────────────┐
 │ backend/preprocessing.py                                 │
 │  1. src.pipeline.preprocessing (read FITS signal)        │
 │  2. src.features.RollingFeatureExtractor (32 features)   │
 │  3. MinMax scaler transform (scaler_f32_w512.json)       │
 │  4. Build torch.Tensor of shape (1, 32, 512)             │
 └────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
 ┌──────────────────────────────────────────────────────────┐
 │ backend/inference.py                                     │
 │  1. Load src.HPINA.models.baseline_tcn.HelioForgeTCN    │
 │  2. Load best_macro_f1.pt checkpoint                     │
 │  3. Output: raw logits → torch.softmax(logits, dim=1)    │
 └────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
                 JSON Response → React Dashboard
```

---

## 7. Frontend Project Structure

```
helioforge-dashboard/
├── app/                            # Next.js App Router
│   ├── layout.tsx                  # Root layout (navbar + persistent sidebar)
│   ├── page.tsx                    # Mission Control landing page
│   ├── sun/page.tsx                # 3D Interactive Sun
│   ├── evolution/page.tsx          # Solar Evolution Timeline
│   ├── prediction/page.tsx         # Live AI Prediction
│   ├── intensity/page.tsx          # Solar Intensity Visualization
│   ├── signals/page.tsx            # Signal Analysis
│   ├── features/page.tsx           # Feature Dashboard (32 features)
│   ├── upload/page.tsx             # Dataset Explorer / FITS Upload
│   ├── performance/page.tsx        # Performance Metrics
│   ├── animation/page.tsx          # Flare Evolution Animation
│   └── forecast/page.tsx           # Future Prediction Mode
│
├── components/
│   ├── three/                      # Three.js / React Three Fiber
│   │   ├── SceneCanvas.tsx         # Root R3F <Canvas> wrapper
│   │   ├── SunSphere.tsx           # Solar sphere with texture + emissive
│   │   ├── ActiveRegion.tsx        # Clickable region marker
│   │   ├── StarField.tsx           # Background star particles
│   │   ├── CoronaRing.tsx          # Corona glow geometry
│   │   └── ParticleSystem.tsx      # CME ejection particles
│   │
│   ├── ui/                         # Reusable UI widgets
│   │   ├── GlowCard.tsx            # Glassmorphism card wrapper
│   │   ├── PredictionCard.tsx      # Class + confidence display
│   │   ├── RiskIndicator.tsx       # LOW / MEDIUM / HIGH / EXTREME
│   │   ├── ConfidenceMeter.tsx     # Animated progress bar
│   │   ├── ClassBadge.tsx          # Colour-coded class pill
│   │   ├── FeatureCard.tsx         # Single feature display card
│   │   └── ProcessingTimeline.tsx  # Step-by-step pipeline visual
│   │
│   ├── charts/                     # Chart components
│   │   ├── SignalChart.tsx          # Recharts line + area chart
│   │   ├── ProbabilityBars.tsx     # Class probability bars
│   │   ├── RadarChart.tsx          # 5-class radar
│   │   ├── ConfusionMatrix.tsx     # Plotly 5×5 heatmap
│   │   ├── TrainingCurve.tsx       # Loss + F1 over epochs
│   │   ├── ROCCurve.tsx            # Plotly ROC per class
│   │   └── PRCurve.tsx             # Precision-Recall per class
│   │
│   ├── layout/
│   │   ├── Navbar.tsx
│   │   ├── Sidebar.tsx             # Persistent risk + confidence sidebar
│   │   └── Footer.tsx
│   │
│   └── evolution/
│       ├── EvolutionTimeline.tsx   # 5-frame evolution strip
│       └── MiniSun.tsx             # Individual mini 3D Sun
│
├── store/
│   └── usePredictionStore.ts       # Zustand global state
│
├── hooks/
│   ├── usePrediction.ts            # API call + state management
│   ├── useSSE.ts                   # Server-Sent Events subscription hook
│   └── useAnimationLoop.ts         # Flare animation state controller
│
├── lib/
│   ├── api.ts                      # Axios client configuration
│   ├── constants.ts                # CLASS_COLORS, RISK_LEVELS, MODEL_INFO
│   └── utils.ts                    # sphericalToCartesian, formatConfidence, etc.
│
├── public/
│   └── textures/
│       ├── sun_texture.jpg         # NASA SDO HMI continuum (public domain)
│       ├── sun_normal.jpg          # Normal map for surface detail
│       └── sun_specular.jpg        # Specular map for limb brightening
│
└── styles/
    └── globals.css                 # CSS custom properties + base styles
```

---

## 8. Component Architecture

### Global State — Zustand

```typescript
// store/usePredictionStore.ts
interface PredictionState {
  // Current observation data
  observation_id:    string;
  predicted_class:   number;         // 0–4
  predicted_label:   string;         // "Quiet" | "B" | "C" | "M" | "X"
  confidence:        number;         // 0.0–1.0
  risk_level:        string;         // "LOW" | "MEDIUM" | "HIGH" | "EXTREME"
  probabilities:     Record<string, number>;
  features:          Record<string, number>;  // all 32 features
  active_regions:    ActiveRegion[];
  rgb_intensity:     { red: number; green: number; blue: number };
  signal:            number[];       // 512-length normalised signal
  processing_time_ms: number;

  // UI state
  isLoading: boolean;
  error:     string | null;

  // Actions
  setPrediction: (data: PredictionResponse) => void;
  fetchDemo:     () => Promise<void>;
  predict:       (file: File) => Promise<void>;
}
```

### ClassBadge Component

```typescript
const CLASS_CONFIG = {
  Quiet: { color: '#22c55e', emoji: '🟢', bg: '#052e16' },
  B:     { color: '#eab308', emoji: '🟡', bg: '#1c1a04' },
  C:     { color: '#f97316', emoji: '🟠', bg: '#1c0a00' },
  M:     { color: '#ef4444', emoji: '🔴', bg: '#1c0000' },
  X:     { color: '#a855f7', emoji: '🟣', bg: '#0f0018' },
} as const;
```

---

## 9. Global Layout

```
┌───────────────────────────────────────────────────────────────────────┐
│  ☀ HELIO-FORGE AI   [Control][Sun][Evolution][Perf][Upload]     UTC   │  ← Navbar
├───────────────────────────────────────────────────────────┬───────────┤
│                                                           │  Risk     │
│                                                           │  🟠 HIGH  │
│                                                           │           │
│              Main Content Area                            │  Conf     │
│           (Route-specific page)                           │  87%      │
│                                                           │           │
│                                                           │  Class    │
│                                                           │  M-class  │
│                                                           │           │
│                                                           │ [Upload]  │
└───────────────────────────────────────────────────────────┴───────────┘
```

The right sidebar (`RiskIndicator`, `ConfidenceMeter`, `ClassBadge`) persists across all routes and reacts to the global Zustand store — updates instantly on every new prediction.

---

## 10. Deployment Architecture

```
Internet
    │
    ▼
  Nginx  (port 80 / 443)
  ├── /        → Next.js frontend  (port 3000)
  └── /api/*   → FastAPI backend   (port 8000)

Docker Compose — AWS EC2
  services:
    frontend:
      build: ./helioforge-dashboard
      ports: ["3000:3000"]
      env: NEXT_PUBLIC_API_URL=http://localhost:8000/api

    backend:
      build: ./backend
      ports: ["8000:8000"]
      volumes:
        - /opt/helioforge-ai/experiments:/app/experiments   # model weights
        - /opt/helioforge-ai/data:/app/data                 # scaler JSON

    nginx:
      image: nginx:alpine
      ports: ["80:80", "443:443"]
      depends_on: [frontend, backend]
```

### Backend Directory Structure

```
backend/
├── main.py             # FastAPI app + CORS + routes
├── inference.py        # HelioForgeTCN loader + predict()
├── preprocessing.py    # FITS → (32, 512) tensor pipeline
├── explanation.py      # Rule-based AI explanation engine
├── requirements.txt    # torch, fastapi, uvicorn, astropy, numpy, scipy
└── Dockerfile
```

---

## 11. Implementation Roadmap

### Phase 1 — Core Dashboard (Week 1–2)

- [ ] Set up Next.js 14 + React Three Fiber project
- [ ] Design system: CSS variables, fonts, glassmorphism cards, scanline overlay
- [ ] Static 3D Sun scene: texture, bloom post-processing, auto-rotation, star field
- [ ] Mission Control landing page (Page 1) — full layout
- [ ] FastAPI backend: `/api/demo` + `/api/predict` + `/api/health`
- [ ] Wire up PyTorch inference pipeline in `inference.py`
- [ ] Zustand global store connected to API
- [ ] Live AI Prediction page with probability bars + radar chart (Page 4)
- [ ] Feature Dashboard — 32 cards with status colours (Page 8)
- [ ] Performance Metrics — summary cards + per-class table (Page 13)

### Phase 2 — 3D Interactivity (Week 2–3)

- [ ] Clickable active regions on 3D Sun with popups (Page 2)
- [ ] Solar Evolution Timeline — 5 mini Suns side by side (Page 3)
- [ ] Signal Analysis — Recharts chart with window highlight (Page 7)
- [ ] Solar Intensity — RGB bars + composite 3D Sun (Page 5)
- [ ] Processing Timeline — animated step-by-step stages (Page 6)
- [ ] FITS upload with SSE live pipeline status (Page 12)
- [ ] Confusion matrix Plotly heatmap + training curves (Page 13)

### Phase 3 — Advanced Features (Week 3–4)

- [ ] Flare Evolution Animation — particle system, per-state bloom (Page 15)
- [ ] AI Explanation panel — rules engine + top 5 reasons (Page 9)
- [ ] Confidence Meter + Risk Indicator as persistent sidebar widgets (Pages 10–11)
- [ ] Active Region Detection sidebar with click-to-focus (Page 14)
- [ ] ROC + Precision-Recall curves (Page 13)
- [ ] Future Prediction Mode — static placeholder UI (Page 16)
- [ ] Docker Compose deployment on EC2
- [ ] Nginx reverse proxy + domain / SSL setup

### Phase 4 — Polish (Week 4)

- [ ] SHAP explanation integration (Page 9)
- [ ] Real-time SSE pipeline status (Page 12)
- [ ] Mobile-responsive layout
- [ ] Loading skeletons + error boundaries on all pages
- [ ] Lazy load `<Canvas>` for performance
- [ ] Performance audit (Lighthouse)
- [ ] Final demo recording

---

## Appendix — Key Constants

```typescript
// lib/constants.ts

export const CLASS_NAMES = ['Quiet', 'B', 'C', 'M', 'X'] as const;

export const CLASS_COLORS: Record<string, string> = {
  Quiet: '#22c55e',
  B:     '#eab308',
  C:     '#f97316',
  M:     '#ef4444',
  X:     '#a855f7',
};

export const RISK_LEVELS: Record<number, string> = {
  0: 'LOW',
  1: 'LOW',
  2: 'MEDIUM',
  3: 'HIGH',
  4: 'EXTREME',
};

export const THRESHOLDS = [100, 500, 2_000, 8_000]; // SoLEXS COUNTS/sec

export const MODEL_INFO = {
  name:       'HelioForgeTCN',
  params:     8_573_573,
  checkpoint: 'best_macro_f1.pt',
  epoch:      25,
  macro_f1:   0.8514,
  accuracy:   0.8941,
  val_f1:     0.8714,
} as const;
```

---

*HELIO-FORGE AI — Dashboard Blueprint v1.0*  
*Last updated: July 29, 2026*  
*Status: Ready for implementation*
