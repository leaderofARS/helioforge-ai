"use client";

import SunScene from "@/components/three/SunScene";
import { usePredictionStore, type WavelengthMode } from "@/store/usePredictionStore";
import { type FlareClass } from "@/lib/api";
import { FiSun, FiEye, FiSliders, FiCompass, FiShield, FiCrosshair } from "react-icons/fi";

const WAVELENGTHS: Array<{ id: WavelengthMode; label: string; spectral: string; range: string }> = [
  { id: "171", label: "AIA 171Å", spectral: "Gold / Extreme UV", range: "Fe IX/X • Upper Transition Region (~0.6 MK)" },
  { id: "304", label: "AIA 304Å", spectral: "Crimson Red / UV", range: "He II • Chromosphere & Prominences (~0.05 MK)" },
  { id: "131", label: "AIA 131Å", spectral: "Cyan / High Temp", range: "Fe VIII/XXI • Flaring Plasma (>10 MK)" },
  { id: "hmi", label: "HMI Magnetogram", spectral: "Monochrome Magnetic", range: "Photospheric Line-of-Sight Magnetic Field" },
  { id: "rgb", label: "Composite RGB", spectral: "Hard/Soft/UV Energy", range: "Multi-band Hard/Soft X-Ray Flux Synthesis" },
];

export default function InteractiveSunView() {
  const { prediction, wavelength, setWavelength, selectedRegion, setSelectedRegion } = usePredictionStore();
  const regions = prediction.active_regions || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/10 pb-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono-code text-amber-400 font-bold mb-1">
            <FiSun className="w-4 h-4" /> PAGE 2 — 3D SOLAR OBSERVATORY
          </div>
          <h1 className="text-2xl md:text-3xl font-heading font-bold text-white tracking-tight">
            Interactive 3D Solar Atmosphere & Active Region Inspector
          </h1>
          <p className="text-sm text-gray-400 max-w-3xl mt-1">
            Inspect the 3D surface geometry, coronal magnetic field lines, and active region hotspots across NASA SDO and ISRO Aditya-L1 multi-spectral wavelengths.
          </p>
        </div>
      </div>

      {/* Main Grid: 3D Sun Scene (Center) + Inspection Controls (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Fullscreen 3D Scene View */}
        <div className="lg:col-span-8 space-y-4">
          <SunScene className="h-[650px] shadow-2xl" />

          {/* Wavelength Spectrum Quick Selector Tabs */}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
            {WAVELENGTHS.map((w) => (
              <button
                key={w.id}
                onClick={() => setWavelength(w.id)}
                className={`p-3 rounded-xl border text-left transition-all ${
                  wavelength === w.id
                    ? "bg-amber-500/20 border-amber-500/50 text-amber-300 font-bold shadow-lg shadow-amber-500/10"
                    : "bg-white/5 border-white/5 text-gray-400 hover:bg-white/10 hover:text-gray-200"
                }`}
              >
                <div className="font-heading text-xs">{w.label}</div>
                <div className="text-[10px] font-mono-code text-gray-400 mt-0.5 truncate">{w.spectral}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Right Sidebar: Active Region Inspector & Spectrograph Telemetry */}
        <div className="lg:col-span-4 space-y-4">
          {/* Spectral Filter Details Card */}
          <div className="glass-panel p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-white/10 pb-2 text-xs font-heading font-semibold text-gray-300">
              <span className="flex items-center gap-1.5">
                <FiSliders className="text-amber-400" /> Wavelength Spectrograph
              </span>
              <span className="text-[10px] font-mono-code text-amber-400 font-bold">
                {wavelength.toUpperCase()}
              </span>
            </div>

            <div className="space-y-2 text-xs font-mono-code">
              <div>
                <span className="text-gray-500 block text-[10px]">Filter Mode</span>
                <span className="text-gray-200 font-bold">
                  {WAVELENGTHS.find((w) => w.id === wavelength)?.label}
                </span>
              </div>

              <div>
                <span className="text-gray-500 block text-[10px]">Physical Target</span>
                <span className="text-amber-300">
                  {WAVELENGTHS.find((w) => w.id === wavelength)?.range}
                </span>
              </div>
            </div>
          </div>

          {/* Active Regions Catalog */}
          <div className="glass-panel p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-white/10 pb-2 text-xs font-heading font-semibold text-gray-300">
              <span className="flex items-center gap-1.5">
                <FiCompass className="text-emerald-400" /> Active Region Catalog ({regions.length})
              </span>
              <span className="text-[10px] text-gray-500 font-mono-code">Click to Focus</span>
            </div>

            <div className="space-y-2 max-h-[380px] overflow-y-auto pr-1">
              {regions.map((region) => {
                const isSelected = selectedRegion?.id === region.id;
                const clsLabel = (region.label || "M") as FlareClass;
                return (
                  <button
                    key={region.id}
                    onClick={() => setSelectedRegion(region)}
                    className={`w-full p-3 rounded-xl border text-left transition-all flex items-center justify-between ${
                      isSelected
                        ? "bg-amber-500/15 border-amber-500/40 text-amber-300 shadow-md"
                        : "bg-white/5 border-white/5 text-gray-300 hover:bg-white/10"
                    }`}
                  >
                    <div className="space-y-0.5">
                      <div className="font-heading font-bold text-xs flex items-center gap-1.5">
                        <FiCrosshair className={isSelected ? "text-amber-400" : "text-gray-500"} />
                        {region.id}
                      </div>
                      <div className="text-[10px] font-mono-code text-gray-400">
                        Lat: {region.lat > 0 ? `+${region.lat}` : region.lat}° • Lon: {region.lon > 0 ? `+${region.lon}` : region.lon}°
                      </div>
                    </div>

                    <div className="text-right">
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono-code font-bold bg-white/10">
                        {clsLabel}-Class
                      </span>
                      {region.intensity && (
                        <div className="text-[10px] font-mono-code text-amber-400 mt-1">
                          {region.intensity.toLocaleString()} c/s
                        </div>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
