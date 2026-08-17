"use client";

import SunScene from "@/components/three/SunScene";
import { usePredictionStore } from "@/store/usePredictionStore";
import { FiRadio, FiActivity, FiLayers, FiSun } from "react-icons/fi";

export default function IntensityView() {
  const { prediction } = usePredictionStore();
  const rgb = prediction.rgb_intensity || { red: 210, green: 140, blue: 45 };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/10 pb-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono-code text-cyan-400 font-bold mb-1">
            <FiRadio className="w-4 h-4" /> PAGE 5 — SOLAR INTENSITY & RGB CHANNEL VIEWER
          </div>
          <h1 className="text-2xl md:text-3xl font-heading font-bold text-white tracking-tight">
            Multi-Channel Hard/Soft X-Ray Flux Synthesis & Composite RGB Sun
          </h1>
          <p className="text-sm text-gray-400 max-w-3xl mt-1">
            Synthesize measured photon count rates across Hard X-Ray (Red), Soft X-Ray (Green), and UV (Blue) channels into a dynamically shifting composite 3D solar rendering.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Intensity Channel Meters */}
        <div className="lg:col-span-5 space-y-4">
          <div className="glass-panel p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-white/10 pb-3 text-xs font-heading font-semibold text-gray-300">
              <span className="flex items-center gap-1.5">
                <FiActivity className="text-cyan-400" /> Channel Count Rates (COUNTS/sec)
              </span>
              <span className="text-[10px] font-mono-code text-gray-500">MinMax Scaled</span>
            </div>

            {/* Red Channel: Hard X-Ray */}
            <div className="glass-card p-4 rounded-xl space-y-2 border border-red-500/20 bg-red-950/10">
              <div className="flex justify-between items-center text-xs font-mono-code">
                <span className="text-red-400 font-bold flex items-center gap-1.5">
                  🔴 Red Channel (Hard X-Ray)
                </span>
                <span className="text-white font-bold">{rgb.red} c/s</span>
              </div>
              <div className="w-full bg-white/5 h-2.5 rounded-full overflow-hidden p-0.5 border border-white/10">
                <div
                  className="h-full bg-red-500 rounded-full transition-all duration-500 shadow-md shadow-red-500/30"
                  style={{ width: `${Math.min(100, Math.round((rgb.red / 255) * 100))}%` }}
                />
              </div>
              <p className="text-[10px] text-gray-400 font-mono-code">
                Measures high-energy non-thermal bremsstrahlung during impulsive flare phase.
              </p>
            </div>

            {/* Green Channel: Soft X-Ray */}
            <div className="glass-card p-4 rounded-xl space-y-2 border border-emerald-500/20 bg-emerald-950/10">
              <div className="flex justify-between items-center text-xs font-mono-code">
                <span className="text-emerald-400 font-bold flex items-center gap-1.5">
                  🟢 Green Channel (Soft X-Ray)
                </span>
                <span className="text-white font-bold">{rgb.green} c/s</span>
              </div>
              <div className="w-full bg-white/5 h-2.5 rounded-full overflow-hidden p-0.5 border border-white/10">
                <div
                  className="h-full bg-emerald-500 rounded-full transition-all duration-500 shadow-md shadow-emerald-500/30"
                  style={{ width: `${Math.min(100, Math.round((rgb.green / 255) * 100))}%` }}
                />
              </div>
              <p className="text-[10px] text-gray-400 font-mono-code">
                Measures thermal plasma heating and gradual flare decay energy.
              </p>
            </div>

            {/* Blue Channel: UV Channel */}
            <div className="glass-card p-4 rounded-xl space-y-2 border border-cyan-500/20 bg-cyan-950/10">
              <div className="flex justify-between items-center text-xs font-mono-code">
                <span className="text-cyan-400 font-bold flex items-center gap-1.5">
                  🔵 Blue Channel (Extreme UV)
                </span>
                <span className="text-white font-bold">{rgb.blue} c/s</span>
              </div>
              <div className="w-full bg-white/5 h-2.5 rounded-full overflow-hidden p-0.5 border border-white/10">
                <div
                  className="h-full bg-cyan-500 rounded-full transition-all duration-500 shadow-md shadow-cyan-500/30"
                  style={{ width: `${Math.min(100, Math.round((rgb.blue / 255) * 100))}%` }}
                />
              </div>
              <p className="text-[10px] text-gray-400 font-mono-code">
                Measures transition region and upper chromospheric emission lines.
              </p>
            </div>
          </div>
        </div>

        {/* Right Column: Composite 3D RGB Sun */}
        <div className="lg:col-span-7 space-y-4">
          <div className="glass-panel p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-white/10 pb-2 text-xs font-heading font-semibold text-gray-300">
              <span className="flex items-center gap-1.5">
                <FiSun className="text-amber-400" /> Composite RGB Solar Sphere Rendering
              </span>
              <span className="text-[10px] font-mono-code text-cyan-400 font-bold">
                RGB ({rgb.red}, {rgb.green}, {rgb.blue})
              </span>
            </div>

            <SunScene className="h-[480px] shadow-2xl" />
          </div>
        </div>
      </div>
    </div>
  );
}
