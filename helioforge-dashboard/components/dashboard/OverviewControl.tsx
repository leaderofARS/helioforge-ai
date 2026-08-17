"use client";

import SunScene from "@/components/three/SunScene";
import { usePredictionStore } from "@/store/usePredictionStore";
import { type FlareClass } from "@/lib/api";
import {
  FiShield,
  FiActivity,
  FiZap,
  FiClock,
  FiCheckCircle,
  FiArrowUpRight,
  FiLayers,
  FiBarChart2,
  FiRadio,
} from "react-icons/fi";

const CLASS_STYLES: Record<FlareClass, { color: string; bg: string; border: string }> = {
  Quiet: { color: "#10b981", bg: "rgba(16, 185, 129, 0.15)", border: "rgba(16, 185, 129, 0.3)" },
  B: { color: "#f59e0b", bg: "rgba(245, 158, 11, 0.15)", border: "rgba(245, 158, 11, 0.3)" },
  C: { color: "#f97316", bg: "rgba(249, 115, 22, 0.15)", border: "rgba(249, 115, 22, 0.3)" },
  M: { color: "#ef4444", bg: "rgba(239, 68, 68, 0.15)", border: "rgba(239, 68, 68, 0.3)" },
  X: { color: "#a855f7", bg: "rgba(168, 85, 247, 0.15)", border: "rgba(168, 85, 247, 0.3)" },
};

export default function OverviewControl() {
  const { prediction, setActiveSection, fetchDemo } = usePredictionStore();
  const label = prediction.predicted_label || "M";
  const classStyle = CLASS_STYLES[label] || CLASS_STYLES.M;
  const confidencePct = Math.round((prediction.confidence || 0.87) * 100);

  return (
    <div className="space-y-6">
      {/* Top Banner / Hero HUD */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/10 pb-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono-code text-amber-400 font-bold mb-1">
            <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping" />
            LIVE MISSION CONTROL OVERVIEW
          </div>
          <h1 className="text-2xl md:text-3xl font-heading font-bold tracking-tight text-white">
            Solar Flare Intelligence System
          </h1>
          <p className="text-sm text-gray-400 max-w-2xl mt-1">
            Real-time automated classification and space weather hazard detection powered by deep temporal convolutions on ISRO Aditya-L1 HEL1OS & SoLEXS telemetry.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => fetchDemo((prediction.sample_index || 0) + 1)}
            className="flex items-center gap-2 px-4 py-2 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 text-gray-200 font-mono-code text-xs transition-all"
          >
            <FiClock className="text-amber-400" /> Cycle Observation
          </button>
          <button
            onClick={() => setActiveSection("upload")}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-amber-500 hover:bg-amber-400 text-black font-heading font-bold text-xs shadow-lg shadow-amber-500/20 transition-all"
          >
            Upload FITS <FiArrowUpRight />
          </button>
        </div>
      </div>

      {/* Grid: 3D Sun Hero Observatory + Telemetry Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Photorealistic 3D Sun Observatory */}
        <div className="lg:col-span-8 space-y-4">
          <div className="relative">
            <SunScene className="shadow-2xl" />
          </div>

          {/* Bottom Telemetry Ticker under 3D Scene */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="glass-card p-3 rounded-xl border border-white/5">
              <span className="text-[10px] text-gray-400 font-mono-code block">Observation ID</span>
              <span className="text-xs font-mono-code font-bold text-amber-300 truncate block">
                {prediction.observation_id || "OBS_20260728_0031"}
              </span>
            </div>

            <div className="glass-card p-3 rounded-xl border border-white/5">
              <span className="text-[10px] text-gray-400 font-mono-code block">Timestep Window</span>
              <span className="text-xs font-mono-code font-bold text-gray-200">
                512 timesteps (RF: 511)
              </span>
            </div>

            <div className="glass-card p-3 rounded-xl border border-white/5">
              <span className="text-[10px] text-gray-400 font-mono-code block">Processing Latency</span>
              <span className="text-xs font-mono-code font-bold text-emerald-400">
                {prediction.processing_time_ms || 78.2} ms
              </span>
            </div>

            <div className="glass-card p-3 rounded-xl border border-white/5">
              <span className="text-[10px] text-gray-400 font-mono-code block">Target Class</span>
              <span className="text-xs font-mono-code font-bold" style={{ color: classStyle.color }}>
                {label}-Class Event
              </span>
            </div>
          </div>
        </div>

        {/* Right Column: Prediction HUD Cards & Class Breakdown */}
        <div className="lg:col-span-4 space-y-4">
          {/* Main Prediction Highlight Card */}
          <div
            className="glass-panel p-5 space-y-4 border"
            style={{ borderColor: classStyle.border, backgroundColor: classStyle.bg }}
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono-code uppercase tracking-wider text-gray-400 flex items-center gap-1.5">
                <FiZap style={{ color: classStyle.color }} /> Primary Model Prediction
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono-code font-bold bg-white/10 text-white">
                HelioForgeTCN
              </span>
            </div>

            <div className="flex items-baseline justify-between pt-1">
              <div>
                <span
                  className="text-4xl font-heading font-extrabold tracking-tight"
                  style={{ color: classStyle.color }}
                >
                  {label}-Class
                </span>
                <span className="block text-xs text-gray-400 font-mono-code mt-0.5">
                  Solar Activity Level
                </span>
              </div>

              <div className="text-right">
                <span className="text-2xl font-mono-code font-bold text-white">
                  {confidencePct}%
                </span>
                <span className="block text-[10px] text-amber-400 font-mono-code">Confidence</span>
              </div>
            </div>

            {/* Risk Badge Bar */}
            <div className="p-3 rounded-xl bg-black/40 border border-white/10 flex items-center justify-between text-xs font-mono-code">
              <span className="text-gray-400 flex items-center gap-1.5">
                <FiShield style={{ color: classStyle.color }} /> Risk Status:
              </span>
              <span className="font-bold tracking-wider" style={{ color: classStyle.color }}>
                {prediction.risk_level || "HIGH"}
              </span>
            </div>
          </div>

          {/* Class Probabilities Distribution */}
          <div className="glass-panel p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-white/10 pb-2 text-xs font-heading font-semibold text-gray-300">
              <span className="flex items-center gap-1.5">
                <FiBarChart2 className="text-amber-400" /> Probability Distribution
              </span>
              <button
                onClick={() => setActiveSection("prediction")}
                className="text-[11px] text-amber-400 hover:underline font-mono-code"
              >
                View Full Radar →
              </button>
            </div>

            <div className="space-y-2 text-xs font-mono-code">
              {(["Quiet", "B", "C", "M", "X"] as FlareClass[]).map((cls) => {
                const prob = prediction.probabilities?.[cls] || 0;
                const probPct = Math.round(prob * 100);
                const isWinner = cls === label;
                const colors: Record<FlareClass, string> = {
                  Quiet: "#10b981",
                  B: "#f59e0b",
                  C: "#f97316",
                  M: "#ef4444",
                  X: "#a855f7",
                };

                return (
                  <div key={cls} className="space-y-1">
                    <div className="flex justify-between items-center text-[11px]">
                      <span className={isWinner ? "font-bold text-white" : "text-gray-400"}>
                        {cls}-Class
                      </span>
                      <span className={isWinner ? "font-bold" : "text-gray-400"} style={{ color: isWinner ? colors[cls] : undefined }}>
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>

                    <div className="w-full bg-white/5 h-2 rounded-full overflow-hidden p-0.5 border border-white/5">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{
                          width: `${probPct}%`,
                          backgroundColor: colors[cls],
                          boxShadow: isWinner ? `0 0 10px ${colors[cls]}` : "none",
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Quick Action Navigation Grid */}
          <div className="grid grid-cols-2 gap-2 text-xs">
            <button
              onClick={() => setActiveSection("evolution")}
              className="p-3 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 hover:border-amber-500/30 text-left space-y-1 transition-all group"
            >
              <FiClock className="text-amber-400 w-4 h-4 group-hover:scale-110 transition-transform" />
              <div className="font-heading font-semibold text-gray-200">Evolution Timeline</div>
              <div className="text-[10px] text-gray-400 font-mono-code">5-frame history</div>
            </button>

            <button
              onClick={() => setActiveSection("signals")}
              className="p-3 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 hover:border-amber-500/30 text-left space-y-1 transition-all group"
            >
              <FiActivity className="text-emerald-400 w-4 h-4 group-hover:scale-110 transition-transform" />
              <div className="font-heading font-semibold text-gray-200">Signal Analysis</div>
              <div className="text-[10px] text-gray-400 font-mono-code">Soft/Hard X-Rays</div>
            </button>

            <button
              onClick={() => setActiveSection("features")}
              className="p-3 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 hover:border-amber-500/30 text-left space-y-1 transition-all group"
            >
              <FiLayers className="text-purple-400 w-4 h-4 group-hover:scale-110 transition-transform" />
              <div className="font-heading font-semibold text-gray-200">32 Features</div>
              <div className="text-[10px] text-gray-400 font-mono-code">Physics metrics</div>
            </button>

            <button
              onClick={() => setActiveSection("intensity")}
              className="p-3 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 hover:border-amber-500/30 text-left space-y-1 transition-all group"
            >
              <FiRadio className="text-cyan-400 w-4 h-4 group-hover:scale-110 transition-transform" />
              <div className="font-heading font-semibold text-gray-200">RGB Intensity</div>
              <div className="text-[10px] text-gray-400 font-mono-code">Multi-channel flux</div>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
