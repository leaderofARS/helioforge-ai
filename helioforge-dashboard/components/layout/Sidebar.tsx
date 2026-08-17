"use client";

import { usePredictionStore } from "@/store/usePredictionStore";
import { type FlareClass, type RiskLevel } from "@/lib/api";
import {
  FiShield,
  FiZap,
  FiCpu,
  FiUploadCloud,
  FiClock,
  FiAlertTriangle,
} from "react-icons/fi";

const CLASS_CONFIG: Record<
  FlareClass,
  { name: string; color: string; bg: string; risk: RiskLevel; desc: string }
> = {
  Quiet: {
    name: "Quiet",
    color: "#10b981",
    bg: "rgba(16, 185, 129, 0.15)",
    risk: "LOW",
    desc: "Calm Solar State (< 100 c/s)",
  },
  B: {
    name: "B-Class",
    color: "#f59e0b",
    bg: "rgba(245, 158, 11, 0.15)",
    risk: "LOW",
    desc: "Background Flare (100–500 c/s)",
  },
  C: {
    name: "C-Class",
    color: "#f97316",
    bg: "rgba(249, 115, 22, 0.15)",
    risk: "MEDIUM",
    desc: "Minor Flare Event (500–2K c/s)",
  },
  M: {
    name: "M-Class",
    color: "#ef4444",
    bg: "rgba(239, 68, 68, 0.15)",
    risk: "HIGH",
    desc: "Strong Flare Hazard (2K–8K c/s)",
  },
  X: {
    name: "X-Class",
    color: "#a855f7",
    bg: "rgba(168, 85, 247, 0.15)",
    risk: "EXTREME",
    desc: "Extreme Radio Blackout (≥ 8K c/s)",
  },
};

export default function Sidebar() {
  const { prediction, setActiveSection, fetchDemo, isLoading } = usePredictionStore();

  const label = prediction.predicted_label || "M";
  const confidencePct = Math.round((prediction.confidence || 0.87) * 100);
  const classInfo = CLASS_CONFIG[label] || CLASS_CONFIG.M;
  const risk = prediction.risk_level || classInfo.risk;

  const isHighRisk = risk === "HIGH" || risk === "EXTREME";

  return (
    <aside className="w-80 flex-shrink-0 hidden lg:block border-l border-white/10 bg-[#050814]/90 p-4 space-y-4 font-sans backdrop-blur-xl">
      {/* 1. Solar Risk Indicator (Blueprint Specification Page 11) */}
      <div
        className={`glass-panel p-4 space-y-3 relative overflow-hidden transition-all ${
          isHighRisk ? "border-red-500/30 bg-red-950/20" : ""
        }`}
      >
        <div className="flex items-center justify-between">
          <span className="text-xs font-heading font-semibold text-gray-400 flex items-center gap-1.5">
            <FiShield className={isHighRisk ? "text-red-400" : "text-amber-400"} /> Solar Risk Level
          </span>
          {isHighRisk && (
            <span className="flex h-2 w-2 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500" />
            </span>
          )}
        </div>

        <div className="flex items-baseline justify-between">
          <span
            className="text-2xl font-heading font-bold tracking-tight"
            style={{ color: classInfo.color }}
          >
            {risk}
          </span>
          <span
            className="px-2 py-0.5 rounded text-xs font-mono-code font-bold"
            style={{ backgroundColor: classInfo.bg, color: classInfo.color }}
          >
            {classInfo.name}
          </span>
        </div>

        {/* Risk Stepper */}
        <div className="grid grid-cols-4 gap-1.5 pt-1">
          {(["LOW", "MEDIUM", "HIGH", "EXTREME"] as RiskLevel[]).map((step) => {
            const isActive = step === risk;
            const stepColors: Record<RiskLevel, string> = {
              LOW: "#10b981",
              MEDIUM: "#f59e0b",
              HIGH: "#ef4444",
              EXTREME: "#a855f7",
            };
            return (
              <div
                key={step}
                className={`h-1.5 rounded-full transition-all ${
                  isActive ? "scale-y-125" : "opacity-30"
                }`}
                style={{
                  backgroundColor: stepColors[step],
                  boxShadow: isActive ? `0 0 10px ${stepColors[step]}` : "none",
                }}
              />
            );
          })}
        </div>
        <p className="text-[11px] text-gray-400 font-mono-code leading-tight">
          {classInfo.desc}
        </p>
      </div>

      {/* 2. AI Confidence Meter (Blueprint Specification Page 10) */}
      <div className="glass-panel p-4 space-y-3">
        <div className="flex items-center justify-between text-xs font-heading font-semibold text-gray-400">
          <span className="flex items-center gap-1.5">
            <FiCpu className="text-amber-400" /> AI Certainty
          </span>
          <span className="font-mono-code font-bold text-amber-400 text-sm">
            {confidencePct}%
          </span>
        </div>

        {/* Progress Bar */}
        <div className="w-full bg-white/5 h-2.5 rounded-full overflow-hidden p-0.5 border border-white/10">
          <div
            className="h-full rounded-full transition-all duration-700 ease-out"
            style={{
              width: `${confidencePct}%`,
              backgroundColor: classInfo.color,
              boxShadow: `0 0 12px ${classInfo.color}`,
            }}
          />
        </div>

        <div className="flex justify-between text-[10px] text-gray-500 font-mono-code">
          <span>Low (&lt;60%)</span>
          <span>Nominal (75%)</span>
          <span>High (&gt;90%)</span>
        </div>
      </div>

      {/* 3. Current Observation Meta Card */}
      <div className="glass-panel p-4 space-y-2.5">
        <div className="text-xs font-heading font-semibold text-gray-400 border-b border-white/10 pb-2 flex items-center justify-between">
          <span className="flex items-center gap-1.5">
            <FiZap className="text-amber-400" /> Active Observation
          </span>
          <span className="text-[10px] text-emerald-400 font-mono-code">LIVE</span>
        </div>

        <div className="space-y-1.5 text-xs font-mono-code">
          <div className="flex justify-between text-gray-400">
            <span>Obs ID:</span>
            <span className="text-gray-200 font-bold truncate max-w-[140px]">
              {prediction.observation_id || "OBS_20260728_0031"}
            </span>
          </div>

          <div className="flex justify-between text-gray-400">
            <span>Window Size:</span>
            <span className="text-gray-200">512 timesteps</span>
          </div>

          <div className="flex justify-between text-gray-400">
            <span>Inference Time:</span>
            <span className="text-amber-400 font-bold">
              {prediction.processing_time_ms || 78} ms
            </span>
          </div>

          <div className="flex justify-between text-gray-400">
            <span>Model:</span>
            <span className="text-gray-300">HelioForgeTCN</span>
          </div>
        </div>
      </div>

      {/* 4. Quick Actions */}
      <div className="space-y-2 pt-2">
        <button
          onClick={() => setActiveSection("upload")}
          className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl bg-gradient-to-r from-amber-500 to-orange-600 text-white font-heading text-xs font-bold shadow-lg shadow-orange-500/20 hover:from-amber-400 hover:to-orange-500 transition-all active:scale-[0.98]"
        >
          <FiUploadCloud className="w-4 h-4" /> Upload FITS Observation
        </button>

        <button
          onClick={() => fetchDemo((prediction.sample_index || 0) + 1)}
          disabled={isLoading}
          className="w-full flex items-center justify-center gap-2 py-2 px-4 rounded-xl border border-white/10 bg-white/5 text-gray-300 hover:bg-white/10 hover:text-white font-mono-code text-xs transition-all disabled:opacity-50"
        >
          <FiClock className="w-3.5 h-3.5" /> Next Demo Observation
        </button>
      </div>

      {/* High Alert Banner */}
      {isHighRisk && (
        <div className="rounded-xl border border-red-500/40 bg-red-500/10 p-3 text-xs space-y-1 text-red-300 animate-pulse">
          <div className="flex items-center gap-1.5 font-bold font-heading text-red-400">
            <FiAlertTriangle className="w-4 h-4" /> FLARE HAZARD ALERT
          </div>
          <p className="text-[11px] leading-tight opacity-90">
            Strong solar flare detected. Potential ionospheric disturbance & High-Frequency radio attenuation.
          </p>
        </div>
      )}
    </aside>
  );
}
