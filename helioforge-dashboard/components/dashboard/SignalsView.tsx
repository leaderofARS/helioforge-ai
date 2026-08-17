"use client";

import { usePredictionStore } from "@/store/usePredictionStore";
import { useState } from "react";
import { FiActivity, FiSearch, FiZap, FiSliders } from "react-icons/fi";

export default function SignalsView() {
  const { prediction } = usePredictionStore();
  const [selectedChannel, setSelectedChannel] = useState<"soft" | "hard" | "uv">("soft");

  const softSignal = prediction.signal || [];
  const hardSignal = prediction.hard_signal || [];
  const uvSignal = prediction.uv_signal || [];

  const activeSignal =
    selectedChannel === "soft" ? softSignal : selectedChannel === "hard" ? hardSignal : uvSignal;

  const maxVal = Math.max(...(activeSignal.length ? activeSignal : [1]), 0.001);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/10 pb-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono-code text-emerald-400 font-bold mb-1">
            <FiActivity className="w-4 h-4" /> PAGE 7 — HIGH-PRECISION SIGNAL ANALYSIS
          </div>
          <h1 className="text-2xl md:text-3xl font-heading font-bold text-white tracking-tight">
            Raw Telemetry Time-Series & Prediction Window (512 Timesteps)
          </h1>
          <p className="text-sm text-gray-400 max-w-3xl mt-1">
            Analyze 512 consecutive timesteps of calibrated photon count rates from Aditya-L1 HEL1OS and SoLEXS payload sensors.
          </p>
        </div>

        {/* Channel Selector Buttons */}
        <div className="flex items-center gap-2 bg-[#0b0f19] p-1.5 rounded-xl border border-white/10">
          <button
            onClick={() => setSelectedChannel("soft")}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono-code transition-all ${
              selectedChannel === "soft"
                ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-bold shadow-md shadow-emerald-500/10"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            Soft X-Ray (SoLEXS Ch 0)
          </button>
          <button
            onClick={() => setSelectedChannel("hard")}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono-code transition-all ${
              selectedChannel === "hard"
                ? "bg-red-500/20 text-red-300 border border-red-500/40 font-bold shadow-md shadow-red-500/10"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            Hard X-Ray (HEL1OS Ch 1)
          </button>
          <button
            onClick={() => setSelectedChannel("uv")}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono-code transition-all ${
              selectedChannel === "uv"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold shadow-md shadow-cyan-500/10"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            UV Channel
          </button>
        </div>
      </div>

      {/* Main Signal Display Canvas Card */}
      <div className="glass-panel p-6 space-y-4">
        <div className="flex items-center justify-between border-b border-white/10 pb-3 text-xs font-heading font-semibold text-gray-300">
          <span className="flex items-center gap-1.5">
            <FiActivity className="text-emerald-400" /> Count Rate Waveform (512 Timesteps)
          </span>
          <span className="text-[11px] font-mono-code text-amber-400 font-bold">
            Prediction Window Highlighted
          </span>
        </div>

        {/* High-Resolution SVG Time Series Plotter */}
        <div className="relative h-72 w-full bg-black/60 rounded-xl border border-white/10 overflow-hidden p-4">
          <svg className="w-full h-full" viewBox="0 0 512 180" preserveAspectRatio="none">
            {/* Grid Lines */}
            {[0, 45, 90, 135, 180].map((y) => (
              <line
                key={y}
                x1="0"
                y1={y}
                x2="512"
                y2={y}
                stroke="rgba(255,255,255,0.06)"
                strokeDasharray="4 4"
              />
            ))}
            {[0, 128, 256, 384, 512].map((x) => (
              <line
                key={x}
                x1={x}
                y1="0"
                x2={x}
                y2="180"
                stroke="rgba(255,255,255,0.06)"
                strokeDasharray="4 4"
              />
            ))}

            {/* Shaded Active Prediction Window */}
            <rect
              x="180"
              y="0"
              width="180"
              height="180"
              fill="rgba(249, 115, 22, 0.12)"
              stroke="rgba(249, 115, 22, 0.4)"
              strokeDasharray="3 3"
            />

            {/* Signal Polyline */}
            {activeSignal.length > 0 && (
              <polyline
                fill="none"
                stroke={
                  selectedChannel === "soft"
                    ? "#10b981"
                    : selectedChannel === "hard"
                    ? "#ef4444"
                    : "#06b6d4"
                }
                strokeWidth="2"
                points={activeSignal
                  .map((val, idx) => {
                    const normY = 175 - (val / maxVal) * 160;
                    return `${idx},${normY}`;
                  })
                  .join(" ")}
              />
            )}
          </svg>

          <div className="absolute top-3 left-3 bg-black/80 backdrop-blur-md px-2.5 py-1 rounded text-[10px] font-mono-code text-gray-300 border border-white/10">
            Peak Intensity: {maxVal.toFixed(3)} c/s
          </div>

          <div className="absolute top-3 right-3 bg-amber-500/20 backdrop-blur-md px-2.5 py-1 rounded text-[10px] font-mono-code text-amber-300 border border-amber-500/30">
            512-Timestep Receptive Field
          </div>
        </div>

        {/* Legend & Stats Ticker */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono-code pt-2">
          <div className="glass-card p-3 rounded-xl">
            <span className="text-[10px] text-gray-500 block">Total Timesteps</span>
            <span className="text-gray-200 font-bold">512 points</span>
          </div>

          <div className="glass-card p-3 rounded-xl">
            <span className="text-[10px] text-gray-500 block">Sampling Frequency</span>
            <span className="text-gray-200 font-bold">1 Hz Cadence</span>
          </div>

          <div className="glass-card p-3 rounded-xl">
            <span className="text-[10px] text-gray-500 block">TCN Receptive Field</span>
            <span className="text-emerald-400 font-bold">511 Timesteps (99.8%)</span>
          </div>

          <div className="glass-card p-3 rounded-xl">
            <span className="text-[10px] text-gray-500 block">Predicted Class</span>
            <span className="text-amber-400 font-bold">{prediction.predicted_label}-Class</span>
          </div>
        </div>
      </div>
    </div>
  );
}
