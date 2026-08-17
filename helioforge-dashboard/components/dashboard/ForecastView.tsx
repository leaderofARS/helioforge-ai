"use client";

import { usePredictionStore } from "@/store/usePredictionStore";
import { FiTrendingUp, FiAlertTriangle, FiCheckCircle, FiClock, FiShield } from "react-icons/fi";

export default function ForecastView() {
  const { prediction } = usePredictionStore();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/10 pb-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono-code text-amber-400 font-bold mb-1">
            <FiTrendingUp className="w-4 h-4" /> PAGE 16 — FUTURE PREDICTION TRAJECTORY FORECAST
          </div>
          <h1 className="text-2xl md:text-3xl font-heading font-bold text-white tracking-tight">
            Multi-Step Solar Flare Forecasting (t → t+30m → t+60m)
          </h1>
          <p className="text-sm text-gray-400 max-w-3xl mt-1">
            Future temporal trajectory forecasting mode evaluating prospective magnetic energy accumulation and eruption probability over the next 60 minutes.
          </p>
        </div>
      </div>

      <div className="glass-panel p-6 space-y-6">
        <div className="flex items-center justify-between border-b border-white/10 pb-3">
          <span className="text-xs font-heading font-bold text-amber-400 flex items-center gap-1.5">
            <FiClock /> Predicted Eruption Trajectory Timeline
          </span>
          <span className="text-xs font-mono-code text-gray-400">
            Observation: {prediction.observation_id || "OBS_20260728_0031"}
          </span>
        </div>

        {/* 3-Step Forecast Path Flow */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="glass-card p-5 rounded-2xl border border-emerald-500/30 bg-emerald-950/10 space-y-2">
            <div className="flex justify-between items-center text-xs font-mono-code">
              <span className="text-emerald-400 font-bold">Current State ($t_0$)</span>
              <span className="text-gray-400">Now</span>
            </div>
            <div className="text-2xl font-heading font-bold text-white">
              {prediction.predicted_label || "C"}-Class Flare
            </div>
            <p className="text-xs text-gray-400 font-mono-code">
              Active observation baseline state.
            </p>
          </div>

          <div className="glass-card p-5 rounded-2xl border border-amber-500/30 bg-amber-950/10 space-y-2">
            <div className="flex justify-between items-center text-xs font-mono-code">
              <span className="text-amber-400 font-bold">Forecast Horizon (+30m)</span>
              <span className="text-amber-300 font-bold">78% Confidence</span>
            </div>
            <div className="text-2xl font-heading font-bold text-amber-400">
              M-Class Flare
            </div>
            <p className="text-xs text-gray-400 font-mono-code">
              Magnetic flux gradient acceleration phase.
            </p>
          </div>

          <div className="glass-card p-5 rounded-2xl border border-red-500/30 bg-red-950/10 space-y-2">
            <div className="flex justify-between items-center text-xs font-mono-code">
              <span className="text-red-400 font-bold">Forecast Horizon (+60m)</span>
              <span className="text-red-400 font-bold">64% Confidence</span>
            </div>
            <div className="text-2xl font-heading font-bold text-red-500">
              X-Class Flare Alert
            </div>
            <p className="text-xs text-gray-400 font-mono-code">
              High risk of major ionospheric disturbance.
            </p>
          </div>
        </div>

        {/* Mission Operational Actions */}
        <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 space-y-2 text-xs font-mono-code">
          <div className="font-bold text-amber-400 flex items-center gap-2">
            <FiShield /> Recommended Spacecraft Operational Safeguards:
          </div>
          <ul className="list-disc list-inside space-y-1 text-gray-300">
            <li>Switch sensitive Aditya-L1 / satellite optical payloads to safe standby mode.</li>
            <li>Issue High-Frequency (HF) radio propagation advisory to ISRO ground telemetry stations.</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
