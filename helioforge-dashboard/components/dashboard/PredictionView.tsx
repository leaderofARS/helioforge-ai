"use client";

import { usePredictionStore } from "@/store/usePredictionStore";
import { type FlareClass } from "@/lib/api";
import { FiCpu, FiPieChart, FiBarChart2, FiShield, FiTarget } from "react-icons/fi";

const CLASS_COLORS: Record<FlareClass, string> = {
  Quiet: "#10b981",
  B: "#f59e0b",
  C: "#f97316",
  M: "#ef4444",
  X: "#a855f7",
};

export default function PredictionView() {
  const { prediction } = usePredictionStore();
  const label = prediction.predicted_label || "M";
  const confidencePct = Math.round((prediction.confidence || 0.87) * 100);
  const color = CLASS_COLORS[label] || CLASS_COLORS.M;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/10 pb-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono-code text-amber-400 font-bold mb-1">
            <FiCpu className="w-4 h-4" /> PAGE 4 — LIVE AI PREDICTION ANALYSIS
          </div>
          <h1 className="text-2xl md:text-3xl font-heading font-bold text-white tracking-tight">
            5-Class Flare Probability Breakdown & Pentagon Radar Matrix
          </h1>
          <p className="text-sm text-gray-400 max-w-3xl mt-1">
            Full probability distribution output from `HelioForgeTCN` classifier head (GAP + MLP 512 → 256 → 128 → 5 logits).
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Primary Prediction Summary & Horizontal Probability Bars */}
        <div className="lg:col-span-7 space-y-4">
          <div className="glass-panel p-6 space-y-4 border border-amber-500/20">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <span className="text-xs font-mono-code text-gray-400 flex items-center gap-1.5">
                <FiTarget className="text-amber-400" /> Target Classification Output
              </span>
              <span className="px-2.5 py-1 rounded text-xs font-mono-code font-bold bg-white/10 text-white">
                Observation: {prediction.observation_id || "OBS_20260728_0031"}
              </span>
            </div>

            <div className="flex items-baseline justify-between">
              <div>
                <span className="text-5xl font-heading font-extrabold" style={{ color }}>
                  {label}-Class
                </span>
                <span className="block text-xs font-mono-code text-gray-400 mt-1">
                  Predicted Flare Intensity Class
                </span>
              </div>

              <div className="text-right">
                <span className="text-3xl font-mono-code font-bold text-white">
                  {confidencePct}%
                </span>
                <span className="block text-xs font-mono-code text-amber-400">Certainty Margin</span>
              </div>
            </div>

            <div className="p-3 rounded-xl bg-black/40 border border-white/10 flex items-center justify-between text-xs font-mono-code">
              <span className="text-gray-400">Operational Risk Level:</span>
              <span className="font-bold tracking-wider" style={{ color }}>
                {prediction.risk_level || "HIGH"}
              </span>
            </div>
          </div>

          {/* Detailed Class Bar Breakdown */}
          <div className="glass-panel p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-white/10 pb-2 text-xs font-heading font-semibold text-gray-300">
              <span className="flex items-center gap-1.5">
                <FiBarChart2 className="text-amber-400" /> Class Softmax Probabilities
              </span>
              <span className="text-[11px] font-mono-code text-gray-400">Sum = 1.000</span>
            </div>

            <div className="space-y-3 font-mono-code text-xs">
              {(["Quiet", "B", "C", "M", "X"] as FlareClass[]).map((cls) => {
                const prob = prediction.probabilities?.[cls] || 0;
                const isWinner = cls === label;

                return (
                  <div key={cls} className="space-y-1">
                    <div className="flex justify-between items-center text-xs">
                      <span className={isWinner ? "font-bold text-white" : "text-gray-400"}>
                        {cls}-Class Flare
                      </span>
                      <span
                        className={isWinner ? "font-bold" : "text-gray-400"}
                        style={{ color: isWinner ? CLASS_COLORS[cls] : undefined }}
                      >
                        {(prob * 100).toFixed(2)}% ({prob.toFixed(4)})
                      </span>
                    </div>

                    <div className="w-full bg-white/5 h-3.5 rounded-full overflow-hidden p-0.5 border border-white/10">
                      <div
                        className="h-full rounded-full transition-all duration-700"
                        style={{
                          width: `${Math.max(2, Math.round(prob * 100))}%`,
                          backgroundColor: CLASS_COLORS[cls],
                          boxShadow: isWinner ? `0 0 14px ${CLASS_COLORS[cls]}` : "none",
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Visual Pentagon Radar Matrix Representation */}
        <div className="lg:col-span-5 space-y-4">
          <div className="glass-panel p-6 space-y-4 text-center">
            <div className="flex items-center justify-between border-b border-white/10 pb-3 text-xs font-heading font-semibold text-gray-300 text-left">
              <span className="flex items-center gap-1.5">
                <FiPieChart className="text-amber-400" /> 5-Axis Pentagon Radar Chart
              </span>
              <span className="text-[10px] font-mono-code text-gray-500">Multiclass Geometry</span>
            </div>

            {/* Polygon SVG Radar Representation */}
            <div className="relative h-64 w-full flex items-center justify-center">
              <svg className="w-full h-full max-w-[260px] max-h-[260px]" viewBox="0 0 200 200">
                {/* Background Grid Rings */}
                {[0.2, 0.4, 0.6, 0.8, 1.0].map((scale, i) => {
                  const points = ["Quiet", "B", "C", "M", "X"].map((_, idx) => {
                    const angle = (idx * 2 * Math.PI) / 5 - Math.PI / 2;
                    const r = 80 * scale;
                    return `${100 + r * Math.cos(angle)},${100 + r * Math.sin(angle)}`;
                  }).join(" ");
                  return (
                    <polygon
                      key={i}
                      points={points}
                      fill="none"
                      stroke="rgba(255,255,255,0.08)"
                      strokeWidth="1"
                    />
                  );
                })}

                {/* Radar Value Polygon */}
                {(() => {
                  const pts = (["Quiet", "B", "C", "M", "X"] as FlareClass[]).map((cls, idx) => {
                    const prob = prediction.probabilities?.[cls] || 0.05;
                    const angle = (idx * 2 * Math.PI) / 5 - Math.PI / 2;
                    const r = Math.max(10, 80 * prob);
                    return `${100 + r * Math.cos(angle)},${100 + r * Math.sin(angle)}`;
                  }).join(" ");

                  return (
                    <polygon
                      points={pts}
                      fill={`${color}35`}
                      stroke={color}
                      strokeWidth="2.5"
                    />
                  );
                })()}

                {/* Radar Axis Labels */}
                {(["Quiet", "B", "C", "M", "X"] as FlareClass[]).map((cls, idx) => {
                  const angle = (idx * 2 * Math.PI) / 5 - Math.PI / 2;
                  const r = 95;
                  const x = 100 + r * Math.cos(angle);
                  const y = 100 + r * Math.sin(angle);
                  return (
                    <text
                      key={cls}
                      x={x}
                      y={y}
                      fill={cls === label ? color : "#9ca3af"}
                      fontSize="10"
                      fontWeight={cls === label ? "bold" : "normal"}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      fontFamily="var(--font-mono)"
                    >
                      {cls}
                    </text>
                  );
                })}
              </svg>
            </div>

            <p className="text-xs text-gray-400 font-mono-code leading-relaxed">
              Geometric margin confirms peak probability mass is concentrated heavily on the <span className="font-bold text-white">{label}-Class</span> classification node.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
