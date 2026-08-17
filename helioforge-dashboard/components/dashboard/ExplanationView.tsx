"use client";

import { useEffect, useState } from "react";
import { usePredictionStore } from "@/store/usePredictionStore";
import { getExplanation, type ExplanationData, type FlareClass } from "@/lib/api";
import { FiCpu, FiAlertTriangle, FiCheckCircle, FiInfo, FiLayers } from "react-icons/fi";

const CLASS_COLORS: Record<FlareClass, string> = {
  Quiet: "#10b981",
  B: "#f59e0b",
  C: "#f97316",
  M: "#ef4444",
  X: "#a855f7",
};

export default function ExplanationView() {
  const { prediction } = usePredictionStore();
  const [explanation, setExplanation] = useState<ExplanationData | null>(null);

  const label = prediction.predicted_label || "M";
  const color = CLASS_COLORS[label] || CLASS_COLORS.M;

  useEffect(() => {
    getExplanation(prediction.predicted_class || 3, prediction.confidence || 0.87).then(setExplanation);
  }, [prediction.predicted_class, prediction.confidence]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/10 pb-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono-code text-amber-400 font-bold mb-1">
            <FiCpu className="w-4 h-4" /> PAGE 9 — EXPLAINABLE AI (XAI) REASONING ENGINE
          </div>
          <h1 className="text-2xl md:text-3xl font-heading font-bold text-white tracking-tight">
            Physical Model Decision Rationale & Feature Attributions
          </h1>
          <p className="text-sm text-gray-400 max-w-3xl mt-1">
            Transparent physical reasoning rules explaining why the `HelioForgeTCN` model classified the current window as <span className="font-bold text-white">{label}-Class</span>.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Main Explanation Rationale Cards */}
        <div className="lg:col-span-8 space-y-4">
          <div
            className="glass-panel p-6 space-y-3 border"
            style={{ borderColor: `${color}40`, backgroundColor: `${color}10` }}
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono-code font-bold uppercase tracking-wider text-gray-300">
                Classification Verdict
              </span>
              <span
                className="px-2.5 py-1 rounded text-xs font-mono-code font-bold"
                style={{ backgroundColor: `${color}25`, color }}
              >
                {label}-Class Solar Event
              </span>
            </div>

            <h2 className="text-lg font-heading font-bold text-white">
              {explanation?.summary || `M-Class flare classification driven by high wavelet energy in decomposition L3 and rapid flux gradient acceleration.`}
            </h2>
          </div>

          {/* Reasoning Rules Attribution List */}
          <div className="glass-panel p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-white/10 pb-2 text-xs font-heading font-semibold text-gray-300">
              <span className="flex items-center gap-1.5">
                <FiLayers className="text-amber-400" /> Physical Feature Decision Triggers
              </span>
              <span className="text-[11px] font-mono-code text-gray-500">
                Top Attributions
              </span>
            </div>

            <div className="space-y-3">
              {(explanation?.rules || []).map((rule, idx) => (
                <div
                  key={idx}
                  className="glass-card p-4 rounded-xl space-y-2 border border-white/5 hover:border-amber-500/30 transition-all"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono-code text-xs font-bold text-amber-300 flex items-center gap-2">
                      <FiAlertTriangle className="text-amber-400" /> {rule.feature}
                    </span>
                    <span className="text-[11px] font-mono-code text-gray-400 bg-white/5 px-2 py-0.5 rounded border border-white/10">
                      Condition: {rule.condition}
                    </span>
                  </div>

                  <p className="text-xs text-gray-300 font-sans leading-relaxed">
                    {rule.reason}
                  </p>

                  <div className="flex items-center justify-between pt-1 text-[10px] font-mono-code text-gray-500">
                    <span>Feature Weight: {Math.round(rule.importance * 100)}%</span>
                    <span>Measured Value: {rule.val}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Sidebar: SHAP & Scientific Credibility Notes */}
        <div className="lg:col-span-4 space-y-4">
          <div className="glass-panel p-5 space-y-3">
            <div className="flex items-center gap-2 text-xs font-heading font-bold text-amber-400 border-b border-white/10 pb-2">
              <FiInfo /> Scientific Verification Notice
            </div>

            <p className="text-xs text-gray-300 leading-relaxed font-sans">
              Rules are derived by tracing activations through the 8 dilated temporal residual blocks and computing integrated gradient feature attributions against background Quiet state baselines.
            </p>

            <div className="p-3 rounded-xl bg-black/40 border border-white/10 text-xs font-mono-code space-y-1">
              <div className="text-gray-400">Interpretability Method:</div>
              <div className="text-emerald-400 font-bold">Rule-based Physics Attribution & Integrated Gradients</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
