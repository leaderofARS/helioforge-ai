"use client";

import { usePredictionStore } from "@/store/usePredictionStore";
import { useState } from "react";
import { FiLayers, FiSearch, FiSliders, FiCheckCircle, FiAlertCircle } from "react-icons/fi";

const FEATURE_DESCRIPTIONS: Record<string, string> = {
  soft_mean: "Mean soft X-ray count rate across window",
  soft_std: "Standard deviation of soft X-ray signal",
  soft_max: "Peak soft X-ray count rate",
  soft_min: "Baseline minimum soft X-ray count rate",
  soft_range: "Dynamic range (Max − Min)",
  soft_skew: "Statistical asymmetry of photon distribution",
  soft_kurtosis: "Impulsiveness & peak sharpness",
  soft_energy: "Integrated soft thermal energy",
  soft_entropy: "Shannon entropy of photon flux",
  soft_peak_count: "Detected micro-burst peak density",
  hard_mean: "Mean hard X-ray count rate",
  hard_std: "Hard X-ray variability",
  hard_max: "Peak non-thermal hard X-ray count rate",
  hard_energy: "Integrated non-thermal hard X-ray energy",
  ratio_hard_soft: "Spectral hardness ratio (Hard / Soft)",
  roll_mean_16: "16-step rolling mean intensity",
  roll_std_16: "16-step rolling standard deviation",
  roll_mean_64: "64-step rolling mean intensity",
  roll_std_64: "64-step rolling standard deviation",
  wavelet_energy_L1: "Wavelet energy at scale L1 (fine detail)",
  wavelet_energy_L2: "Wavelet energy at scale L2",
  wavelet_energy_L3: "Wavelet energy at scale L3 (MHD waves)",
  wavelet_energy_L4: "Wavelet energy at scale L4 (coarse envelope)",
  spectral_entropy: "Frequency-domain spectral entropy",
  dominant_freq: "Dominant oscillation frequency (Hz)",
  rise_rate: "Thermal flux growth rate (+c/s)",
  decay_rate: "Gradual phase decay rate (-c/s)",
  delta_mean: "Mean delta between window halves",
  log_energy: "Log-transformed integrated energy",
  zero_crossing_rate: "Oscillation zero-crossing rate",
  temporal_gradient: "First derivative temporal gradient",
  channel_correlation: "HEL1OS ↔ SoLEXS cross-correlation",
};

export default function FeaturesView() {
  const { prediction } = usePredictionStore();
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<"all" | "normal" | "anomalous">("all");
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null);

  const features = prediction.features || {};
  const featureEntries = Object.entries(features);

  const filteredFeatures = featureEntries.filter(([key]) => {
    const matchesSearch = key.toLowerCase().includes(search.toLowerCase()) || (FEATURE_DESCRIPTIONS[key] || "").toLowerCase().includes(search.toLowerCase());
    return matchesSearch;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/10 pb-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono-code text-purple-400 font-bold mb-1">
            <FiLayers className="w-4 h-4" /> PAGE 8 — 32 ENGINEERED PHYSICAL FEATURES
          </div>
          <h1 className="text-2xl md:text-3xl font-heading font-bold text-white tracking-tight">
            Multivariate Physics Feature Matrix (32 Features x 512 Timesteps)
          </h1>
          <p className="text-sm text-gray-400 max-w-3xl mt-1">
            Engineered temporal, statistical, spectral, and wavelet domain features extracted per second and fed into the 8-block dilated TCN encoder.
          </p>
        </div>

        {/* Search & Filter */}
        <div className="flex items-center gap-2">
          <div className="relative">
            <FiSearch className="absolute left-3 top-3 text-gray-400 w-3.5 h-3.5" />
            <input
              type="text"
              placeholder="Search 32 features..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9 pr-3 py-1.5 rounded-xl border border-white/10 bg-white/5 text-xs font-mono-code text-white focus:outline-none focus:border-amber-500/50 w-52"
            />
          </div>
        </div>
      </div>

      {/* 32 Feature Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {filteredFeatures.map(([key, val]) => {
          const isSelected = selectedFeature === key;
          const description = FEATURE_DESCRIPTIONS[key] || "Physical temporal feature metric";
          const isElevated = Math.abs(val) > 100 || key.includes("max") || key.includes("L3");

          return (
            <div
              key={key}
              onClick={() => setSelectedFeature(key)}
              className={`glass-card p-4 rounded-2xl cursor-pointer transition-all ${
                isSelected
                  ? "border-amber-500/50 bg-amber-500/10 shadow-lg shadow-amber-500/10"
                  : "hover:border-white/20"
              }`}
            >
              <div className="flex items-center justify-between border-b border-white/10 pb-2">
                <span className="font-mono-code text-xs font-bold text-amber-300 truncate max-w-[170px]">
                  {key}
                </span>
                {isElevated ? (
                  <span className="flex items-center gap-1 text-[10px] font-mono-code text-amber-400 font-bold bg-amber-500/10 px-1.5 py-0.5 rounded border border-amber-500/20">
                    <FiAlertCircle className="w-3 h-3" /> Elevated
                  </span>
                ) : (
                  <span className="flex items-center gap-1 text-[10px] font-mono-code text-emerald-400 font-bold bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">
                    <FiCheckCircle className="w-3 h-3" /> Normal
                  </span>
                )}
              </div>

              <div className="pt-2 flex items-baseline justify-between">
                <span className="text-xl font-mono-code font-bold text-white">
                  {typeof val === "number" ? val.toLocaleString() : val}
                </span>
              </div>

              <p className="text-[11px] text-gray-400 font-mono-code mt-1.5 line-clamp-2">
                {description}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
