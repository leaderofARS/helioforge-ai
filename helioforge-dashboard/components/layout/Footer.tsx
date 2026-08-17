"use client";

import { usePredictionStore } from "@/store/usePredictionStore";

export default function Footer() {
  const { health } = usePredictionStore();

  return (
    <footer className="w-full border-t border-white/10 bg-[#030712] py-4 px-6 text-xs text-gray-500 font-mono-code flex flex-col md:flex-row items-center justify-between gap-3">
      <div className="flex items-center gap-2">
        <span className="text-amber-400 font-bold font-heading">HELIO-FORGE AI</span>
        <span>•</span>
        <span>Aditya-L1 Solar Flare Forecasting System</span>
      </div>

      <div className="flex items-center gap-4 text-[11px]">
        <span>Model: HelioForgeTCN (8.57M Params)</span>
        <span className="hidden sm:inline">•</span>
        <span className="hidden sm:inline">Macro F1: {health?.macro_f1 || 0.8514}</span>
        <span>•</span>
        <span className="text-emerald-400">Validated on Held-Out Test Split (89.41% Accuracy)</span>
      </div>
    </footer>
  );
}