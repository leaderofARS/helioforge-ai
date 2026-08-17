"use client";

import { useEffect, useState } from "react";
import SunScene from "@/components/three/SunScene";
import { getEvolutionData, type EvolutionStep, type FlareClass } from "@/lib/api";
import { FiClock, FiPlay, FiPause, FiChevronRight, FiActivity } from "react-icons/fi";

const CLASS_COLORS: Record<FlareClass, string> = {
  Quiet: "#10b981",
  B: "#f59e0b",
  C: "#f97316",
  M: "#ef4444",
  X: "#a855f7",
};

export default function EvolutionView() {
  const [steps, setSteps] = useState<EvolutionStep[]>([]);
  const [activeStepIdx, setActiveStepIdx] = useState(4);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    getEvolutionData().then((res) => {
      if (res?.sequence?.length) {
        setSteps(res.sequence);
        setActiveStepIdx(res.sequence.length - 1);
      }
    });
  }, []);

  useEffect(() => {
    if (!isPlaying || steps.length === 0) return;
    const interval = setInterval(() => {
      setActiveStepIdx((prev) => (prev + 1) % steps.length);
    }, 1800);
    return () => clearInterval(interval);
  }, [isPlaying, steps]);

  const activeStep = steps[activeStepIdx] || {
    timestamp: new Date().toISOString(),
    class: 3,
    label: "M" as FlareClass,
    confidence: 0.87,
    intensity: 5820,
    activeRegionId: "AR-3089",
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/10 pb-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono-code text-amber-400 font-bold mb-1">
            <FiClock className="w-4 h-4" /> PAGE 3 — SOLAR EVOLUTION TIMELINE
          </div>
          <h1 className="text-2xl md:text-3xl font-heading font-bold text-white tracking-tight">
            5-Hour Temporal Flare Evolution Trajectory
          </h1>
          <p className="text-sm text-gray-400 max-w-3xl mt-1">
            Observe the step-by-step physical evolution of solar magnetic activity and flare intensity buildup across sequential observation windows (t-4h → current).
          </p>
        </div>

        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl font-heading font-bold text-xs shadow-lg transition-all ${
            isPlaying
              ? "bg-red-500 hover:bg-red-400 text-white shadow-red-500/20"
              : "bg-amber-500 hover:bg-amber-400 text-black shadow-amber-500/20"
          }`}
        >
          {isPlaying ? <FiPause /> : <FiPlay />}
          {isPlaying ? "Pause Simulation" : "Auto-Play Evolution"}
        </button>
      </div>

      {/* 5-Mini Suns Timeline Sequence Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
        {steps.map((step, idx) => {
          const isSelected = activeStepIdx === idx;
          const color = CLASS_COLORS[step.label || "M"];

          return (
            <button
              key={idx}
              onClick={() => {
                setActiveStepIdx(idx);
                setIsPlaying(false);
              }}
              className={`p-3 rounded-2xl border text-left transition-all ${
                isSelected
                  ? "bg-amber-500/15 border-amber-500/50 shadow-xl shadow-amber-500/10 scale-105"
                  : "bg-white/5 border-white/5 opacity-70 hover:opacity-100 hover:bg-white/10"
              }`}
            >
              <div className="flex items-center justify-between text-[10px] font-mono-code text-gray-400 mb-2">
                <span>{idx === steps.length - 1 ? "CURRENT" : `t - ${steps.length - 1 - idx}h`}</span>
                <span className="font-bold" style={{ color }}>
                  {step.label}-Class
                </span>
              </div>

              {/* Mini 3D Sun */}
              <div className="h-28 rounded-xl overflow-hidden bg-black/50 border border-white/5 my-2">
                <SunScene mini />
              </div>

              <div className="space-y-1 text-[10px] font-mono-code">
                <div className="flex justify-between text-gray-400">
                  <span>Confidence:</span>
                  <span className="text-gray-200 font-bold">{Math.round(step.confidence * 100)}%</span>
                </div>
                {step.intensity && (
                  <div className="flex justify-between text-gray-400">
                    <span>Flux:</span>
                    <span className="text-amber-400 font-bold">{step.intensity.toLocaleString()} c/s</span>
                  </div>
                )}
              </div>
            </button>
          );
        })}
      </div>

      {/* Active Evolution Frame Detail Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-8">
          <SunScene className="h-[500px] shadow-2xl" />
        </div>

        <div className="lg:col-span-4 space-y-4">
          <div className="glass-panel p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-white/10 pb-2">
              <span className="text-xs font-heading font-bold text-amber-400 flex items-center gap-1.5">
                <FiActivity /> Evolution Frame Telemetry
              </span>
              <span
                className="px-2 py-0.5 rounded text-xs font-mono-code font-bold"
                style={{
                  backgroundColor: `${CLASS_COLORS[activeStep.label]}25`,
                  color: CLASS_COLORS[activeStep.label],
                }}
              >
                {activeStep.label}-Class Flare
              </span>
            </div>

            <div className="space-y-3 text-xs font-mono-code">
              <div>
                <span className="text-gray-500 block text-[10px]">Observation Timestamp</span>
                <span className="text-gray-200 font-bold">{activeStep.timestamp}</span>
              </div>

              <div>
                <span className="text-gray-500 block text-[10px]">Active Region Hotspot</span>
                <span className="text-amber-300 font-bold">{activeStep.activeRegionId || "AR-3089"}</span>
              </div>

              <div>
                <span className="text-gray-500 block text-[10px]">Peak Hard/Soft X-Ray Flux</span>
                <span className="text-2xl font-heading font-bold text-white">
                  {(activeStep.intensity || 5820).toLocaleString()}{" "}
                  <span className="text-xs text-gray-400 font-mono-code font-normal">COUNTS/sec</span>
                </span>
              </div>

              <div>
                <span className="text-gray-500 block text-[10px]">AI Classification Confidence</span>
                <div className="w-full bg-white/5 h-2 rounded-full overflow-hidden mt-1 p-0.5 border border-white/10">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${Math.round(activeStep.confidence * 100)}%`,
                      backgroundColor: CLASS_COLORS[activeStep.label],
                    }}
                  />
                </div>
                <span className="text-right text-[10px] text-gray-400 block mt-0.5 font-bold">
                  {Math.round(activeStep.confidence * 100)}% Certainty
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
