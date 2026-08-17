"use client";

import { useState } from "react";
import SunScene from "@/components/three/SunScene";
import { type FlareClass } from "@/lib/api";
import { FiPlay, FiPause, FiRotateCw, FiZap, FiActivity } from "react-icons/fi";

const SIMULATION_STATES: Array<{ class: FlareClass; desc: string; temp: string; cme: string }> = [
  { class: "Quiet", desc: "Background Solar State", temp: "~1.5 MK", cme: "None" },
  { class: "B", desc: "Background Micro-Burst", temp: "~3.0 MK", cme: "Faint Streamers" },
  { class: "C", desc: "Minor Thermal Flare", temp: "~8.0 MK", cme: "Moderate Plasma Expansion" },
  { class: "M", desc: "Strong Magnetic Reconnection", temp: "~18.0 MK", cme: "Prominent CME Particle Ejection" },
  { class: "X", desc: "Extreme Solar Flare & Radio Blackout", temp: ">30.0 MK", cme: "Massive CME Shockwave & Relativistic Particles" },
];

export default function AnimationView() {
  const [activeIdx, setActiveIdx] = useState<number>(3);

  const current = SIMULATION_STATES[activeIdx];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/10 pb-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono-code text-purple-400 font-bold mb-1">
            <FiPlay className="w-4 h-4" /> PAGE 15 — 3D FLARE EVOLUTION SIMULATOR
          </div>
          <h1 className="text-2xl md:text-3xl font-heading font-bold text-white tracking-tight">
            Flare Lifecycle & Coronal Mass Ejection (CME) Particle Simulator
          </h1>
          <p className="text-sm text-gray-400 max-w-3xl mt-1">
            Simulate 3D physical flare growth across energy classes (Quiet → B → C → M → X) with expanding coronal plasma, thermal flare-ups, and particle ejections.
          </p>
        </div>

        {/* State Jump Selector */}
        <div className="flex items-center gap-1.5 bg-[#0b0f19] p-1.5 rounded-xl border border-white/10">
          {SIMULATION_STATES.map((s, idx) => (
            <button
              key={s.class}
              onClick={() => setActiveIdx(idx)}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono-code transition-all ${
                activeIdx === idx
                  ? "bg-purple-500/25 text-purple-300 border border-purple-500/40 font-bold shadow-lg shadow-purple-500/10"
                  : "text-gray-400 hover:text-gray-200"
              }`}
            >
              {s.class}-Class
            </button>
          ))}
        </div>
      </div>

      {/* Main 3D Simulation Observatory Canvas */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-8">
          <SunScene className="h-[600px] shadow-2xl" />
        </div>

        <div className="lg:col-span-4 space-y-4">
          <div className="glass-panel p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-white/10 pb-2 text-xs font-heading font-semibold text-gray-300">
              <span className="flex items-center gap-1.5">
                <FiZap className="text-purple-400" /> Simulation State Telemetry
              </span>
              <span className="text-[10px] font-mono-code text-purple-400 font-bold">
                {current.class}-Class Mode
              </span>
            </div>

            <div className="space-y-3 text-xs font-mono-code">
              <div>
                <span className="text-gray-500 block text-[10px]">Physical Phase</span>
                <span className="text-white font-bold text-sm">{current.desc}</span>
              </div>

              <div>
                <span className="text-gray-500 block text-[10px]">Plasma Temperature</span>
                <span className="text-amber-400 font-bold">{current.temp}</span>
              </div>

              <div>
                <span className="text-gray-500 block text-[10px]">Coronal Mass Ejection (CME)</span>
                <span className="text-purple-300 font-bold">{current.cme}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
