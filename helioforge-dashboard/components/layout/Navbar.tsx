"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePredictionStore, type ActiveSection } from "@/store/usePredictionStore";
import {
  FiSun,
  FiActivity,
  FiBarChart2,
  FiUploadCloud,
  FiCpu,
  FiLayers,
  FiPlayCircle,
  FiRadio,
  FiClock,
  FiShield,
} from "react-icons/fi";

const NAVIGATION_ITEMS: Array<{ id: ActiveSection; label: string; icon: React.ComponentType<{ className?: string }> }> = [
  { id: "control", label: "Mission Control", icon: FiShield },
  { id: "sun", label: "3D Observatory", icon: FiSun },
  { id: "evolution", label: "Solar Evolution", icon: FiClock },
  { id: "prediction", label: "AI Prediction", icon: FiCpu },
  { id: "intensity", label: "RGB Intensity", icon: FiRadio },
  { id: "signals", label: "Signal Analysis", icon: FiActivity },
  { id: "features", label: "32 Features", icon: FiLayers },
  { id: "upload", label: "Dataset Explorer", icon: FiUploadCloud },
  { id: "performance", label: "Performance", icon: FiBarChart2 },
  { id: "animation", label: "3D Simulation", icon: FiPlayCircle },
];

export default function Navbar() {
  const { activeSection, setActiveSection, demoIndex, fetchDemo, health, fetchHealth } = usePredictionStore();
  const [utcTime, setUtcTime] = useState<string>("");

  useEffect(() => {
    fetchHealth();
    fetchDemo(0);
    const updateTime = () => {
      const now = new Date();
      setUtcTime(now.toISOString().substring(11, 19) + " UTC");
    };
    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, [fetchHealth, fetchDemo]);

  return (
    <header className="sticky top-0 z-40 w-full border-b border-white/10 bg-[#030712]/90 backdrop-blur-xl">
      {/* Top Telemetry Banner */}
      <div className="flex items-center justify-between border-b border-white/5 bg-[#070b14] px-4 py-1.5 text-[11px] font-mono-code text-gray-400">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5 text-amber-400 font-bold">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500" />
            </span>
            ADITYA-L1 MISSION CONTROL
          </span>
          <span className="hidden md:inline text-gray-600">|</span>
          <span className="hidden md:inline">HEL1OS + SoLEXS Spacecraft Telemetry</span>
          <span className="hidden lg:inline text-gray-600">|</span>
          <span className="hidden lg:inline text-emerald-400">
            Model: HelioForgeTCN (8.57M Params • Macro F1: 85.14%)
          </span>
        </div>

        <div className="flex items-center gap-4">
          {/* Quick Demo Sample Selector */}
          <div className="flex items-center gap-1 text-gray-400">
            <span className="hidden sm:inline text-[10px] text-gray-500">Sample:</span>
            {[0, 1, 2].map((idx) => (
              <button
                key={idx}
                onClick={() => fetchDemo(idx)}
                className={`px-1.5 py-0.5 rounded text-[10px] transition-colors ${
                  demoIndex === idx
                    ? "bg-amber-500/20 text-amber-300 border border-amber-500/40 font-bold"
                    : "hover:bg-white/5 text-gray-400"
                }`}
              >
                #{idx + 1}
              </button>
            ))}
          </div>

          <span className="text-gray-600">|</span>

          {/* Backend Status */}
          <span className="flex items-center gap-1">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                health?.status === "ok" ? "bg-emerald-400" : "bg-amber-400"
              }`}
            />
            <span className="text-gray-300">
              {health?.status === "ok" ? "API ONLINE" : "STANDALONE"}
            </span>
          </span>

          <span className="text-gray-600">|</span>
          <span className="text-amber-400 font-bold">{utcTime || "00:00:00 UTC"}</span>
        </div>
      </div>

      {/* Main Navigation Bar */}
      <div className="flex items-center justify-between px-4 py-2.5 max-w-[1700px] mx-auto">
        {/* Brand Logo */}
        <Link
          href="/"
          onClick={() => setActiveSection("control")}
          className="flex items-center gap-3 group text-decoration-none"
        >
          <div className="relative flex items-center justify-center w-9 h-9 rounded-xl bg-gradient-to-br from-amber-500 to-orange-600 p-0.5 shadow-lg shadow-orange-500/20 group-hover:scale-105 transition-transform">
            <div className="w-full h-full bg-[#070b14] rounded-[10px] flex items-center justify-center">
              <FiSun className="w-5 h-5 text-amber-400 group-hover:rotate-45 transition-transform duration-500" />
            </div>
          </div>
          <div>
            <span className="font-heading font-bold text-lg tracking-wider text-white flex items-center gap-1.5">
              HELIO-FORGE <span className="text-amber-400 text-xs font-mono-code px-1.5 py-0.5 rounded bg-amber-500/10 border border-amber-500/20">AI</span>
            </span>
            <span className="text-[10px] font-mono-code text-gray-400 block -mt-1 tracking-tight">
              Solar Flare Intelligence System
            </span>
          </div>
        </Link>

        {/* Scrollable Navigation Links */}
        <nav className="flex items-center gap-1 overflow-x-auto no-scrollbar py-1 px-2">
          {NAVIGATION_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = activeSection === item.id;
            return (
              <Link
                key={item.id}
                href={item.id === "control" ? "/" : `/${item.id}`}
                onClick={() => setActiveSection(item.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all whitespace-nowrap ${
                  isActive
                    ? "bg-amber-500/15 text-amber-300 border border-amber-500/30 shadow-md shadow-amber-500/5 font-semibold"
                    : "text-gray-400 hover:text-gray-200 hover:bg-white/5"
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? "text-amber-400" : "text-gray-400"}`} />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
