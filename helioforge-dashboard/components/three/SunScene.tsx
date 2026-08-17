"use client";

import { useRef, useMemo, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Stars, Html } from "@react-three/drei";
import * as THREE from "three";
import { usePredictionStore, type WavelengthMode } from "@/store/usePredictionStore";
import { type ActiveRegion, type FlareClass } from "@/lib/api";
import {
  FiCompass,
  FiEye,
  FiZap,
  FiRotateCw,
  FiMaximize2,
  FiLayers,
  FiActivity,
} from "react-icons/fi";

const CLASS_COLORS: Record<FlareClass, string> = {
  Quiet: "#10b981",
  B: "#f59e0b",
  C: "#f97316",
  M: "#ef4444",
  X: "#a855f7",
};

const WAVELENGTH_CONFIG: Record<
  WavelengthMode,
  { name: string; desc: string; coreColor: string; emissiveColor: string; roughness: number }
> = {
  "171": {
    name: "AIA 171Å",
    desc: "Fe IX/X • Quiet Corona & Upper Transition Region",
    coreColor: "#ea580c",
    emissiveColor: "#f97316",
    roughness: 0.45,
  },
  "304": {
    name: "AIA 304Å",
    desc: "He II • Chromosphere & Transition Filaments",
    coreColor: "#991b1b",
    emissiveColor: "#dc2626",
    roughness: 0.6,
  },
  "131": {
    name: "AIA 131Å",
    desc: "Fe VIII/XXI • Extreme Flaring Plasma (>10 MK)",
    coreColor: "#0891b2",
    emissiveColor: "#06b6d4",
    roughness: 0.35,
  },
  hmi: {
    name: "HMI Magnetogram",
    desc: "Photospheric Line-of-Sight Magnetic Field Polarity",
    coreColor: "#334155",
    emissiveColor: "#64748b",
    roughness: 0.8,
  },
  rgb: {
    name: "Composite RGB",
    desc: "Hard X-Ray (R) • Soft X-Ray (G) • UV (B)",
    coreColor: "#d97706",
    emissiveColor: "#eab308",
    roughness: 0.5,
  },
};

function latLonToVector3(lat: number, lon: number, radius = 1.02): THREE.Vector3 {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);
  const x = -(radius * Math.sin(phi) * Math.cos(theta));
  const z = radius * Math.sin(phi) * Math.sin(theta);
  const y = radius * Math.cos(phi);
  return new THREE.Vector3(x, y, z);
}

// Shader Material for Solar Noise Surface
function SolarSphere({ wavelength, rotationSpeed }: { wavelength: WavelengthMode; rotationSpeed: number }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const config = WAVELENGTH_CONFIG[wavelength];
  const { prediction } = usePredictionStore();
  const flareClass = prediction.predicted_label || "M";
  const statusColor = CLASS_COLORS[flareClass];

  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.12 * rotationSpeed;
    }
  });

  return (
    <group>
      {/* Primary Solar Core */}
      <mesh ref={meshRef}>
        <sphereGeometry args={[1, 64, 64]} />
        <meshStandardMaterial
          color={config.coreColor}
          emissive={config.emissiveColor}
          emissiveIntensity={flareClass === "X" ? 1.4 : flareClass === "M" ? 1.1 : 0.75}
          roughness={config.roughness}
          metalness={0.1}
        />
      </mesh>

      {/* Atmospheric Coronal Shell */}
      <mesh scale={1.12}>
        <sphereGeometry args={[1, 48, 48]} />
        <meshBasicMaterial
          color={statusColor}
          transparent
          opacity={flareClass === "X" ? 0.28 : flareClass === "M" ? 0.22 : 0.14}
          side={THREE.BackSide}
          blending={THREE.AdditiveBlending}
        />
      </mesh>

      {/* Outer Glow Shield */}
      <mesh scale={1.25}>
        <sphereGeometry args={[1, 32, 32]} />
        <meshBasicMaterial
          color={config.emissiveColor}
          transparent
          opacity={0.06}
          side={THREE.BackSide}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
    </group>
  );
}

// Coronal Magnetic Loop Streamers
function MagneticLoops({ show }: { show: boolean }) {
  const { prediction } = usePredictionStore();
  const regions = prediction.active_regions || [];

  const curves = useMemo(() => {
    if (!show || regions.length < 2) return [];
    const list: THREE.CatmullRomCurve3[] = [];
    for (let i = 0; i < regions.length - 1; i++) {
      const p1 = latLonToVector3(regions[i].lat, regions[i].lon, 1.01);
      const p2 = latLonToVector3(regions[i + 1].lat, regions[i + 1].lon, 1.01);
      const mid = p1.clone().add(p2).multiplyScalar(0.5).normalize().multiplyScalar(1.35);
      list.push(new THREE.CatmullRomCurve3([p1, mid, p2]));
    }
    return list;
  }, [regions, show]);

  if (!show || curves.length === 0) return null;

  return (
    <group>
      {curves.map((curve, idx) => (
        <mesh key={idx}>
          <tubeGeometry args={[curve, 32, 0.008, 8, false]} />
          <meshBasicMaterial color="#f97316" transparent opacity={0.65} blending={THREE.AdditiveBlending} />
        </mesh>
      ))}
    </group>
  );
}

// CME Particle Ejection Streamer
function CMEParticleStream({ show }: { show: boolean }) {
  const { prediction } = usePredictionStore();
  const isHighEnergy = prediction.predicted_class >= 3;
  const count = isHighEnergy ? 600 : 150;

  const points = useMemo(() => {
    const p = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const u = Math.random();
      const v = Math.random();
      const theta = u * 2.0 * Math.PI;
      const phi = Math.acos(2.0 * v - 1.0);
      const r = 1.08 + Math.random() * 1.5;
      p[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      p[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      p[i * 3 + 2] = r * Math.cos(phi);
    }
    return p;
  }, [count]);

  const pointsRef = useRef<THREE.Points>(null);

  useFrame((_, delta) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y += delta * 0.05;
      pointsRef.current.rotation.x += delta * 0.02;
    }
  });

  if (!show) return null;

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[points, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.025}
        color={isHighEnergy ? "#ef4444" : "#f97316"}
        transparent
        opacity={0.7}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

// Interactive Active Region Beacons on Surface
function ActiveRegionMarkers({
  show,
  selectedRegion,
  onSelect,
}: {
  show: boolean;
  selectedRegion: ActiveRegion | null;
  onSelect: (region: ActiveRegion) => void;
}) {
  const { prediction } = usePredictionStore();
  const regions = prediction.active_regions || [];

  if (!show) return null;

  return (
    <group>
      {regions.map((region) => {
        const pos = latLonToVector3(region.lat, region.lon, 1.03);
        const isSelected = selectedRegion?.id === region.id;
        const color = CLASS_COLORS[region.label || (region.class === 4 ? "X" : region.class === 3 ? "M" : region.class === 2 ? "C" : region.class === 1 ? "B" : "Quiet")];

        return (
          <group key={region.id} position={pos}>
            {/* Beacon Sphere */}
            <mesh
              onClick={(e) => {
                e.stopPropagation();
                onSelect(region);
              }}
            >
              <sphereGeometry args={[isSelected ? 0.045 : 0.032, 16, 16]} />
              <meshStandardMaterial color={color} emissive={color} emissiveIntensity={isSelected ? 3.0 : 1.8} />
            </mesh>

            {/* Pulsing Beacon Ring */}
            <mesh scale={isSelected ? 1.6 : 1.2}>
              <ringGeometry args={[0.03, 0.045, 32]} />
              <meshBasicMaterial color={color} transparent opacity={isSelected ? 0.8 : 0.4} side={THREE.DoubleSide} />
            </mesh>

            {/* 3D Label Callout */}
            {isSelected && (
              <Html distanceFactor={4} position={[0, 0.08, 0]} center>
                <div className="pointer-events-none rounded-lg border border-amber-500/40 bg-[#070b14]/90 p-2.5 shadow-2xl backdrop-blur-md text-xs min-w-[140px]">
                  <div className="flex items-center justify-between font-mono-code font-bold text-amber-400">
                    <span>{region.id}</span>
                    <span
                      className="px-1.5 py-0.5 rounded text-[10px]"
                      style={{ backgroundColor: `${color}25`, color }}
                    >
                      {region.label || "Region"}
                    </span>
                  </div>
                  <div className="mt-1 text-[11px] text-gray-300 font-mono-code">
                    Lat: {region.lat > 0 ? `+${region.lat}` : region.lat}°
                    <br />
                    Lon: {region.lon > 0 ? `+${region.lon}` : region.lon}°
                  </div>
                  {region.intensity && (
                    <div className="mt-1 text-[10px] text-amber-200/80 font-mono-code">
                      {region.intensity.toLocaleString()} c/s
                    </div>
                  )}
                </div>
              </Html>
            )}
          </group>
        );
      })}
    </group>
  );
}

export default function SunScene({
  mini = false,
  className = "",
}: {
  mini?: boolean;
  className?: string;
}) {
  const {
    prediction,
    wavelength,
    setWavelength,
    selectedRegion,
    setSelectedRegion,
  } = usePredictionStore();

  const [showRegions, setShowRegions] = useState(true);
  const [showLoops, setShowLoops] = useState(true);
  const [showParticles, setShowParticles] = useState(true);
  const [rotationSpeed, setRotationSpeed] = useState(1);

  const activeClass = prediction.predicted_label || "M";

  return (
    <div
      className={`relative overflow-hidden rounded-2xl border border-white/10 bg-[#030712] ${
        mini ? "h-48 w-full" : "h-[620px] w-full"
      } ${className}`}
    >
      {/* Background Starfield & Scene Canvas */}
      <Canvas camera={{ position: [0, 0, 2.9], fov: 45 }}>
        <ambientLight intensity={0.25} />
        <pointLight position={[5, 5, 5]} intensity={1.8} color="#fff" />
        <pointLight position={[-5, -5, -2]} intensity={0.5} color="#f97316" />
        <Stars radius={100} depth={50} count={mini ? 400 : 3500} factor={4} fade speed={1} />

        <SolarSphere wavelength={wavelength} rotationSpeed={rotationSpeed} />
        <MagneticLoops show={showLoops && !mini} />
        <CMEParticleStream show={showParticles && !mini} />
        <ActiveRegionMarkers
          show={showRegions}
          selectedRegion={selectedRegion}
          onSelect={setSelectedRegion}
        />

        <OrbitControls
          enableZoom={!mini}
          autoRotate={mini}
          autoRotateSpeed={0.6}
          minDistance={1.8}
          maxDistance={5.5}
        />
      </Canvas>

      {/* Full Observatory Overlay HUD */}
      {!mini && (
        <>
          {/* Top Left: Active Observatory Wavelength Selector */}
          <div className="absolute top-4 left-4 z-10 flex flex-col gap-2">
            <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-[#0b0f19]/80 p-1.5 backdrop-blur-md">
              {(["171", "304", "131", "hmi", "rgb"] as WavelengthMode[]).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setWavelength(mode)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono-code transition-all ${
                    wavelength === mode
                      ? "bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-lg shadow-amber-500/10 font-bold"
                      : "text-gray-400 hover:text-gray-200 hover:bg-white/5"
                  }`}
                >
                  {WAVELENGTH_CONFIG[mode].name.split(" ")[1] || mode.toUpperCase()}
                </button>
              ))}
            </div>

            <div className="rounded-lg border border-white/5 bg-[#0b0f19]/60 px-3 py-1.5 text-[11px] text-gray-400 font-mono-code backdrop-blur-md">
              <span className="text-amber-400 font-bold">{WAVELENGTH_CONFIG[wavelength].name}</span> •{" "}
              {WAVELENGTH_CONFIG[wavelength].desc}
            </div>
          </div>

          {/* Top Right: Observatory Layer & Simulation Controls */}
          <div className="absolute top-4 right-4 z-10 flex flex-col gap-2">
            <div className="rounded-xl border border-white/10 bg-[#0b0f19]/85 p-3 backdrop-blur-md text-xs space-y-2.5 min-w-[200px]">
              <div className="flex items-center justify-between text-gray-300 font-heading font-semibold text-xs border-b border-white/10 pb-1.5">
                <span className="flex items-center gap-1.5">
                  <FiLayers className="text-amber-400" /> Observatory Layers
                </span>
              </div>

              <label className="flex items-center justify-between text-gray-300 cursor-pointer hover:text-white">
                <span className="flex items-center gap-1.5">
                  <FiCompass className="text-emerald-400" /> Active Regions
                </span>
                <input
                  type="checkbox"
                  checked={showRegions}
                  onChange={(e) => setShowRegions(e.target.checked)}
                  className="accent-amber-500 rounded cursor-pointer"
                />
              </label>

              <label className="flex items-center justify-between text-gray-300 cursor-pointer hover:text-white">
                <span className="flex items-center gap-1.5">
                  <FiActivity className="text-amber-400" /> Coronal Loops
                </span>
                <input
                  type="checkbox"
                  checked={showLoops}
                  onChange={(e) => setShowLoops(e.target.checked)}
                  className="accent-amber-500 rounded cursor-pointer"
                />
              </label>

              <label className="flex items-center justify-between text-gray-300 cursor-pointer hover:text-white">
                <span className="flex items-center gap-1.5">
                  <FiZap className="text-purple-400" /> Solar Wind / CME
                </span>
                <input
                  type="checkbox"
                  checked={showParticles}
                  onChange={(e) => setShowParticles(e.target.checked)}
                  className="accent-amber-500 rounded cursor-pointer"
                />
              </label>

              <div className="pt-1 border-t border-white/5 space-y-1">
                <div className="flex justify-between text-[11px] text-gray-400 font-mono-code">
                  <span className="flex items-center gap-1">
                    <FiRotateCw /> Rotation
                  </span>
                  <span>{rotationSpeed.toFixed(1)}x</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="3"
                  step="0.25"
                  value={rotationSpeed}
                  onChange={(e) => setRotationSpeed(parseFloat(e.target.value))}
                  className="w-full accent-amber-500 h-1 bg-white/10 rounded cursor-pointer"
                />
              </div>
            </div>
          </div>

          {/* Bottom Left: Active Region Telemetry Focus HUD */}
          {selectedRegion && (
            <div className="absolute bottom-4 left-4 z-10 rounded-xl border border-amber-500/30 bg-[#0b0f19]/90 p-3.5 text-xs backdrop-blur-md max-w-xs shadow-2xl space-y-2 animate-in fade-in slide-in-from-bottom-2">
              <div className="flex items-center justify-between border-b border-white/10 pb-1.5 font-heading">
                <span className="font-bold text-amber-400 flex items-center gap-1.5">
                  <FiEye /> {selectedRegion.id}
                </span>
                <span
                  className="px-2 py-0.5 rounded text-[10px] font-mono-code font-bold"
                  style={{
                    backgroundColor: `${CLASS_COLORS[selectedRegion.label || "M"]}25`,
                    color: CLASS_COLORS[selectedRegion.label || "M"],
                  }}
                >
                  {selectedRegion.label || "Active Region"}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-[11px] font-mono-code text-gray-300">
                <div>
                  <span className="text-gray-500 block text-[10px]">Latitude</span>
                  {selectedRegion.lat > 0 ? `+${selectedRegion.lat}` : selectedRegion.lat}°
                </div>
                <div>
                  <span className="text-gray-500 block text-[10px]">Longitude</span>
                  {selectedRegion.lon > 0 ? `+${selectedRegion.lon}` : selectedRegion.lon}°
                </div>
                {selectedRegion.intensity && (
                  <div>
                    <span className="text-gray-500 block text-[10px]">Peak Intensity</span>
                    {selectedRegion.intensity.toLocaleString()} c/s
                  </div>
                )}
                {selectedRegion.magneticField && (
                  <div>
                    <span className="text-gray-500 block text-[10px]">Mag Field</span>
                    {selectedRegion.magneticField}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Bottom Right: Live Solar State Status Badge */}
          <div className="absolute bottom-4 right-4 z-10 flex items-center gap-3 rounded-xl border border-white/10 bg-[#0b0f19]/80 px-4 py-2 text-xs backdrop-blur-md font-mono-code">
            <span className="text-gray-400">Current Status:</span>
            <span
              className="flex items-center gap-1.5 font-bold px-2 py-0.5 rounded"
              style={{
                backgroundColor: `${CLASS_COLORS[activeClass]}25`,
                color: CLASS_COLORS[activeClass],
              }}
            >
              <span
                className="w-2 h-2 rounded-full animate-ping"
                style={{ backgroundColor: CLASS_COLORS[activeClass] }}
              />
              {activeClass}-Class Solar Alert
            </span>
          </div>
        </>
      )}
    </div>
  );
}
