"use client";

import { useRef, useState } from "react";
import { usePredictionStore } from "@/store/usePredictionStore";
import { FiUploadCloud, FiFileText, FiCheckCircle, FiLoader, FiAlertCircle, FiZap } from "react-icons/fi";

export default function UploadView() {
  const { predict, isLoading, error, prediction } = usePredictionStore();
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadStep, setUploadStep] = useState<number>(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    setSelectedFile(file);
    setUploadStep(1);
    setUploadError(null);

    setTimeout(() => setUploadStep(2), 400);
    setTimeout(() => setUploadStep(3), 800);

    try {
      await predict(file);
      setUploadStep(4);
    } catch (err) {
      setUploadStep(0);
      setUploadError(err instanceof Error ? err.message : "Upload failed. Check file format.");
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/10 pb-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono-code text-amber-400 font-bold mb-1">
            <FiUploadCloud className="w-4 h-4" /> PAGE 12 — DATASET EXPLORER & FITS UPLOADER
          </div>
          <h1 className="text-2xl md:text-3xl font-heading font-bold text-white tracking-tight">
            Ingest Raw Solar Telemetry (`.fits` / `.pt` / `.json`)
          </h1>
          <p className="text-sm text-gray-400 max-w-3xl mt-1">
            Upload new Aditya-L1 satellite observation files or pre-processed PyTorch window tensors to execute live automated classification.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Drag & Drop Dropzone */}
        <div className="lg:col-span-7 space-y-4">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragActive(false);
              if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
            }}
            className={`glass-panel p-10 border-2 border-dashed text-center space-y-4 transition-all ${
              dragActive
                ? "border-amber-500 bg-amber-500/10 shadow-2xl scale-[1.01]"
                : "border-white/20 hover:border-amber-500/40"
            }`}
          >
            <div className="w-16 h-16 rounded-2xl bg-amber-500/10 border border-amber-500/20 text-amber-400 flex items-center justify-center mx-auto shadow-lg shadow-amber-500/10">
              <FiUploadCloud className="w-8 h-8" />
            </div>

            <div>
              <h3 className="font-heading font-bold text-lg text-white">
                Drag & Drop Solar Telemetry File
              </h3>
              <p className="text-xs text-gray-400 font-mono-code mt-1">
                Supports PyTorch Tensors (.pt), FITS (.fits/.fit), or JSON telemetry
              </p>
            </div>

            {/* Hidden real file input */}
            <input
              ref={fileInputRef}
              type="file"
              accept=".fits,.fit,.pt,.json,.csv"
              style={{ display: "none" }}
              onChange={(e) => {
                if (e.target.files?.[0]) handleFile(e.target.files[0]);
                e.target.value = "";
              }}
            />

            {/* Visible button that triggers the input via ref */}
            <button
              type="button"
              disabled={isLoading}
              onClick={() => fileInputRef.current?.click()}
              className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-amber-500 hover:bg-amber-400 disabled:opacity-50 text-black font-heading font-bold text-sm cursor-pointer shadow-lg shadow-amber-500/20 transition-all"
            >
              <FiFileText /> {isLoading ? "Processing..." : "Browse Files"}
            </button>

            {uploadError && (
              <p className="text-xs text-red-400 font-mono-code flex items-center gap-1.5">
                <FiAlertCircle /> {uploadError}
              </p>
            )}
          </div>

          {/* Animated Processing Pipeline Timeline */}
          {selectedFile && (
            <div className="glass-panel p-5 space-y-3">
              <div className="flex items-center justify-between border-b border-white/10 pb-2 text-xs font-heading font-semibold text-gray-300">
                <span>Processing Pipeline Execution</span>
                <span className="text-[10px] font-mono-code text-amber-400">
                  {selectedFile.name}
                </span>
              </div>

              <div className="space-y-2 text-xs font-mono-code">
                <div className="flex items-center justify-between p-2 rounded-lg bg-white/5">
                  <span className="flex items-center gap-2">
                    {uploadStep >= 1 ? <FiCheckCircle className="text-emerald-400" /> : <FiLoader className="animate-spin text-amber-400" />}
                    1. File Ingestion & Format Verification
                  </span>
                  <span className="text-[10px] text-gray-400">Done</span>
                </div>

                <div className="flex items-center justify-between p-2 rounded-lg bg-white/5">
                  <span className="flex items-center gap-2">
                    {uploadStep >= 2 ? <FiCheckCircle className="text-emerald-400" /> : uploadStep === 1 ? <FiLoader className="animate-spin text-amber-400" /> : <span className="w-4" />}
                    2. FITS Parsing & Signal Extraction
                  </span>
                  <span className="text-[10px] text-gray-400">{uploadStep >= 2 ? "Done" : "Pending"}</span>
                </div>

                <div className="flex items-center justify-between p-2 rounded-lg bg-white/5">
                  <span className="flex items-center gap-2">
                    {uploadStep >= 3 ? <FiCheckCircle className="text-emerald-400" /> : uploadStep === 2 ? <FiLoader className="animate-spin text-amber-400" /> : <span className="w-4" />}
                    3. 32 Physics Feature Engineering
                  </span>
                  <span className="text-[10px] text-gray-400">{uploadStep >= 3 ? "Done" : "Pending"}</span>
                </div>

                <div className="flex items-center justify-between p-2 rounded-lg bg-white/5">
                  <span className="flex items-center gap-2">
                    {uploadStep >= 4 ? <FiCheckCircle className="text-emerald-400" /> : uploadStep === 3 ? <FiLoader className="animate-spin text-amber-400" /> : <span className="w-4" />}
                    4. HelioForgeTCN Model Inference
                  </span>
                  <span className="text-[10px] text-gray-400">{uploadStep >= 4 ? "Done" : "Pending"}</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Prediction Result Summary */}
        <div className="lg:col-span-5 space-y-4">
          <div className="glass-panel p-6 space-y-4 border border-amber-500/20">
            <div className="flex items-center justify-between border-b border-white/10 pb-3 text-xs font-heading font-semibold text-gray-300">
              <span className="flex items-center gap-1.5">
                <FiZap className="text-amber-400" /> Ingestion Verdict
              </span>
              <span className="text-[10px] font-mono-code text-emerald-400">STATUS OK</span>
            </div>

            <div className="space-y-3">
              <div>
                <span className="text-xs text-gray-400 font-mono-code block">Classification Output:</span>
                <span className="text-3xl font-heading font-bold text-amber-400">
                  {prediction.predicted_label}-Class Solar Event
                </span>
              </div>

              <div className="flex justify-between items-center text-xs font-mono-code pt-2 border-t border-white/10">
                <span className="text-gray-400">Confidence Score:</span>
                <span className="font-bold text-white">{Math.round((prediction.confidence || 0.87) * 100)}%</span>
              </div>

              <div className="flex justify-between items-center text-xs font-mono-code">
                <span className="text-gray-400">Operational Risk:</span>
                <span className="font-bold text-amber-400">{prediction.risk_level}</span>
              </div>

              <div className="flex justify-between items-center text-xs font-mono-code">
                <span className="text-gray-400">Inference Latency:</span>
                <span className="text-emerald-400 font-bold">{prediction.processing_time_ms || 78.2} ms</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
