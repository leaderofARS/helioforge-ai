"use client";

import { useRef, useState, useCallback } from "react";
import { usePredictionStore } from "@/store/usePredictionStore";
import { predictFile, Prediction } from "@/lib/api";
import {
  FiUploadCloud, FiFolder, FiFileText, FiCheckCircle,
  FiLoader, FiAlertCircle, FiZap, FiX, FiClock,
} from "react-icons/fi";

type FileStatus = "pending" | "processing" | "done" | "error";

type QueueItem = {
  id: string;
  file: File;
  status: FileStatus;
  result?: Prediction;
  error?: string;
};

const RISK_ORDER = ["EXTREME", "HIGH", "MEDIUM", "LOW"] as const;
const CLASS_COLORS: Record<string, string> = {
  Quiet: "#22c55e",
  B:     "#3b82f6",
  C:     "#f59e0b",
  M:     "#f97316",
  X:     "#ef4444",
};

const ACCEPTED = ".fits,.fit,.pt,.json,.csv";

function classifyBadge(label: string) {
  const color = CLASS_COLORS[label] ?? "#9ca3af";
  return (
    <span
      className="inline-block px-2 py-0.5 rounded font-mono-code text-[10px] font-bold"
      style={{ background: `${color}22`, color }}
    >
      {label}
    </span>
  );
}

export default function UploadView() {
  const { setPrediction } = usePredictionStore();
  const [dragActive, setDragActive] = useState(false);
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [running, setRunning] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const fileInputRef   = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const updateItem = useCallback(
    (id: string, patch: Partial<QueueItem>) =>
      setQueue((prev) => prev.map((q) => (q.id === id ? { ...q, ...patch } : q))),
    []
  );

  const addFiles = useCallback((files: File[]) => {
    const valid = files.filter((f) => /\.(fits|fit|pt|json|csv)$/i.test(f.name));
    if (!valid.length) {
      setUploadError("No supported files found (.fits .pt .json .csv).");
      return undefined;
    }
    setUploadError(null);
    const items: QueueItem[] = valid.map((file) => ({
      id: `${file.name}-${Date.now()}-${Math.random()}`,
      file,
      status: "pending",
    }));
    setQueue((prev) => [...prev, ...items]);
    return items;
  }, []);

  const processQueue = useCallback(async (items: QueueItem[]) => {
    setRunning(true);
    let best: Prediction | undefined;
    for (const item of items) {
      updateItem(item.id, { status: "processing" });
      try {
        const result = await predictFile(item.file);
        updateItem(item.id, { status: "done", result });
        if (!best) {
          best = result;
        } else {
          const ri = RISK_ORDER.indexOf(result.risk_level as typeof RISK_ORDER[number]);
          const bi = RISK_ORDER.indexOf(best.risk_level as typeof RISK_ORDER[number]);
          if (ri < bi) best = result;
        }
      } catch (err) {
        updateItem(item.id, {
          status: "error",
          error: err instanceof Error ? err.message : "Inference failed",
        });
      }
    }
    if (best) setPrediction(best);
    setRunning(false);
  }, [updateItem, setPrediction]);

  const handleInputChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      if (!e.target.files?.length) return;
      const files = Array.from(e.target.files);
      e.target.value = "";
      const items = addFiles(files);
      if (items?.length) await processQueue(items);
    },
    [addFiles, processQueue]
  );

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      const files = Array.from(e.dataTransfer.files);
      const items = addFiles(files);
      if (items?.length) await processQueue(items);
    },
    [addFiles, processQueue]
  );

  const done  = queue.filter((q) => q.status === "done");
  const errs  = queue.filter((q) => q.status === "error");
  const total = queue.length;

  const classCounts = done.reduce<Record<string, number>>((acc, q) => {
    const lbl = q.result!.predicted_label;
    acc[lbl] = (acc[lbl] ?? 0) + 1;
    return acc;
  }, {});

  const avgConfidence = done.length
    ? done.reduce((s, q) => s + q.result!.confidence, 0) / done.length
    : 0;

  const totalMs = done.reduce((s, q) => s + (q.result?.processing_time_ms ?? 0), 0);

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/10 pb-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono-code text-amber-400 font-bold mb-1">
            <FiUploadCloud className="w-4 h-4" /> PAGE 12 — DATASET EXPLORER &amp; FITS UPLOADER
          </div>
          <h1 className="text-2xl md:text-3xl font-heading font-bold text-white tracking-tight">
            Ingest Raw Solar Telemetry
          </h1>
          <p className="text-sm text-gray-400 max-w-3xl mt-1">
            Upload individual <code>.fits</code> / <code>.pt</code> files or an entire
            observation folder — every file is run sequentially through HelioForgeTCN.
          </p>
        </div>
        {queue.length > 0 && !running && (
          <button
            onClick={() => setQueue([])}
            className="flex items-center gap-1.5 text-xs font-mono-code text-gray-400 hover:text-red-400 transition-colors"
          >
            <FiX /> Clear queue
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7 space-y-4">
          <div
            onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
            className={`glass-panel p-8 border-2 border-dashed text-center space-y-5 transition-all ${
              dragActive
                ? "border-amber-500 bg-amber-500/10 shadow-2xl scale-[1.01]"
                : "border-white/20 hover:border-amber-500/40"
            }`}
          >
            <div className="w-16 h-16 rounded-2xl bg-amber-500/10 border border-amber-500/20 text-amber-400 flex items-center justify-center mx-auto">
              <FiUploadCloud className="w-8 h-8" />
            </div>
            <div>
              <h3 className="font-heading font-bold text-lg text-white">
                Drag &amp; Drop Solar Telemetry
              </h3>
              <p className="text-xs text-gray-400 font-mono-code mt-1">
                Accepts <code>.fits</code> &nbsp;·&nbsp; <code>.pt</code> &nbsp;·&nbsp;
                <code>.json</code> &nbsp;·&nbsp; <code>.csv</code>
              </p>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept={ACCEPTED}
              multiple
              style={{ display: "none" }}
              onChange={handleInputChange}
            />
            <input
              ref={folderInputRef}
              type="file"
              accept={ACCEPTED}
              // @ts-expect-error webkitdirectory is non-standard
              webkitdirectory=""
              multiple
              style={{ display: "none" }}
              onChange={handleInputChange}
            />

            <div className="flex flex-wrap gap-3 justify-center">
              <button
                type="button"
                disabled={running}
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center gap-2 px-6 py-2.5 rounded-xl bg-amber-500 hover:bg-amber-400 disabled:opacity-50 text-black font-heading font-bold text-sm transition-all shadow-lg shadow-amber-500/20"
              >
                <FiFileText />
                {running ? "Processing…" : "Browse Files"}
              </button>
              <button
                type="button"
                disabled={running}
                onClick={() => folderInputRef.current?.click()}
                className="inline-flex items-center gap-2 px-6 py-2.5 rounded-xl bg-white/10 hover:bg-white/20 disabled:opacity-50 text-white font-heading font-bold text-sm transition-all border border-white/20"
              >
                <FiFolder /> Upload Folder
              </button>
            </div>

            {uploadError && (
              <p className="text-xs text-red-400 font-mono-code flex items-center justify-center gap-1.5">
                <FiAlertCircle /> {uploadError}
              </p>
            )}
          </div>

          {queue.length > 0 && (
            <div className="glass-panel p-5 space-y-3">
              <div className="flex items-center justify-between border-b border-white/10 pb-2">
                <span className="text-xs font-heading font-semibold text-gray-300">Processing Queue</span>
                <span className="text-[10px] font-mono-code text-amber-400">
                  {done.length}/{total} done
                  {errs.length > 0 && <span className="text-red-400 ml-2">{errs.length} failed</span>}
                </span>
              </div>
              <div className="w-full h-1 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full bg-amber-500 transition-all duration-500"
                  style={{ width: total ? `${(done.length / total) * 100}%` : "0%" }}
                />
              </div>
              <div className="space-y-1.5 max-h-64 overflow-y-auto pr-1">
                {queue.map((item) => (
                  <div
                    key={item.id}
                    className="flex items-center justify-between p-2.5 rounded-lg bg-white/5 gap-3 min-w-0"
                  >
                    <div className="flex items-center gap-2 min-w-0 flex-1">
                      {item.status === "done"       && <FiCheckCircle className="text-emerald-400 flex-shrink-0" />}
                      {item.status === "processing"  && <FiLoader className="animate-spin text-amber-400 flex-shrink-0" />}
                      {item.status === "pending"     && <span className="w-4 h-4 flex-shrink-0" />}
                      {item.status === "error"       && <FiAlertCircle className="text-red-400 flex-shrink-0" />}
                      <span className="text-xs font-mono-code text-gray-300 truncate">{item.file.name}</span>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      {item.status === "done" && item.result && (
                        <>
                          {classifyBadge(item.result.predicted_label)}
                          <span className="text-[10px] font-mono-code text-gray-400">{Math.round(item.result.confidence * 100)}%</span>
                          <span className="text-[10px] font-mono-code text-emerald-400">{item.result.processing_time_ms?.toFixed(1)}ms</span>
                        </>
                      )}
                      {item.status === "error"      && <span className="text-[10px] font-mono-code text-red-400 max-w-[140px] truncate">{item.error}</span>}
                      {item.status === "pending"    && <span className="text-[10px] font-mono-code text-gray-500">queued</span>}
                      {item.status === "processing" && <span className="text-[10px] font-mono-code text-amber-400">running…</span>}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="lg:col-span-5 space-y-4">
          <div className="glass-panel p-6 space-y-4 border border-amber-500/20">
            <div className="flex items-center justify-between border-b border-white/10 pb-3 text-xs font-heading font-semibold text-gray-300">
              <span className="flex items-center gap-1.5"><FiZap className="text-amber-400" /> Batch Ingestion Summary</span>
              <span className="text-[10px] font-mono-code text-emerald-400">
                {done.length > 0 ? "RESULTS READY" : "AWAITING FILES"}
              </span>
            </div>
            {done.length === 0 ? (
              <p className="text-xs text-gray-500 font-mono-code text-center py-6">
                Upload files to see live inference results here.
              </p>
            ) : (
              <div className="space-y-4">
                <div>
                  <p className="text-[10px] font-mono-code text-gray-400 mb-2">
                    CLASS DISTRIBUTION ({done.length} observations)
                  </p>
                  <div className="space-y-1.5">
                    {Object.entries(classCounts).sort(([,a],[,b]) => b - a).map(([cls, count]) => (
                      <div key={cls} className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: CLASS_COLORS[cls] ?? "#9ca3af" }} />
                        <span className="text-xs font-mono-code text-gray-300 w-12">{cls}</span>
                        <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-700"
                            style={{ width: `${(count / done.length) * 100}%`, background: CLASS_COLORS[cls] ?? "#9ca3af" }}
                          />
                        </div>
                        <span className="text-[10px] font-mono-code text-gray-400 w-8 text-right">{count}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="border-t border-white/10 pt-3 space-y-2 text-xs font-mono-code">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Files Processed</span>
                    <span className="text-white font-bold">{done.length} / {total}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Avg Confidence</span>
                    <span className="text-white font-bold">{Math.round(avgConfidence * 100)}%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400 flex items-center gap-1"><FiClock className="w-3 h-3" /> Total Inference</span>
                    <span className="text-emerald-400 font-bold">{totalMs.toFixed(1)} ms</span>
                  </div>
                  {errs.length > 0 && (
                    <div className="flex justify-between">
                      <span className="text-gray-400">Failures</span>
                      <span className="text-red-400 font-bold">{errs.length}</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
