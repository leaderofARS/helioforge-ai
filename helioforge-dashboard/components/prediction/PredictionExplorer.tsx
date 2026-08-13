"use client";

import { useState } from "react";

type Prediction = {
  predicted_class: number;
  predicted_label: "Quiet" | "B" | "C" | "M" | "X";
  confidence: number;
  risk_level: "LOW" | "MEDIUM" | "HIGH" | "EXTREME";
  probabilities: {
    Quiet: number;
    B: number;
    C: number;
    M: number;
    X: number;
  };
  observation_id?: string;
  sample_index?: number;
  input_shape?: number[];
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const riskColors = {
  LOW: "bg-green-500/20 text-green-400 border-green-500/30",
  MEDIUM: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  HIGH: "bg-orange-500/20 text-orange-400 border-orange-500/30",
  EXTREME: "bg-red-500/20 text-red-400 border-red-500/30",
};

export default function PredictionExplorer() {
  const [index, setIndex] = useState(300);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function predict() {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/demo/${index}`
      );

      if (!response.ok) {
        throw new Error(`Prediction failed: ${response.status}`);
      }

      const data: Prediction = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to prediction API"
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="rounded-2xl border border-gray-800 bg-gray-950 p-6">
      <div className="mb-6">
        <h2 className="text-xl font-bold">
          Prediction Explorer
        </h2>

        <p className="mt-1 text-sm text-gray-400">
          Run the real HelioForgeTCN model against a test sample.
        </p>
      </div>

      <div className="mb-6 flex flex-col gap-3 sm:flex-row">
        <div className="flex-1">
          <label
            htmlFor="sample-index"
            className="mb-2 block text-sm text-gray-400"
          >
            Sample index
          </label>

          <input
            id="sample-index"
            type="number"
            min="0"
            max="405"
            value={index}
            onChange={(e) => setIndex(Number(e.target.value))}
            className="w-full rounded-xl border border-gray-700 bg-gray-900 px-4 py-3 text-white outline-none focus:border-cyan-400"
          />
        </div>

        <div className="flex items-end">
          <button
            type="button"
            onClick={predict}
            disabled={loading}
            className="w-full rounded-xl bg-cyan-500 px-6 py-3 font-semibold text-black transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-50 sm:w-auto"
          >
            {loading ? "Predicting..." : "Predict"}
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-6 rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-red-400">
          {error}
        </div>
      )}

      {prediction && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <div className="rounded-xl border border-gray-800 bg-gray-900 p-5">
              <p className="text-sm text-gray-400">
                Prediction
              </p>

              <p className="mt-2 text-4xl font-bold text-cyan-400">
                {prediction.predicted_label}
              </p>
            </div>

            <div className="rounded-xl border border-gray-800 bg-gray-900 p-5">
              <p className="text-sm text-gray-400">
                Confidence
              </p>

              <p className="mt-2 text-3xl font-bold">
                {(prediction.confidence * 100).toFixed(4)}%
              </p>
            </div>

            <div className="rounded-xl border border-gray-800 bg-gray-900 p-5">
              <p className="text-sm text-gray-400">
                Risk
              </p>

              <div className="mt-2">
                <span
                  className={`inline-block rounded-lg border px-4 py-2 font-bold ${riskColors[prediction.risk_level]}`}
                >
                  {prediction.risk_level}
                </span>
              </div>
            </div>
          </div>

          <div>
            <h3 className="mb-4 font-semibold">
              Class Probabilities
            </h3>

            <div className="space-y-4">
              {Object.entries(prediction.probabilities).map(
                ([label, probability]) => (
                  <div key={label}>
                    <div className="mb-1 flex justify-between text-sm">
                      <span>{label}</span>

                      <span className="text-gray-400">
                        {(probability * 100).toFixed(4)}%
                      </span>
                    </div>

                    <div className="h-3 overflow-hidden rounded-full bg-gray-800">
                      <div
                        className="h-full rounded-full bg-cyan-400 transition-all duration-500"
                        style={{
                          width: `${Math.max(
                            probability * 100,
                            probability > 0 ? 0.5 : 0
                          )}%`,
                        }}
                      />
                    </div>
                  </div>
                )
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 gap-3 text-sm text-gray-400 sm:grid-cols-3">
            <div>
              Observation:{" "}
              <span className="text-white">
                {prediction.observation_id ?? "-"}
              </span>
            </div>

            <div>
              Sample:{" "}
              <span className="text-white">
                {prediction.sample_index ?? index}
              </span>
            </div>

            <div>
              Input:{" "}
              <span className="text-white">
                {prediction.input_shape
                  ? prediction.input_shape.slice(1).join(" × ")
                  : "32 × 512"}
              </span>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
