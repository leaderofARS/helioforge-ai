"use client";
import { create } from "zustand";
import { getDemoPrediction, predictFile, type Prediction } from "@/lib/api";

const fallback: Prediction = { predicted_class: 3, predicted_label: "M", confidence: .87, risk_level: "HIGH", probabilities: { Quiet: 0, B: .03, C: .09, M: .84, X: .04 }, observation_id: "OBS_20260728_0031" };
type State = { prediction: Prediction; isLoading: boolean; error: string | null; setPrediction: (prediction: Prediction) => void; fetchDemo: (index?: number) => Promise<void>; predict: (file: File) => Promise<void> };
export const usePredictionStore = create<State>((set) => ({
  prediction: fallback, isLoading: false, error: null,
  setPrediction: (prediction) => set({ prediction, error: null }),
  fetchDemo: async (index = 0) => { set({ isLoading: true, error: null }); try { set({ prediction: await getDemoPrediction(index), isLoading: false }); } catch { set({ isLoading: false, error: "Live API unavailable — showing mission reference data." }); } },
  predict: async (file) => { set({ isLoading: true, error: null }); try { set({ prediction: await predictFile(file), isLoading: false }); } catch (error) { set({ isLoading: false, error: error instanceof Error ? error.message : "Upload failed." }); } },
}));
