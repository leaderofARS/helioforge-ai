import FeatureCard from "@/components/ui/FeatureCard";
import ProcessingTimeline from "@/components/ui/ProcessingTimeline";
import DatasetSummary from "@/components/ui/DatasetSummary";
import ModelStatus from "@/components/ui/ModelStatus";
import PredictionCard from "@/components/ui/PredictionCard";
import ProbabilityBars from "@/components/charts/ProbabilityBars";
import { getDemoPrediction, getHealth } from "@/lib/api";
import PredictionExplorer from "@/components/prediction/PredictionExplorer";
export default async function Home() {
  const [health, prediction] = await Promise.all([
    getHealth(),
    getDemoPrediction(300),
  ]);

  const probabilities = Object.entries(
    prediction.probabilities
  ).map(([className, probability]) => ({
    className,
    probability,
  }));

  return (
    <main className="space-y-8">

      <div>
        <h1 className="text-5xl font-bold">
          Mission Control
        </h1>

        <p className="mt-2 text-gray-400">
          AI-powered Solar Flare Prediction and Analysis using Aditya-L1
        </p>
      </div>

      <div className="grid grid-cols-1 gap-5 md:grid-cols-2 lg:grid-cols-4">
        <FeatureCard
          title="Validation Macro F1"
          value={health.macro_f1.toFixed(4)}
        />

        <FeatureCard
          title="Model Epoch"
          value={String(health.epoch)}
        />

        <FeatureCard
          title="Input Features"
          value={String(health.input_shape[0])}
        />

        <FeatureCard
          title="Window Size"
          value={String(health.input_shape[1])}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <PredictionCard
          label={prediction.predicted_label}
          risk={prediction.risk_level}
          confidence={prediction.confidence}
        />

        <ModelStatus
          model={health.model}
          checkpoint={health.checkpoint}
          epoch={health.epoch}
          macroF1={health.macro_f1}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <DatasetSummary />

        <div className="rounded-2xl border border-gray-800 bg-gray-950 p-6">
          <h2 className="mb-5 text-xl font-bold">
            Class Probabilities
          </h2>

          <ProbabilityBars
            probabilities={probabilities}
          />
        </div>
      </div>
      <PredictionExplorer />

      <ProcessingTimeline />

    </main>
  );
}
