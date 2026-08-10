import PredictionCard from "@/components/ui/PredictionCard";
import FeatureCard from "@/components/ui/FeatureCard";
import ProcessingTimeline from "@/components/ui/ProcessingTimeline";
import DatasetSummary from "@/components/ui/DatasetSummary";
import ModelStatus from "@/components/ui/ModelStatus";

export default function Home() {
  return (
    <div className="space-y-8">

      <div>
        <h1 className="text-5xl font-bold">
          Mission Control
        </h1>

        <p className="text-gray-400 mt-2">
          AI-powered Solar Flare Prediction Dashboard
        </p>
      </div>

      <div className="grid grid-cols-3 gap-6">

        <PredictionCard />

        <FeatureCard
          title="Test Accuracy"
          value="89.41%"
        />

        <FeatureCard
          title="Macro F1"
          value="0.8514"
        />

      </div>

      <div className="grid grid-cols-2 gap-6">

        <DatasetSummary />

        <ModelStatus />

      </div>

      <ProcessingTimeline />

    </div>
  );
}