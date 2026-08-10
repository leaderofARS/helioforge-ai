import FeatureCard from "@/components/ui/FeatureCard";
import ProcessingTimeline from "@/components/ui/ProcessingTimeline";
import DatasetSummary from "@/components/ui/DatasetSummary";
import ModelStatus from "@/components/ui/ModelStatus";

export default function Home() {
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
          title="Test Accuracy"
          value="89.41%"
        />

        <FeatureCard
          title="Macro F1"
          value="0.8514"
        />

        <FeatureCard
          title="Precision"
          value="0.8488"
        />

        <FeatureCard
          title="Recall"
          value="0.8698"
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <DatasetSummary />

        <ModelStatus />
      </div>

      <ProcessingTimeline />

    </main>
  );
}
