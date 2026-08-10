import PredictionCard from "@/components/ui/PredictionCard";
import FeatureCard from "@/components/ui/FeatureCard";
import ProcessingTimeline from "@/components/ui/ProcessingTimeline";
import DatasetSummary from "@/components/ui/DatasetSummary";
import ModelStatus from "@/components/ui/ModelStatus";
import ProbabilityBars from "@/components/charts/ProbabilityBars";
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
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
  <PredictionCard />

  <div className="rounded-xl border border-gray-800 bg-gray-950 p-6">
    <h2 className="mb-5 text-xl font-semibold">
      Flare Class Probabilities
    </h2>

    <ProbabilityBars />
  </div>
</div>

     

      <div className="grid grid-cols-2 gap-6">

        <DatasetSummary />

        <ModelStatus />

      </div>

      <ProcessingTimeline />

    </div>
  );
}