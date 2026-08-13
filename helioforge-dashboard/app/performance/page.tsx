import ConfusionMatrixClient from "@/components/charts/ConfusionMatrixClient";
import FeatureCard from "@/components/ui/FeatureCard";
import ClassPerformanceClient from "@/components/charts/ClassPerformanceClient";
<ConfusionMatrixClient />


export default function PerformancePage() {
  return (
    <main className="space-y-8">
      <div>
        <h1 className="text-5xl font-bold">
          Model Performance
        </h1>

        <p className="mt-2 text-gray-400">
          HelioForgeTCN evaluation on the held-out test set
        </p>
      </div>

      <div className="grid grid-cols-1 gap-5 md:grid-cols-2 lg:grid-cols-4">
        <FeatureCard
          title="Accuracy"
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

      <ConfusionMatrixClient />
      <ClassPerformanceClient />
    </main>
  );
}
