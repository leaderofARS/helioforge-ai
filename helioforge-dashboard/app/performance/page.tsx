import ConfusionMatrix from "@/components/charts/ConfusionMatrix";
import FeatureCard from "@/components/ui/FeatureCard";

export default function PerformancePage() {
  return (

<div className="space-y-8">

<h1 className="text-5xl font-bold">
Model Performance
</h1>

<div className="grid grid-cols-4 gap-5">

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

<ConfusionMatrix/>

</div>

  );
}