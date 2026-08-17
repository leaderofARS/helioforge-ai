"use client";

import { usePredictionStore, type ActiveSection } from "@/store/usePredictionStore";
import OverviewControl from "@/components/dashboard/OverviewControl";
import InteractiveSunView from "@/components/dashboard/InteractiveSunView";
import EvolutionView from "@/components/dashboard/EvolutionView";
import PredictionView from "@/components/dashboard/PredictionView";
import IntensityView from "@/components/dashboard/IntensityView";
import SignalsView from "@/components/dashboard/SignalsView";
import FeaturesView from "@/components/dashboard/FeaturesView";
import ExplanationView from "@/components/dashboard/ExplanationView";
import UploadView from "@/components/dashboard/UploadView";
import PerformanceView from "@/components/dashboard/PerformanceView";
import AnimationView from "@/components/dashboard/AnimationView";
import ForecastView from "@/components/dashboard/ForecastView";

export default function DashboardPage({ section }: { section?: ActiveSection }) {
  const { activeSection } = usePredictionStore();
  const currentSection = section || activeSection || "control";

  switch (currentSection) {
    case "sun":
      return <InteractiveSunView />;
    case "evolution":
      return <EvolutionView />;
    case "prediction":
      return <PredictionView />;
    case "intensity":
      return <IntensityView />;
    case "signals":
      return <SignalsView />;
    case "features":
      return <FeaturesView />;
    case "upload":
      return <UploadView />;
    case "performance":
      return <PerformanceView />;
    case "animation":
      return <AnimationView />;
    case "forecast":
      return <ForecastView />;
    case "control":
    default:
      return <OverviewControl />;
  }
}
