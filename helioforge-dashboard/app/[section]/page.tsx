import DashboardPage from "@/components/dashboard/DashboardPage";
import { type ActiveSection } from "@/store/usePredictionStore";

const allowed = new Set<string>([
  "control",
  "sun",
  "evolution",
  "prediction",
  "intensity",
  "signals",
  "features",
  "upload",
  "performance",
  "animation",
  "forecast",
]);

export default async function SectionPage({
  params,
}: {
  params: Promise<{ section: string }>;
}) {
  const { section } = await params;
  const validSection = (allowed.has(section) ? section : "control") as ActiveSection;
  return <DashboardPage section={validSection} />;
}
