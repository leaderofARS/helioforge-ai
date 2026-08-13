import GlowCard from "./GlowCard";
import ClassBadge from "./ClassBadge";
import ConfidenceMeter from "./ConfidenceMeter";
import RiskIndicator from "./RiskIndicator";

type Props = {
  label: "Quiet" | "B" | "C" | "M" | "X";
  risk: "LOW" | "MEDIUM" | "HIGH" | "EXTREME";
  confidence: number;
};

export default function PredictionCard({
  label,
  risk,
  confidence,
}: Props) {
  return (
    <GlowCard>
      <h2 className="mb-4 text-xl font-bold">
        Latest Prediction
      </h2>

      <div className="mb-4">
        <ClassBadge label={label} />
      </div>

      <div className="mb-4">
        <RiskIndicator risk={risk} />
      </div>

      <ConfidenceMeter value={confidence * 100} />
    </GlowCard>
  );
}
