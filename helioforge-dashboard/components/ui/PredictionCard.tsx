import GlowCard from "./GlowCard";
import ClassBadge from "./ClassBadge";
import ConfidenceMeter from "./ConfidenceMeter";
import RiskIndicator from "./RiskIndicator";

export default function PredictionCard() {
  return (
    <GlowCard>
      <h2 className="text-xl font-bold mb-4">Latest Prediction</h2>

      <div className="mb-4">
        <ClassBadge label="M" />
      </div>

      <div className="mb-4">
        <RiskIndicator risk="HIGH" />
      </div>

      <ConfidenceMeter value={87.14} />
    </GlowCard>
  );
}