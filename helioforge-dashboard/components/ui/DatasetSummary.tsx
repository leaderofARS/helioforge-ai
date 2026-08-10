import GlowCard from "./GlowCard";

export default function DatasetSummary() {
  return (
    <GlowCard>

      <h2 className="text-xl font-bold mb-5">
        Dataset Summary
      </h2>

      <div className="space-y-2">

        <p>HEL1OS : 268 Observations</p>

        <p>Synchronized : 51</p>

        <p>Features : 38</p>

        <p>Missing Values : 0</p>

      </div>

    </GlowCard>
  );
}