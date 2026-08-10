import GlowCard from "./GlowCard";

export default function ModelStatus() {
  return (
    <GlowCard>

      <h2 className="text-xl font-bold mb-5">
        HPINA Status
      </h2>

      <div className="space-y-3">

        <p>
          🟢 Model Loaded
        </p>

        <p>
          📦 Checkpoint
        </p>

        <p className="text-green-400">
          best_macro_f1.pt
        </p>

        <p>
          ⚙ Device : CPU
        </p>

      </div>

    </GlowCard>
  );
}