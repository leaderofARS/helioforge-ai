import GlowCard from "./GlowCard";

export default function ModelStatus() {
  return (
    <GlowCard>
      <h2 className="mb-5 text-xl font-bold">
        HelioForgeTCN Status
      </h2>

      <div className="space-y-3 text-slate-300">
        <p>
          🟢 Model :{" "}
          <span className="font-semibold text-white">
            Loaded
          </span>
        </p>

        <p>
          📦 Checkpoint
        </p>

        <p className="text-green-400">
          best_macro_f1.pt
        </p>

        <p>
          Epoch : <span className="text-white">25</span>
        </p>

        <p>
          Device : <span className="text-white">CPU</span>
        </p>

        <p>
          Validation F1 :{" "}
          <span className="text-white">
            0.8714
          </span>
        </p>
      </div>
    </GlowCard>
  );
}
