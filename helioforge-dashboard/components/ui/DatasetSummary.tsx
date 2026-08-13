import GlowCard from "./GlowCard";

export default function DatasetSummary() {
  return (
    <GlowCard>
      <h2 className="mb-5 text-xl font-bold">
        Test Dataset
      </h2>

      <div className="space-y-3 text-slate-300">
        <p>
          Test Windows : <span className="font-semibold text-white">406</span>
        </p>

        <p>
          Split : <span className="font-semibold text-white">Held-out Test</span>
        </p>

        <p>
          Classes :{" "}
          <span className="font-semibold text-white">
            Quiet / B / C / M / X
          </span>
        </p>

        <p>
          Evaluation :{" "}
          <span className="font-semibold text-green-400">
            Completed
          </span>
        </p>
      </div>
    </GlowCard>
  );
}
