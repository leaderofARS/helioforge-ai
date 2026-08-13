import GlowCard from "./GlowCard";

type Props = {
  model: string;
  checkpoint: string;
  epoch: number;
  macroF1: number;
};

export default function ModelStatus({
  model,
  checkpoint,
  epoch,
  macroF1,
}: Props) {
  return (
    <GlowCard>
      <h2 className="mb-5 text-xl font-bold">
        {model} Status
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
          {checkpoint}
        </p>

        <p>
          Epoch :{" "}
          <span className="text-white">
            {epoch}
          </span>
        </p>

        <p>
          Device :{" "}
          <span className="text-white">
            CPU
          </span>
        </p>

        <p>
          Validation F1 :{" "}
          <span className="text-white">
            {macroF1.toFixed(4)}
          </span>
        </p>
      </div>
    </GlowCard>
  );
}
