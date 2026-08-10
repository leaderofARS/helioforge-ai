"use client";

type ClassProbability = {
  className: string;
  probability: number;
};

type Props = {
  probabilities?: ClassProbability[];
};

const defaultProbabilities: ClassProbability[] = [
  { className: "Quiet", probability: 0.08 },
  { className: "B", probability: 0.12 },
  { className: "C", probability: 0.18 },
  { className: "M", probability: 0.52 },
  { className: "X", probability: 0.10 },
];

export default function ProbabilityBars({
  probabilities = defaultProbabilities,
}: Props) {
  return (
    <div className="space-y-4">
      {probabilities.map((item) => (
        <div key={item.className}>
          <div className="mb-1 flex justify-between text-sm">
            <span className="font-medium">
              {item.className}
            </span>

            <span className="text-gray-400">
              {(item.probability * 100).toFixed(1)}%
            </span>
          </div>

          <div className="h-3 w-full overflow-hidden rounded-full bg-gray-800">
            <div
              className="h-full rounded-full bg-cyan-400 transition-all duration-500"
              style={{
                width: `${item.probability * 100}%`,
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}