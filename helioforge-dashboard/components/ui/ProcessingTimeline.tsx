import GlowCard from "./GlowCard";

const steps = [
  "Data Ingestion",
  "Preprocessing",
  "Feature Engineering",
  "TCN Model",
  "Prediction",
];

export default function ProcessingTimeline() {
  return (
    <GlowCard>
      <h2 className="mb-6 text-xl font-bold">
        Data Processing Pipeline
      </h2>

      <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
        {steps.map((step, index) => (
          <div
            key={step}
            className="flex flex-1 flex-col items-center"
          >
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-cyan-600 font-bold">
              {index + 1}
            </div>

            <p className="mt-3 text-center text-sm">
              {step}
            </p>

            {index !== steps.length - 1 && (
              <div className="mt-4 hidden h-1 w-full bg-cyan-500 md:block" />
            )}
          </div>
        ))}
      </div>
    </GlowCard>
  );
}
