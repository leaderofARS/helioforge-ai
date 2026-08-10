const steps = [
  "Raw FITS",
  "Preprocessing",
  "Feature Engineering",
  "HPINA Model",
  "Prediction",
];

export default function ProcessingTimeline() {
  return (
    <div className="rounded-2xl border border-slate-700 bg-slate-900 p-6">
      <h2 className="text-xl font-bold mb-6">
        Data Processing Pipeline
      </h2>

      <div className="flex justify-between items-center">
        {steps.map((step, index) => (
          <div
            key={step}
            className="flex flex-col items-center flex-1"
          >
            <div className="w-12 h-12 rounded-full bg-cyan-600 flex items-center justify-center font-bold">
              {index + 1}
            </div>

            <p className="mt-3 text-center text-sm">
              {step}
            </p>

            {index !== steps.length - 1 && (
              <div className="w-full h-1 bg-cyan-500 mt-4"></div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
