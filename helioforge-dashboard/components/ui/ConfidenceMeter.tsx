type Props = {
  value: number;
};

export default function ConfidenceMeter({ value }: Props) {
  return (
    <div>
      <div className="flex justify-between mb-2">
        <span>Confidence</span>
        <span>{value}%</span>
      </div>

      <div className="w-full bg-slate-700 rounded-full h-3">
        <div
          className="bg-cyan-500 h-3 rounded-full"
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}