type Props = {
  label: string;
};

const colors: Record<string, string> = {
  Quiet: "bg-green-500",
  B: "bg-blue-500",
  C: "bg-yellow-500",
  M: "bg-orange-500",
  X: "bg-red-600",
};

export default function ClassBadge({ label }: Props) {
  return (
    <span
      className={`px-3 py-1 rounded-full text-white font-bold ${
        colors[label] || "bg-gray-600"
      }`}
    >
      {label}
    </span>
  );
}