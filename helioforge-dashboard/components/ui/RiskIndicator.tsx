type Props = {
  risk: "LOW" | "MEDIUM" | "HIGH" | "EXTREME";
};

const colors = {
  LOW: "bg-green-500",
  MEDIUM: "bg-yellow-500",
  HIGH: "bg-orange-500",
  EXTREME: "bg-red-600",
};

export default function RiskIndicator({ risk }: Props) {
  return (
    <div
      className={`rounded-xl px-4 py-2 text-center font-bold text-white ${colors[risk]}`}
    >
      {risk}
    </div>
  );
}