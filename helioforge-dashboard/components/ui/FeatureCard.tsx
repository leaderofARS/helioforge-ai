import GlowCard from "./GlowCard";

type Props = {
  title: string;
  value: string | number;
};

export default function FeatureCard({ title, value }: Props) {
  return (
    <GlowCard>
      <p className="text-gray-400">{title}</p>

      <h2 className="text-3xl font-bold mt-2">{value}</h2>
    </GlowCard>
  );
}