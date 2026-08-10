import GlowCard from "./GlowCard";

type Props = {
  title: string;
  value: string | number;
};

export default function FeatureCard({ title, value }: Props) {
  return (
    <GlowCard>
      <p className="text-sm text-gray-400">
        {title}
      </p>

      <h2 className="mt-2 text-3xl font-bold">
        {value}
      </h2>
    </GlowCard>
  );
}