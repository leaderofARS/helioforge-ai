type GlowCardProps = {
  children: React.ReactNode;
};

export default function GlowCard({ children }: GlowCardProps) {
  return (
    <div className="rounded-2xl border border-slate-700 bg-slate-900/80 backdrop-blur-md p-6 shadow-lg hover:shadow-cyan-500/20 transition duration-300">
      {children}
    </div>
  );
}