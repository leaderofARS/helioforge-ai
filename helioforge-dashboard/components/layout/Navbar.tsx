export default function Navbar() {
  return (
    <nav className="w-full h-16 bg-slate-900 border-b border-slate-700 flex items-center justify-between px-8">
      <div className="flex items-center gap-3">
        <div className="w-3 h-3 rounded-full bg-yellow-400"></div>

        <h1 className="text-2xl font-bold text-white">
          HelioForge AI
        </h1>
      </div>

      <div className="text-gray-300 text-sm">
        Solar Prelude Dashboard
      </div>
    </nav>
  );
}