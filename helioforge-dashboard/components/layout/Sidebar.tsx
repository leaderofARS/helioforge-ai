export default function Sidebar() {
  return (
    <aside className="w-64 min-h-screen bg-slate-950 border-r border-slate-700 p-6">

      <h2 className="text-xl font-bold text-white mb-8">
        Navigation
      </h2>

      <ul className="space-y-4 text-gray-300">

        <li>🏠 Mission Control</li>

        <li>☀️ Sun</li>

        <li>📈 Prediction</li>

        <li>📊 Signals</li>

        <li>🧠 Features</li>

        <li>📂 Upload</li>

        <li>
             <a href="/performance">
             ⚡ Performance
              </a>
        </li>

        <li>🎞 Animation</li>

        <li>🔮 Forecast</li>

      </ul>

    </aside>
  );
}