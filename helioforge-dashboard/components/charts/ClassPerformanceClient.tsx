"use client";

import dynamic from "next/dynamic";

const ClassPerformance = dynamic(
  () => import("./ClassPerformance"),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-[500px] items-center justify-center rounded-2xl border border-slate-700 bg-slate-900/80">
        Loading class performance...
      </div>
    ),
  }
);

export default function ClassPerformanceClient() {
  return <ClassPerformance />;
}
