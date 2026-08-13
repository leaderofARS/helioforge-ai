"use client";

import dynamic from "next/dynamic";

const ConfusionMatrix = dynamic(
  () => import("./ConfusionMatrix"),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-[550px] items-center justify-center rounded-xl border border-gray-800">
        Loading confusion matrix...
      </div>
    ),
  }
);

export default function ConfusionMatrixClient() {
  return <ConfusionMatrix />;
}
