"use client";

import Plot from "react-plotly.js";

export default function ClassPerformance() {
  const classes = ["Quiet", "B", "C", "M", "X"];

  const precision = [
    0.9485,
    0.8716,
    0.9775,
    0.8696,
    0.5769,
  ];

  const recall = [
    1.0000,
    0.9627,
    0.7982,
    0.7547,
    0.8333,
  ];

  const f1 = [
    0.9735,
    0.9149,
    0.8788,
    0.8081,
    0.6818,
  ];

  return (
    <Plot
      data={[
        {
          x: classes,
          y: precision,
          name: "Precision",
          type: "bar",
        },
        {
          x: classes,
          y: recall,
          name: "Recall",
          type: "bar",
        },
        {
          x: classes,
          y: f1,
          name: "F1",
          type: "bar",
        },
      ]}
      layout={{
        title: "Per-Class Performance",
        barmode: "group",
        yaxis: {
          title: "Score",
          range: [0, 1],
        },
        xaxis: {
          title: "Flare Class",
        },
        height: 500,
        margin: {
          l: 60,
          r: 30,
          t: 70,
          b: 60,
        },
      }}
      config={{
        responsive: true,
      }}
      style={{
        width: "100%",
      }}
    />
  );
}
