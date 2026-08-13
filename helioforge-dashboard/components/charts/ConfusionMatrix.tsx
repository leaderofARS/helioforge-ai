"use client";

import Plot from "react-plotly.js";

export default function ConfusionMatrix() {
  return (
    <Plot
      data={[
        {
          z: [
            [92,0,0,0,0],
            [5,129,0,0,0],
            [0,19,87,3,0],
            [0,0,2,40,11],
            [0,0,0,3,15],
          ],
          x:["Quiet","B","C","M","X"],
          y:["Quiet","B","C","M","X"],
          type:"heatmap",
          colorscale:"Viridis",
        },
      ]}
      layout={{
        title:"Confusion Matrix",
        width:650,
        height:550,
      }}
    />
  );
}