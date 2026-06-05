"use client";

import {
  CartesianGrid,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ResponsiveContainer,
  ZAxis,
} from "recharts";
import { Candidate } from "./types";

type Point = Candidate & { robPct: number };

export default function Frontier({
  candidates,
  selected,
  onSelect,
}: {
  candidates: Candidate[];
  selected: number;
  onSelect: (i: number) => void;
}) {
  if (!candidates.length) {
    return <div className="placeholder">No candidate routings — try a higher α or fewer customers.</div>;
  }

  const data: Point[] = candidates.map((c) => ({ ...c, robPct: c.robustness * 100 }));

  return (
    <ResponsiveContainer width="100%" height={360}>
      <ScatterChart margin={{ top: 12, right: 16, bottom: 36, left: 8 }}>
        <CartesianGrid stroke="#21303f" strokeDasharray="3 3" />
        <XAxis
          type="number"
          dataKey="robPct"
          name="Robustness"
          unit="%"
          domain={[0, 100]}
          tick={{ fill: "#8aa0b4", fontSize: 12 }}
          stroke="#21303f"
          label={{ value: "Robustness  (P stays feasible)", position: "bottom", fill: "#8aa0b4", fontSize: 12 }}
        />
        <YAxis
          type="number"
          dataKey="objective"
          name="Distance"
          tick={{ fill: "#8aa0b4", fontSize: 12 }}
          stroke="#21303f"
          label={{ value: "Total distance", angle: -90, position: "insideLeft", fill: "#8aa0b4", fontSize: 12 }}
        />
        <ZAxis range={[70, 70]} />
        <Tooltip
          cursor={{ strokeDasharray: "3 3", stroke: "#2dd4bf" }}
          contentStyle={{ background: "#0f1620", border: "1px solid #21303f", borderRadius: 8, color: "#e7eef6" }}
          formatter={(value: number, name: string) =>
            name === "Robustness" ? [`${value.toFixed(1)}%`, name] : [value.toFixed(2), name]
          }
        />
        <Scatter
          data={data}
          fill="#60a5fa"
          onClick={(_: unknown, index: number) => onSelect(index)}
          shape={(props: any) => {
            const isSel = props.payload && data[selected] === props.payload;
            return (
              <circle
                cx={props.cx}
                cy={props.cy}
                r={isSel ? 8 : 5}
                fill={isSel ? "#f59e0b" : "#60a5fa"}
                stroke={isSel ? "#fff" : "none"}
                strokeWidth={isSel ? 2 : 0}
                style={{ cursor: "pointer" }}
              />
            );
          }}
        />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
