"use client";

import { Node, Candidate, ROUTE_COLORS } from "./types";

const W = 520;
const H = 420;
const PAD = 28;

export default function RouteMap({
  nodes,
  candidate,
}: {
  nodes: Node[];
  candidate: Candidate | null;
}) {
  if (!nodes.length) {
    return <div className="placeholder">Pick an instance to see its map.</div>;
  }

  const xs = nodes.map((n) => n.x);
  const ys = nodes.map((n) => n.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;

  const px = (x: number) => PAD + ((x - minX) / spanX) * (W - 2 * PAD);
  // Flip Y so larger coordinates are higher on screen.
  const py = (y: number) => H - PAD - ((y - minY) / spanY) * (H - 2 * PAD);

  const byIndex = new Map(nodes.map((n) => [n.index, n]));

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      style={{ background: "var(--panel-2)", borderRadius: 10, border: "1px solid var(--border)" }}
    >
      {/* routes */}
      {candidate?.routes.map((route, ri) => {
        const color = ROUTE_COLORS[ri % ROUTE_COLORS.length];
        const pts = route
          .map((idx) => {
            const n = byIndex.get(idx);
            return n ? `${px(n.x)},${py(n.y)}` : "";
          })
          .filter(Boolean)
          .join(" ");
        return (
          <polyline
            key={ri}
            points={pts}
            fill="none"
            stroke={color}
            strokeWidth={2}
            strokeOpacity={0.85}
            strokeLinejoin="round"
          />
        );
      })}

      {/* customers */}
      {nodes
        .filter((n) => !n.depot)
        .map((n) => (
          <g key={n.index}>
            <circle cx={px(n.x)} cy={py(n.y)} r={5} fill="#9fb3c8" stroke="#0c1118" strokeWidth={1} />
          </g>
        ))}

      {/* depot */}
      {nodes
        .filter((n) => n.depot)
        .map((n) => (
          <g key="depot">
            <rect
              x={px(n.x) - 6}
              y={py(n.y) - 6}
              width={12}
              height={12}
              fill="var(--amber)"
              stroke="#0c1118"
              strokeWidth={1.5}
              transform={`rotate(45 ${px(n.x)} ${py(n.y)})`}
            />
          </g>
        ))}
    </svg>
  );
}
