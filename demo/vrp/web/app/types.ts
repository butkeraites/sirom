export type Node = {
  index: number;
  id: number;
  x: number;
  y: number;
  demand: number;
  depot: boolean;
};

export type InstanceMeta = {
  dataset: string;
  name: string;
  capacity: number;
  vehicles: number;
  nodes: Node[];
  total_customers: number;
  used_customers: number;
};

export type Candidate = {
  objective: number;
  robustness: number;
  vehicles: number;
  routes: number[][]; // node indices, depot-bookended
};

export type SolveResult = {
  instance: InstanceMeta;
  mode: "demand" | "travel_time";
  alpha: number;
  shift_factor: number;
  max_shift: number | null;
  subsample_note: string | null;
  candidates: Candidate[];
  summary: Record<string, number>;
  warnings: string[];
};

export type Dataset = {
  slug: string;
  title: string;
  variant: string;
};

// Per-vehicle route colours.
export const ROUTE_COLORS = [
  "#2dd4bf",
  "#f59e0b",
  "#60a5fa",
  "#c084fc",
  "#f472b6",
  "#4ade80",
  "#fb7185",
  "#facc15",
];
