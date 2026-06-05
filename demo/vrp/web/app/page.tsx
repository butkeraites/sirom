"use client";

import { useCallback, useEffect, useState } from "react";
import RouteMap from "./RouteMap";
import Frontier from "./Frontier";
import { Dataset, InstanceMeta, SolveResult } from "./types";

type Mode = "demand" | "travel_time";

// The VRP backend's base URL, reachable from the browser. Baked at build time
// (NEXT_PUBLIC_*). Defaults to the docker-compose published port.
const API = process.env.NEXT_PUBLIC_VRP_API || "http://localhost:8801";

export default function Page() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [instances, setInstances] = useState<string[]>([]);
  const [slug, setSlug] = useState("");
  const [name, setName] = useState("");
  const [mode, setMode] = useState<Mode>("demand");
  const [alpha, setAlpha] = useState(0.4);
  const [k, setK] = useState(8);
  const [scenarios, setScenarios] = useState(40);
  const [shiftFactor, setShiftFactor] = useState(1.1);

  const [preview, setPreview] = useState<InstanceMeta | null>(null);
  const [previewNote, setPreviewNote] = useState<string | null>(null);
  const [result, setResult] = useState<SolveResult | null>(null);
  const [selected, setSelected] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load the dataset catalog once.
  useEffect(() => {
    fetch(`${API}/vrp/datasets`)
      .then((r) => r.json())
      .then((d) => {
        setDatasets(d.datasets);
        if (d.datasets.length) setSlug(d.datasets[0].slug);
      })
      .catch(() => setError("Could not load the dataset catalog."));
  }, []);

  // Fetch the selected dataset's instances (downloaded lazily from its zip).
  useEffect(() => {
    if (!slug) return;
    let cancelled = false;
    setInstances([]);
    setName("");
    fetch(`${API}/vrp/datasets/${slug}/instances`)
      .then(async (r) => {
        if (!r.ok) throw new Error((await r.json()).detail || "instance list failed");
        return r.json();
      })
      .then((d) => {
        if (cancelled) return;
        setInstances(d.instances);
        setName(d.instances[0] ?? "");
        setError(null);
      })
      .catch((e) => !cancelled && setError(String(e.message || e)));
    return () => {
      cancelled = true;
    };
  }, [slug]);

  // Fetch a map preview whenever the selected instance or customer cap changes.
  useEffect(() => {
    if (!slug || !name) return;
    let cancelled = false;
    setResult(null);
    fetch(`${API}/vrp/instances/${slug}/${name}?k=${k}`)
      .then(async (r) => {
        if (!r.ok) throw new Error((await r.json()).detail || "preview failed");
        return r.json();
      })
      .then((d) => {
        if (cancelled) return;
        setPreview(d.instance);
        setPreviewNote(d.subsample_note);
        setError(null);
      })
      .catch((e) => !cancelled && setError(String(e.message || e)));
    return () => {
      cancelled = true;
    };
  }, [slug, name, k]);

  const solve = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(`${API}/vrp/solve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          slug, name, mode, alpha, k, n_scenarios: scenarios,
          shift_factor: shiftFactor,
        }),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail || "solve failed");
      setResult(body as SolveResult);
      setSelected(0);
    } catch (e: any) {
      setError(String(e.message || e));
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [slug, name, mode, alpha, k, scenarios, shiftFactor]);

  const nodes = result?.instance.nodes ?? preview?.nodes ?? [];
  const candidate = result?.candidates[selected] ?? null;
  const note = result?.subsample_note ?? previewNote;

  return (
    <div className="wrap">
      <header>
        <h1>
          Robust vehicle routing with <span className="accent">SIROM</span>
        </h1>
        <p>
          Pick a CVRP instance from{" "}
          <a href="http://www.vrp-rep.org/datasets.html" target="_blank" rel="noreferrer">
            VRP-REP
          </a>
          . The backend injects interval uncertainty (demand or travel time) and
          asks the SIROM API for a frontier of candidate routings — trading{" "}
          <b>total distance</b> against the <b>probability the plan stays
          feasible</b> under that uncertainty. Click a point to draw its routes.
        </p>
      </header>

      <div className="grid">
        {/* ---- controls ---- */}
        <div className="panel">
          <h2>Problem</h2>

          <div className="field">
            <label>Dataset</label>
            <select value={slug} onChange={(e) => setSlug(e.target.value)}>

              {datasets.map((d) => (
                <option key={d.slug} value={d.slug}>
                  {d.title}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label>Instance</label>
            <select value={name} onChange={(e) => setName(e.target.value)}>
              {instances.map((i) => (
                <option key={i} value={i}>
                  {i}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label>Uncertainty</label>
            <div className="toggle">
              <button
                className={mode === "demand" ? "active" : ""}
                onClick={() => setMode("demand")}
              >
                Demand
              </button>
              <button
                className={mode === "travel_time" ? "active" : ""}
                onClick={() => setMode("travel_time")}
              >
                Travel time
              </button>
            </div>
          </div>

          <div className="field">
            <label>
              Uncertainty level α <span className="val">±{Math.round(alpha * 100)}%</span>
            </label>
            <input
              type="range"
              min={0}
              max={0.5}
              step={0.01}
              value={alpha}
              onChange={(e) => setAlpha(parseFloat(e.target.value))}
            />
          </div>

          <div className="field">
            <label>
              Customers (nearest to depot) <span className="val">{k}</span>
            </label>
            <input
              type="range"
              min={4}
              max={12}
              step={1}
              value={k}
              onChange={(e) => setK(parseInt(e.target.value))}
            />
          </div>

          <div className="field">
            <label>
              Scenarios <span className="val">{scenarios}</span>
            </label>
            <input
              type="range"
              min={20}
              max={120}
              step={10}
              value={scenarios}
              onChange={(e) => setScenarios(parseInt(e.target.value))}
            />
          </div>

          {mode === "travel_time" && (
            <div className="field">
              <label>
                Shift tightness <span className="val">×{shiftFactor.toFixed(2)}</span>
              </label>
              <input
                type="range"
                min={0.7}
                max={1.6}
                step={0.05}
                value={shiftFactor}
                onChange={(e) => setShiftFactor(parseFloat(e.target.value))}
              />
              <div className="muted" style={{ fontSize: 11.5, marginTop: 4 }}>
                Scales the per-route shift limit. Lower = tighter = lower robustness.
              </div>
            </div>
          )}

          <button className="solve" onClick={solve} disabled={loading || !name}>
            {loading ? "Solving via SIROM…" : "Solve"}
          </button>

          {preview && (
            <div className="muted" style={{ marginTop: 14, lineHeight: 1.6 }}>
              Capacity <b>{preview.capacity}</b> · vehicles <b>{preview.vehicles}</b>
              <br />
              {preview.used_customers} of {preview.total_customers} customers
            </div>
          )}
        </div>

        {/* ---- results ---- */}
        <div className="panel">
          <h2>{result ? `${result.instance.name} — candidate routings` : "Map & frontier"}</h2>

          {error && <div className="error">{error}</div>}
          {note && <div className="note">{note}</div>}

          <div className="maprow">
            <div>
              <RouteMap nodes={nodes} candidate={candidate} />
              <div className="legend">
                <span>
                  <i style={{ width: 10, height: 10, background: "var(--amber)", transform: "rotate(45deg)" }} />{" "}
                  depot
                </span>
                <span>
                  <i className="swatch" style={{ background: "#9fb3c8" }} /> customer
                </span>
                {candidate &&
                  candidate.routes.map((_, i) => (
                    <span key={i}>
                      <i style={{ background: ["#2dd4bf", "#f59e0b", "#60a5fa", "#c084fc", "#f472b6", "#4ade80", "#fb7185", "#facc15"][i % 8] }} />{" "}
                      vehicle {i + 1}
                    </span>
                  ))}
              </div>
            </div>

            <div>
              {result ? (
                <>
                  <Frontier
                    candidates={result.candidates}
                    selected={selected}
                    onSelect={setSelected}
                  />
                  {candidate && (
                    <div className="stats">
                      <div className="stat">
                        <div className="k">Distance</div>
                        <div className="v">{candidate.objective.toFixed(1)}</div>
                      </div>
                      <div className="stat">
                        <div className="k">Robustness</div>
                        <div className="v teal">{(candidate.robustness * 100).toFixed(0)}%</div>
                      </div>
                      <div className="stat">
                        <div className="k">Vehicles</div>
                        <div className="v amber">{candidate.vehicles}</div>
                      </div>
                      {result.mode === "travel_time" && result.max_shift != null && (
                        <div className="stat">
                          <div className="k">Shift limit</div>
                          <div className="v">{result.max_shift}</div>
                        </div>
                      )}
                    </div>
                  )}
                </>
              ) : (
                <div className="placeholder">
                  {loading
                    ? "Sampling scenarios and solving each routing…"
                    : "Press Solve to build the robustness/distance frontier."}
                </div>
              )}
            </div>
          </div>

          {result?.warnings?.length ? (
            <div className="muted" style={{ marginTop: 14 }}>
              {result.warnings.map((w, i) => (
                <div key={i}>⚠︎ {w}</div>
              ))}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
