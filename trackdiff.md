# AI Studio Export Diff Report

**Source:** `C:\Users\hemas\Downloads\st-act-cell-tracking-studio2` (exported from Google AI Studio "ST-ACT Cell Tracking Studio")
**Compared against:** this repo (`c:\Users\hemas\Downloads\st_act_pipeline`)
**Date:** 2026-07-03

---

## 1. What the export actually is

It is **not** a Python backend. It's a standalone **Vite + React 19 + Express + TypeScript** web app
(`package.json`: `react`, `vite`, `express`, `@google/genai`, `motion`, `tailwindcss`). Nothing in it
executes Python. The Python source you see in the AI Studio "Code" panel is **plain string content
embedded in `src/data.ts`** (`export const PIPELINE_FILES: CodeFile[] = [...]`, each entry
`{name, path, language, description, content}`), used only to (a) render in the `CodeExplorer` UI
tab and (b) get fed as chat context to a real Gemini API call. When the AI Studio chat "fixed"
`tracker.py`, it edited that string — it never ran or tested it.

### Project layout
```
st-act-cell-tracking-studio2/
├── server.ts            # Express server: dev/prod static serving + /api/chat -> Gemini
├── src/
│   ├── App.tsx           # Tab shell (Configurator / CodeExplorer / CellVisualizer / AssistantChat / KaggleSubmitter / SubmissionAnalyzer)
│   ├── data.ts            # PIPELINE_FILES: embedded text copies of requirements.txt, data_loader.py, model.py, tracker.py, run_pipeline.py
│   ├── simulator.ts        # generateSimulation() / buildLineageTree() - fabricates fake cell tracks for the visualizer
│   ├── components/
│   │   ├── Configurator.tsx      # RAM/VRAM sizing calculator (heuristic formulas, no real measurement)
│   │   ├── CodeExplorer.tsx      # File-tree viewer over PIPELINE_FILES + a themed fake "terminal"
│   │   ├── CellVisualizer.tsx    # 2300-line animated 3D-ish cell viewer, driven entirely by simulator.ts fake data
│   │   ├── AssistantChat.tsx     # Real Gemini chat via /api/chat (only genuinely "live" feature)
│   │   ├── KaggleSubmitter.tsx   # 2760-line panel: fake pipeline-run simulator, kaggle.json generator, project-scaffold script generator, troubleshooting guide
│   │   └── SubmissionAnalyzer.tsx # 1900-line submission CSV "analyzer" (heuristic-based, not verified against real logic)
├── system_diagnostics.py   # loose copy, trivially different from repo's
├── .env.example            # GEMINI_API_KEY, APP_URL
```

---

## 2. Python file diff (the part that matters for the actual pipeline)

| File | Result |
|---|---|
| `requirements.txt` | identical |
| `src/data_loader.py` | **byte-identical** to repo |
| `src/model.py` | **byte-identical** to repo |
| `system_diagnostics.py` | trivially different (one print-statement variant), not material |
| `src/tracker.py` | **1 real, substantive fix** — see §2.1 |
| `run_pipeline.py` | **1 real, performance-only fix** — see §2.2 |

None of the five critical blockers identified in `PRD.md` §4.1 (wrong submission schema, no real
data ingestion, model never trained/wired, hardcoded motion vectors, ILP scalability) were touched.

### 2.1 `src/tracker.py` — flow-conservation constraint fix (real bug, worth porting)

**Repo (`src/tracker.py:141-145`):**
```python
# Every detected node must be explained by exactly one incoming flow (a transition or a birth)
prob += (b_vars[n] + pulp.lpSum([y_vars[e] for e in incoming]) == 1)

# Every detected node must resolve to exactly one outgoing flow, unless it dies or divides
prob += (pulp.lpSum([y_vars[e] for e in outgoing]) + d_vars[n] == 1 + s_vars[n])
```

**AI Studio (`data.ts:569-573`):**
```python
# A node can have at most one incoming flow (either a transition or a birth)
prob += (b_vars[n] + pulp.lpSum([y_vars[e] for e in incoming]) <= 1)

# Lineage structural constraint: Outflow limit is 1 normally, 2 if cell is dividing
prob += (pulp.lpSum([y_vars[e] for e in outgoing]) <= 1 + s_vars[n])
```

**Why the repo version is a real bug:** the `==` formulation forces *every* detected node to be
explained by exactly one incoming flow and exactly one outgoing flow. If a node has no plausible
neighbor within the distance/pruning gates (expected on real noisy data — e.g. a detection at the
very start/end of the movie, or an isolated false positive), the solver is forced to satisfy both
equalities simultaneously:
- `incoming == 1` with zero real incoming edges ⟹ `b_vars[n] = 1` (forced birth)
- `outgoing == 1 + s_vars[n]` with zero real outgoing edges and `s_vars[n] = 0` ⟹ `d_vars[n] = 1` (forced death)

But `b_vars[n] + d_vars[n] <= 1` is *also* a hard constraint (line 154/582 in both versions) — so
`b=1 AND d=1` is disallowed. **Any isolated node makes the whole ILP infeasible.** On the real
competition data (sparse ground truth, imperfect detections), isolated/orphan detections are
inevitable, so this would likely break `prob.solve()` outright rather than just degrade quality.

The AI Studio version relaxes both to `<=`, allowing a node to simply be unexplained (zero
incoming, zero outgoing) without triggering the birth+death contradiction. This is a legitimate
correctness fix, not a stylistic change.

**Side effect to note:** relaxing to `<=` also decouples `d_vars` from being forced whenever a
node has no outgoing flow — so `death_cost` in the objective is weaker at actually penalizing
"unexplained drop-offs" (a node can just have 0 outgoing edges "for free" without the solver ever
needing to set `d_vars[n]=1`). This is a reasonable trade for feasibility, but worth knowing: the
current formulation (even after the fix) doesn't strongly incentivize the solver to *prefer*
explained continuations over silent drop-offs beyond the transition-cost terms already in the
objective.

### 2.2 `run_pipeline.py` — vectorized peak extraction (perf only, not a logic fix)

**Repo (`run_pipeline.py:56-68`):**
```python
def extract_peaks_from_volume(vol: np.ndarray, threshold=0.4, offset_bias=0.0):
    peaks = []
    nz, ny, nx = vol.shape
    for z in range(nz):
        for y in range(4, ny-4, 8):
            for x in range(4, nx-4, 8):
                if vol[z, y, x] > threshold:
                    peaks.append([float(z), float(y) + offset_bias, float(x) + offset_bias])
    return peaks
```

**AI Studio (`data.ts:728-749`):**
```python
def extract_peaks_from_volume(vol: np.ndarray, threshold=0.4, offset_bias=0.0):
    nz, ny, nx = vol.shape
    z_indices = np.arange(nz)
    y_indices = np.arange(4, ny-4, 8)
    x_indices = np.arange(4, nx-4, 8)

    if len(y_indices) == 0 or len(x_indices) == 0:
        return []

    zz, yy, xx = np.meshgrid(z_indices, y_indices, x_indices, indexing='ij')
    values = vol[zz, yy, xx]
    mask = values > threshold

    z_hits = zz[mask]
    y_hits = yy[mask].astype(float) + offset_bias
    x_hits = xx[mask].astype(float) + offset_bias

    return np.column_stack([z_hits, y_hits, x_hits]).tolist()
```

Same stride-8 grid sampling, same threshold logic, mathematically equivalent output — just
computed via NumPy meshgrid/masking instead of a Python triple-nested loop. Meaningfully faster on
large volumes, but it's still the same naive fixed-stride **placeholder** — it does not call the
real `STACTCentroidPredictor` model, so it doesn't address PRD blocker #3 (model never wired).
Also added a defensive early-return for degenerate volume sizes (`len(y_indices)==0`), which the
loop version didn't need since it just wouldn't iterate — harmless, not a behavior change.

Everything else in `run_pipeline.py` is unchanged: hardcoded `anisotropy = np.array([5.0, 1.0, 1.0])`
(should be `4.0:1:1` per real physical spacing — see PRD §4.2), hardcoded motion vectors
`[0.05, 0.2, 0.3]` for every cell (PRD blocker #4), and the wrong submission schema
`Time,TrackID,ParentTrackID,Z,Y,X` (PRD blocker #1).

---

## 3. UI code inventory — what's real vs. simulated

This matters because several panels *look* like they're running the real pipeline but aren't:

| Component | What it actually does | Real or simulated? |
|---|---|---|
| `AssistantChat.tsx` | Sends chat messages to `server.ts`'s `/api/chat`, which calls the real Gemini API (`gemini-3.5-flash`) with a "Kaggle Grandmaster" system prompt | **Real** (needs `GEMINI_API_KEY`) |
| `Configurator.tsx` | Sliders for T/Z/Y/X/dtype/CNN-channels feed hardcoded arithmetic formulas (e.g. `estStandardRAM = sizeTotalGB * 1.6 + 0.8`) to estimate RAM/VRAM and flag OOM risk | Heuristic estimate, not measured — reasonable as a rough planning tool, but the formulas are guesses, not profiled numbers |
| `CodeExplorer.tsx` | Renders `PIPELINE_FILES` (the embedded text) as a browsable file tree with a themed fake terminal | Display only |
| `CellVisualizer.tsx` | Animates cells using `simulator.ts`'s `generateSimulation()` — fully fabricated synthetic cell positions/lineages, `motion/react` animation | **Entirely fake data**, not connected to any real Zarr volume or model output |
| `KaggleSubmitter.tsx` | "Runner" tab: a `setInterval`-driven fake log stream with **scripted messages and a hardcoded scoring formula** (`edgeJaccard = 0.9840` baseline, minus penalties per toggle) that fabricates a plausible-looking Jaccard score. Also contains a genuinely useful **project-scaffold generator** (`generateKaggleCellCode()`) that writes `PIPELINE_FILES` out to `~/Downloads/st_act_pipeline/...` as real `.py` files, a client-side `kaggle.json` credential file generator, and a troubleshooting guide/log-schema reference | **The "pipeline run" and its score are 100% theater** — no Python executes, no real volume is read, the score is computed by a made-up formula reacting to UI toggle state. The scaffold-generator and kaggle.json helper are real, useful utilities. |
| `SubmissionAnalyzer.tsx` | Not fully read (1900 lines) — named/typed around `Submission` and `HeuristicUpgrade` interfaces, suggesting it evaluates a submission CSV against heuristic rules | Likely heuristic/fabricated like `KaggleSubmitter`'s runner — **not yet confirmed real or fake**, flagged for follow-up if you want it checked |

**Important:** the score shown in your screenshot's UI (and anything `KaggleSubmitter`'s
"End-to-End Execution" tab reports) is **not a real evaluation of your pipeline** — it's a
hardcoded formula reacting to which toggles are enabled, deliberately designed to look like a
Kaggle Jaccard score. Don't use it as a signal for actual pipeline quality.

---

## 4. Recommendation

1. **Port the two real Python fixes into this repo** (§2.1 tracker.py constraint relaxation, §2.2
   run_pipeline.py vectorization) — both are safe, verified improvements.
2. **Do not treat AI Studio's "pipeline run" score as real** — it's fabricated in
   `KaggleSubmitter.tsx`. All real validation must happen against the actual competition metric
   (PRD §3.3 / FR-5), not this UI.
3. The UI's genuinely useful piece is the **project-scaffold generator** — it can regenerate the
   Python file layout from `PIPELINE_FILES`, which is handy if you ever want to bootstrap a fresh
   copy, but since it's generated from the same (slightly stale) embedded strings, this repo is
   already ahead of it once §2.1/§2.2 are ported.
4. If useful, the UI shell (Configurator/CodeExplorer/AssistantChat) could be kept as a companion
   dev tool, but `CellVisualizer` and `KaggleSubmitter`'s runner/scoring should not be presented as
   reflecting real pipeline behavior without being rewired to actual data.
