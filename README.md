# GlitchVision

**Visual regression triage for gameplay and simulation footage.**
CPU-first. ResNet-18 embeddings + Isolation Forest, reference-based
kNN, and a hybrid blend, with segment-level analysis and a synthetic
benchmark. Streamlit UI.

GlitchVision surfaces the frames and intervals of a capture that look
suspicious — either relative to the rest of the same clip, or relative
to a known-good reference build. It is a triage aid: it tells a human
reviewer *where to look*, not *what is broken*.

---

## Why this matters

Manual QA of gameplay footage does not scale. Human reviewers can't
realistically watch every second of every build, and most automated
approaches either need labeled glitch data (which is scarce) or tight
engine integration (which an external team does not have).

GlitchVision offers a practical middle ground:

- **Build-over-build QA.** Capture reference footage from the last
  known-good build, then flag frames and segments of a new build that
  drift away from it.
- **Simulation / capture review.** Surface unusual frames in long
  capture sessions without pre-defining failure modes.
- **General visual anomaly triage.** Anywhere "this frame doesn't
  belong" is a useful signal.

---

## Architecture

Three scoring modes are exposed behind one pipeline API:

1. **Within-clip (Isolation Forest).** Flags frames that are
   statistically unusual inside the same clip. No reference needed.
2. **Reference distance (kNN).** Per-frame anomaly score is the mean
   cosine distance to the `k` nearest neighbors in a precomputed
   reference embedding bank.
3. **Hybrid.** Min-max normalizes each of the two scores above and
   blends them with configurable weights (default `0.5 / 0.5`).

On top of any mode, per-frame scores are aggregated into
**non-overlapping segments** so reviewers can jump directly to
suspicious intervals instead of scrubbing a timeline.

```
       Reference videos                     Candidate video
              |                                    |
              v                                    v
      +-----------------+                 +-----------------+
      | FrameExtractor  |                 | FrameExtractor  |
      | (1 fps, 224x224)|                 | (1 fps, 224x224)|
      +--------+--------+                 +--------+--------+
               |                                   |
               v                                   v
      +-----------------+                 +-----------------+
      | Backbone        |                 | Backbone        |
      | (ResNet-18 /    |                 | (same backbone) |
      |  DINO / CLIP)   |                 |                 |
      +--------+--------+                 +--------+--------+
               |                                   |
               v                                   |
      +-----------------+                          |
      |  ReferenceBank  |                          |
      |  embeddings.npz |                          |
      |  metadata.json  |                          |
      +--------+--------+                          |
               |                                   |
               +-----------------+-----------------+
                                 |
              +------------------+------------------+
              |                  |                  |
              v                  v                  v
      +---------------+  +-----------------+  +---------------+
      | Within-clip   |  | Reference kNN   |  | Hybrid blend  |
      | Isolation F.  |  | cosine / L2     |  | normalized    |
      +-------+-------+  +--------+--------+  +-------+-------+
              |                   |                   |
              +---------+---------+---------+---------+
                        |                   |
                        v                   v
              +-----------------+  +-----------------+
              | Top-K frames    |  | Top segments    |
              +--------+--------+  +--------+--------+
                       \                   /
                        \                 /
                         v               v
                  +--------------------------+
                  | anomalies.csv           |
                  | segments.csv            |
                  | score_plot.png          |
                  | report.md               |
                  | frames/rank*.jpg        |
                  +--------------------------+
```

Pipeline orchestration lives in `src/pipeline/pipeline.py`.

---

## Repo layout

```
glitchvision/
├── app/
│   ├── main.py                 # Streamlit UI (mode selector + ref bank mgmt)
│   └── config.py
├── src/
│   ├── ingestion/              # YouTube stream resolution + local upload
│   ├── processing/             # frame sampling
│   ├── features/               # pluggable backbone (resnet18 / dino / clip)
│   ├── models/
│   │   ├── anomaly_detector.py # Isolation Forest wrapper
│   │   ├── reference_scorer.py # kNN distance to reference bank
│   │   └── hybrid_scorer.py    # normalized blend
│   ├── reference/              # durable reference embedding bank
│   ├── reporting/              # markdown run report
│   ├── benchmark/              # synthetic glitch injection + metrics
│   ├── pipeline/               # end-to-end orchestration
│   └── utils/                  # IO, scoring, segments, visualization
├── tests/                      # unit + end-to-end tests
├── data/
│   ├── samples/                # your own clips (gitignored)
│   ├── reference_banks/        # saved banks (gitignored)
│   └── outputs/                # per-run outputs (gitignored)
├── .github/workflows/ci.yml
├── requirements.txt
├── run_app.py                  # one-liner launcher
└── README.md
```

---

## Tech stack

- **Python** 3.11+ (tested through 3.14)
- **PyTorch / torchvision** (CPU build) — ResNet-18 backbone
- **scikit-learn** — Isolation Forest
- **OpenCV** — frame sampling + resize
- **NumPy** — embedding math, kNN distances, score normalization
- **Streamlit** — demo UI
- **yt-dlp** — YouTube stream resolution
- **matplotlib** — score-vs-time plot
- **pytest** — unit + end-to-end test suite
- **GitHub Actions** — CI for the lightweight unit tests

---

## Setup

Tested target: **Windows 10/11, macOS, Linux; Python 3.11–3.14; CPU
only; 8 GB RAM.**

```powershell
# 1. Create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # (Windows PowerShell)
# source .venv/bin/activate      # (macOS/Linux)

# 2. Install PyTorch CPU build
pip install torch torchvision

# 3. Install the rest
pip install -r requirements.txt
```

The default path only needs the dependencies in `requirements.txt`.
Optional alternate backbones are not required:

- `dino` loads `dino_vits16` via `torch.hub` (internet on first load).
- `clip` requires `pip install git+https://github.com/openai/CLIP.git`.

If an optional backbone fails to load, the app logs a warning and
falls back to ResNet-18 automatically.

---

## Running the app

```powershell
python run_app.py
```

The Streamlit UI opens in your browser.

1. Pick a **Scoring mode**:
   - *Within-clip (baseline)* — no reference needed.
   - *Reference distance* — requires a saved reference bank.
   - *Hybrid* — requires a saved reference bank.
2. Pick an **Input source**:
   - *YouTube URL* (primary) — resolved via `yt-dlp`; the source video
     is never persisted to disk.
   - *Local upload (fallback)* — for offline use or when a YouTube
     stream is not OpenCV-compatible.
3. For reference / hybrid modes, either **load an existing bank** from
   the dropdown or expand **Build a new reference bank** and upload
   one or more known-good clips.
4. Tune frame interval, top-K, contamination, segment window, etc.
5. Click **Run anomaly detection**.
6. Review the score plot, top anomalous frames, and top segments
   in-page. Use the **Download** buttons to save individual artifacts
   to your machine — nothing is downloaded automatically.

### Build a reference bank programmatically

```python
from src.pipeline import GlitchVisionPipeline, PipelineConfig

pipe = GlitchVisionPipeline(PipelineConfig(interval_sec=1.0, backbone="resnet18"))
bank = pipe.build_reference(
    [("data/samples/known_good_run1.mp4", "known_good_run1"),
     ("data/samples/known_good_run2.mp4", "known_good_run2")],
    out_dir="data/reference_banks/known_good_v1",
)
print("Bank size:", bank.size)
```

### Run a candidate clip in reference mode

```python
from src.pipeline import GlitchVisionPipeline, PipelineConfig
from src.reference import ReferenceBank

bank = ReferenceBank.load("data/reference_banks/known_good_v1")
pipe = GlitchVisionPipeline(PipelineConfig(
    interval_sec=1.0,
    mode="reference_distance",
    top_k=12,
    reference_k=5,
))
result = pipe.run(
    video_source="data/samples/candidate_build.mp4",
    source_type="local_upload",
    source_label="candidate_build.mp4",
    reference_bank=bank,
)
print("Run dir:", result.run_dir)
```

### Run the synthetic benchmark

The benchmark utilities let you sanity-check the pipeline on known
corrupted intervals. Ground truth is the injection schedule itself.

```python
from src.benchmark import plan_glitch_schedule, inject_glitches, evaluate_run

# `clean_frames` is a list of BGR uint8 frames sampled from a clean clip.
schedule = plan_glitch_schedule(n_frames=len(clean_frames), n_intervals=3, seed=0)
corrupted, intervals = inject_glitches(clean_frames, schedule, seed=0)

# Write `corrupted` to a file and run the pipeline on it, then:
metrics = evaluate_run(
    top_frame_indices=[r.frame_index for r in result.top_records],
    ground_truth=intervals,
    n_sampled_frames=result.total_sampled_frames,
    top_segment_ranges=[(s.start_frame, s.end_frame) for s in result.top_segments],
)
print(metrics)
```

Benchmark numbers are only meaningful relative to a specific clip,
schedule, and seed, so the utilities are shipped without canned scores.

---

## Output artifacts

Each run creates a timestamped folder under `data/outputs/run_<ts>/`:

| File | Description |
| --- | --- |
| `anomalies.csv`    | Per-sampled-frame scores (rank, timestamp, raw + normalized scores, within/reference components, mode). |
| `segments.csv`     | Top anomalous segments with start/end time and representative frame. |
| `score_plot.png`   | Score-vs-time curve with top-K frames highlighted. |
| `report.md`        | Human-readable run summary (config, top frames, top segments, limitations). |
| `frames/rank*.jpg` | Thumbnail JPEGs of the top-K anomalous frames. |

Reference banks live under `data/reference_banks/<name>/` as
`embeddings.npz` + `metadata.json`.

Run outputs, sample clips, and reference banks are all **gitignored** —
they are user-specific and regenerated per run.

---

## Run tests

```powershell
pytest -q
```

The suite covers:

- scoring and I/O utilities,
- reference-bank save/load round-trip,
- kNN reference scorer,
- hybrid blend math,
- segment aggregation,
- synthetic glitch injection and benchmark metrics,
- report builder,
- end-to-end run in within-clip mode,
- end-to-end run in reference and hybrid modes on a synthetic clip.

CI (`.github/workflows/ci.yml`) runs the lightweight unit tests on
every push / PR. The heavy end-to-end tests (which pull torch and
OpenCV) are run locally against your `.venv`.

---

## Technical design decisions

- **ResNet-18 default.** 512-D pooled features, ~11M params, fast on
  CPU, robust ImageNet representation. Good enough for frame-level
  outlier detection without a training loop.
- **Pluggable backbone registry.** `dino` and `clip` are wired via a
  small factory so future experimentation is easy. *No backbone is
  trained from scratch in this project;* the optional paths use
  pretrained checkpoints.
- **Isolation Forest for within-clip scoring.** Unsupervised, fast on
  CPU, one meaningful knob (`contamination`), well-understood
  baseline.
- **kNN for reference-distance scoring.** Real reference captures are
  multi-modal (menus, cutscenes, combat). kNN naturally respects the
  modes; a single-centroid model would not. Its one hyperparameter
  (`k`) has a clear meaning.
- **L2-normalized embeddings.** Make cosine and Euclidean distance
  interchangeable up to a monotone transform and keep the distance
  matrix numerically stable.
- **Segment-level aggregation.** Reviewers care about intervals, not
  isolated frames. Non-overlapping windows map 1:1 to segment IDs on
  disk and avoid top-K dedup confusion.
- **YouTube ingestion via resolved stream URL.** No persistent
  download of the source video. If OpenCV can't open the chosen
  format, the app fails loudly and points the user at the
  local-upload fallback.
- **`max_frames` safety cap.** A hard rail against runaway runs on a
  laptop.

---

## Limitations

- **Unsupervised ≠ bug detector.** Scores are *statistical* outliers.
  Cutscenes, menus, and fade-to-black frames can score high without
  being bugs. Every output is a candidate for human review.
- **Per-frame backbone.** ResNet-18 sees each frame independently;
  true motion anomalies (stuck animations, frozen physics) need
  temporal modeling (see roadmap).
- **Synthetic benchmark is a proxy.** Injected glitches are not a
  substitute for real QA data; benchmark numbers are sanity checks,
  not production KPIs.
- **Reference generalization is bounded.** A reference bank only
  covers the scenes it has actually seen; genuinely new content will
  look anomalous to the reference scorer.
- **YouTube stream compatibility varies.** DASH-only videos still
  require the local-upload fallback.
- **CPU-only by default.** GPU support is a one-line change
  (`device="cuda"`) but is not officially tested here.

---

## Roadmap

- **Temporal modeling** — short-window embedding deltas or a small
  temporal head to catch motion-related regressions.
- **Better stream ingestion** — thin FFmpeg wrapper for DASH streams.
- **Segment-level contact sheets** — one image per top segment for
  faster human triage.
- **Reference bank curation** — filtering and deduplication so banks
  stay compact as more known-good captures accumulate.
- **Optional fine-tune hook** — a light contrastive head on top of the
  frozen backbone, trained on domain-specific footage.

---
