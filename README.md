# GlitchVision

**Unsupervised visual anomaly detection for gameplay footage.**
ResNet-18 embeddings + Isolation Forest. CPU-only. Streamlit UI.

GlitchVision is a practical MVP that helps a human QA reviewer triage long
gameplay videos by surfacing the frames that *look different from everything
else* in the clip — the kind of frames a reviewer would want to eyeball for
graphical corruption, rendering glitches, stuck animations, or UI bugs.

> This is a CodePath showcase MVP. It is honest about its scope:
> frame-level, unsupervised, CPU-only. It is not a "real-time production QA"
> system, and it doesn't claim to be state of the art.

---

## Why this matters

Manual QA of gameplay footage is expensive and incomplete. Human reviewers
can't realistically watch every second of every build. Most automated QA
tools either require labeled glitch datasets (which don't exist) or tight
engine integration (which an intern / student team doesn't have).

An unsupervised pipeline gives you a **useful middle ground**:

- **Game QA** — surface the 10 weirdest frames in a 15-minute playtest
  recording for a human to look at.
- **Simulation & robotics** — flag unusual sensor/visualization frames
  without pre-defining failure modes.
- **AR/VR content review** — catch rendering hiccups in long capture sessions.
- **General anomaly monitoring** — any workflow where "this frame doesn't
  belong" is a useful signal.

The tool doesn't tell you *what* a glitch is; it tells you *where to look*.

---

## Architecture

```
+----------------------+     +----------------------+     +-------------------------+
|   YouTube URL        |     |   Local video upload |     |   Streamlit UI controls |
|   (primary path)     |     |   (fallback path)    |     |   interval / K / etc.   |
+----------+-----------+     +----------+-----------+     +-------------+-----------+
           |                            |                               |
           v                            v                               |
+------------------------+   +------------------------+                 |
| yt-dlp resolves a      |   | temp file written to   |                 |
| progressive MP4 stream |   | the OS temp dir        |                 |
+-----------+------------+   +-----------+------------+                 |
            \                           /                               |
             \                         /                                |
              v                       v                                 |
          +----------------------------------+                          |
          | OpenCV VideoCapture              |                          |
          | sample 1 frame/sec, resize 224px | <------------------------+
          +------------------+---------------+
                             |
                             v
          +----------------------------------+
          | ResNet-18 (torchvision, eval()) |
          | strip FC head -> 512-D vector   |
          +------------------+---------------+
                             |
                             v
          +----------------------------------+
          | IsolationForest (scikit-learn)   |
          | fit on embeddings, score each    |
          +------------------+---------------+
                             |
                             v
          +----------------------------------+
          | rank top-K, smooth, save outputs |
          |   - anomalies.csv                |
          |   - frames/rank01_*.jpg          |
          |   - score_plot.png               |
          +----------------------------------+
```

See `src/pipeline/pipeline.py` for the orchestration code.

---

## Repo layout

```
glitchvision/
├── app/
│   ├── main.py          # Streamlit UI
│   └── config.py
├── src/
│   ├── ingestion/       # YouTube stream + local upload
│   ├── processing/      # frame sampling
│   ├── features/        # ResNet-18 embedding extractor
│   ├── models/          # Isolation Forest wrapper
│   ├── pipeline/        # end-to-end orchestration
│   └── utils/           # IO, scoring, visualization
├── tests/
│   ├── test_utils.py    # unit tests
│   └── test_smoke.py    # end-to-end synthetic-video test
├── data/
│   ├── samples/         # (empty; for your local test clips)
│   └── outputs/         # per-run output folders live here
├── requirements.txt
├── run_app.py           # one-liner launcher
└── README.md
```

---

## Setup

Tested target: **Windows 10/11, Python 3.10–3.12, CPU only, 8 GB RAM.**

```powershell
# 1. Create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install PyTorch CPU build from the official index (Windows-friendly)
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# 3. Install the rest
pip install -r requirements.txt
```

macOS / Linux are equivalent (use `source .venv/bin/activate`). The
PyTorch step can also be run without the custom index on Linux/macOS — the
CPU wheels are the default there.

---

## Running the app

```powershell
python run_app.py
```

This opens the Streamlit UI in your browser. From there:

1. Choose **YouTube URL** (primary) or **Local upload (fallback)** in the
   sidebar.
2. Adjust frame interval, top-K, contamination, and max frames.
3. Click **Run anomaly detection**.
4. Review the top-K anomaly gallery, score plot, and full CSV.
5. Download `anomalies.csv` or open the timestamped folder under
   `data/outputs/` for raw artifacts.

### Example usage (local file)

```powershell
# drop a clip into data/samples/ then use Local upload in the UI,
# or call the pipeline directly:
python -c "from src.pipeline import GlitchVisionPipeline, PipelineConfig; \
r = GlitchVisionPipeline(PipelineConfig(interval_sec=1.0, top_k=10)) \
.run('data/samples/your_clip.mp4', 'local_upload', 'your_clip.mp4'); \
print(r.run_dir)"
```

### Run tests

```powershell
pytest -q
```

The smoke test generates a tiny synthetic MP4, runs the full pipeline on it,
and checks that CSV + top-anomaly JPEGs are written correctly.

---

## Technical design decisions

- **ResNet-18 over heavier backbones.** 512-D embeddings, ~11M params,
  runs fast on CPU. Plenty discriminative for frame-level outlier detection
  on a laptop. No need for video-specific models for an MVP.
- **Isolation Forest over autoencoders / one-class SVMs.** No training loop,
  no hyperparameter grief, well-understood baseline. Fits in seconds on
  hundreds of embeddings.
- **YouTube ingestion via resolved stream URL, not full download.**
  `yt-dlp` gives us a playable progressive MP4 URL; OpenCV reads frames
  from it directly. We don't persist the source video on disk. If a given
  YouTube format isn't OpenCV-readable, we fail loudly and point the user
  at the local upload fallback instead of faking success.
- **Frame interval default = 1 fps.** The right middle ground between
  missing short anomalies and melting an 8 GB laptop. Configurable in the
  UI.
- **`max_frames` cap = 600.** Hard safety rail so a 2-hour video doesn't
  eat all your RAM. Tunable in the sidebar.
- **Higher score = more anomalous.** Scikit-learn's `score_samples` is
  "higher = more normal"; we flip the sign so the UI reads naturally.
- **Optional moving-average smoothing and min-gap dedup.** Cheap wins
  against noisy adjacent duplicates in the top-K gallery. Off by default
  isn't a thing — smoothing = 3, min_gap = 2 in the defaults.

---

## Limitations

- **Frame-level only.** The model sees each frame independently; it has no
  notion of motion, continuity, or audio.
- **Unsupervised means "statistically unusual," not "actually broken."** A
  dramatic cutscene, a HUD popup, a fade-to-black — all of these can score
  high. The tool is an *aid to human review*, not a replacement for it.
- **False positives are expected.** Tune contamination and top-K to taste.
- **YouTube stream compatibility varies.** Some videos only expose DASH
  fragments that OpenCV can't open directly. When that happens the UI
  says so and offers the local-upload path.
- **ResNet-18 is a general-purpose backbone.** A game-specific fine-tune
  would help — out of scope for this MVP.
- **No GPU acceleration wired in.** Intentional: the target machine is an
  8 GB CPU-only laptop. Moving to CUDA is mostly a one-line change
  (`device="cuda"`) if a GPU is available.

---

## Future roadmap

- **Temporal modeling** — e.g. short-window embedding deltas, or a small
  temporal conv over pooled features, to catch *motion* anomalies (stuck
  animations, frozen physics) that a per-frame model misses.
- **Better stream ingestion** — fragmented/DASH stream support via a thin
  FFmpeg wrapper, so DASH-only YouTube videos work without local upload.
- **Synthetic glitch benchmarking** — procedurally inject artifacts
  (texture corruption, color shifts, missing meshes) into clean footage to
  get a *quantitative* precision/recall number instead of a vibes check.
- **CI / build integration** — run GlitchVision on each build's QA capture
  and attach the top-K report to the build artifact.
- **Game-domain fine-tune** — a light contrastive fine-tune of the
  backbone on a few hours of "normal" footage from the target title.

---

## Resume bullets (honest versions)

- Built **GlitchVision**, an end-to-end MVP that flags visual anomalies in
  gameplay footage using pretrained ResNet-18 embeddings and Isolation
  Forest, packaged behind a Streamlit UI and runnable on a CPU-only laptop.
- Designed a streaming YouTube ingestion path (yt-dlp → OpenCV via resolved
  MP4 URL) that analyzes videos without persisting the source file on disk.
- Shipped the full pipeline (sampling, embedding, anomaly ranking, CSV +
  gallery outputs, smoke tests) in under a week under a strict no-GPU,
  minimal-dependency constraint.
- Explicitly chose Isolation Forest over a custom deep anomaly model for
  time, honesty, and demo reliability — documented the trade-off in the
  README.

---

## License

MIT (or your project's default — adjust before publishing).
