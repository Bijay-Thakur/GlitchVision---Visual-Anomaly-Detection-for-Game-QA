# GlitchVision — Model Card and AI Reflection

This card documents what GlitchVision is, where it can fail, and
how AI tooling shaped the way it was built. It complements the
[README](/README.md) and exists because *AI isn't just about
what works — it's about what's responsible.*

---

## System summary

- **Task.** Visual anomaly *triage* on gameplay and capture
  footage. The system ranks frames and intervals by how unusual
  they look so a human reviewer can jump straight to suspicious
  moments.
- **Inputs.** Gameplay videos via YouTube URL or local upload.
- **Outputs.** A ranked CSV of frames, a CSV of segments, a
  score-vs-time chart, top-K thumbnails, and a markdown run report.
- **Models.** Frozen pretrained ResNet-18 embeddings, an
  Isolation Forest for within-clip outlier detection, a kNN scorer
  against a curated reference bank, a hybrid normalized blend, and
  a small LogisticRegression classifier for the synthetic
  benchmark.
- **Out of scope.** Bug *classification*, root-cause analysis,
  performance / framerate measurement, audio analysis, or any
  judgment of player behavior or identity.

---

## Limitations and biases

- **Unsupervised ≠ bug detector.** The model flags *statistical*
  outliers, not bugs. Cutscenes, menus, fade-to-black frames, and
  rare-but-correct visual events can score high without being
  defects. Every flagged frame is a candidate for human review.
- **Pretrained-backbone bias.** ResNet-18 was pretrained on
  ImageNet, which is heavily biased toward natural daytime photos
  of objects, animals, and people. Stylized art (cel-shaded games,
  pixel art, abstract UIs) lives further from that distribution,
  which can both hide real glitches *and* fabricate spurious
  outliers.
- **Reference-bank coverage bias.** Reference kNN can only
  recognize scenes it has *seen*. If the bank skews toward, say,
  daytime outdoor combat, then nighttime indoor scenes will look
  anomalous even when they are perfectly correct gameplay. Bank
  curators effectively define what "normal" means.
- **Per-frame, single-backbone view.** The default model has no
  temporal awareness. Frozen animations, stuck physics, or
  synchronization regressions that look fine frame-by-frame are
  systematically under-reported. A lightweight temporal feature
  set partially mitigates this in the benchmark, but not in the
  default Streamlit app.
- **Synthetic benchmark is a proxy.** Injected glitches
  (brightness shift, blur, HUD blocks, freeze, reorder, etc.) are
  controllable and reproducible, but they do not represent the
  long tail of real engine bugs. Benchmark numbers are sanity
  checks, not production KPIs.
- **Recall vs precision skew.** On the synthetic benchmark
  precision is `1.000` and recall is `0.27`–`0.33` for thresholded
  classification. The system is conservative by design — it would
  rather miss a glitch than waste a reviewer's time — but anyone
  using the thresholded numbers as a *coverage* claim would be
  misled. The honest metrics are the ranking ones (`Precision@K`,
  `ROC-AUC`, `interval recall`).
- **Stream-format bias.** YouTube ingestion via `yt-dlp` + OpenCV
  works reliably for progressive / HLS streams but fails on
  DASH-only videos, which biases what footage is easy to evaluate.
- **CPU-only profile.** All numbers in the README are from a
  laptop CPU run. Performance and energy cost characterizations do
  not transfer to GPU or server environments.

---

## Misuse considerations and prevention

GlitchVision is built as a developer / QA triage tool. The most
plausible misuses, and how the system is designed to discourage
them, are:

- **Mistaking flagged frames for bugs in published claims.**
  Someone could screenshot a high-scoring frame from a competitor's
  build and call it a "bug detected by AI." *Prevention:*
  the README, the run `report.md`, and this model card all state
  that scores are statistical outliers, not defects. The output
  artifacts label the mode (within-clip vs reference vs hybrid)
  and never use the word "bug," only "anomalous" / "candidate."
- **Surveillance / player-identification creep.** A naive user
  might point GlitchVision at streaming footage of *people*
  rather than gameplay and try to flag "unusual" players.
  *Prevention:* the system has no face / person detection, no
  identity tracking, and the README explicitly scopes the project
  to gameplay and simulation footage. The reference bank is a
  bag-of-frames embedding store with no per-person index.
- **Overstated automation in QA pipelines.** A team could wire
  GlitchVision into CI, threshold its scores, and auto-fail
  builds. Given the recall numbers and the unsupervised framing,
  that would generate noise and false confidence.
  *Prevention:* the documentation insists on human-in-the-loop
  review, the default UI surfaces top-K *for review* (not pass /
  fail), and the synthetic benchmark explicitly distinguishes
  "engineering validation" from "production accuracy."
- **Privacy leakage via reference banks.** If someone built a
  reference bank from private internal builds and then shipped
  the bank, it could leak visual content.
  *Prevention:* `data/reference_banks/` is gitignored by default;
  the README calls out that banks may contain private content;
  banks store derived embeddings + a thumbnail grid, not full
  video.
- **Cost / abuse via uncapped runs.** Without limits, a user could
  point the tool at an arbitrarily long stream and exhaust a
  laptop's RAM or YouTube ToS allowance.
  *Prevention:* there is a hard `max_frames` cap, frame sampling
  is capped per video in the bank builder, the verifier rejects
  unavailable / age-gated / live URLs up front, and the source
  video is never persisted to disk.

These are mitigations, not guarantees. The tool is an aid, and the
team using it carries responsibility for how its outputs are
interpreted and acted on.

---

## What surprised me while testing reliability

A few things surprised me, in roughly increasing order of
usefulness to remember next time:

1. **Menus and fade-to-black frames are champion outliers.** The
   single most reliable "anomaly" the system finds in the wild is
   *not a bug* — it's a title card. That immediately reframed the
   project from "bug detector" to "triage aid" and changed how I
   wrote the report copy.
2. **Reference kNN is both the strongest ranker and the cheapest
   scorer.** I expected the trained classifier to dominate; instead
   reference kNN reached `ROC-AUC = 0.993` with `0.003 s` scoring
   latency because the embeddings were already computed. That was
   a real "the boring solution wins" moment.
3. **Precision-recall asymmetry.** Across all four benchmark
   models, precision sat at `1.000` while recall hovered around
   `0.27`–`0.33`. The models agreed strongly on what *was* an
   anomaly, but disagreed on how much of the anomaly *space* to
   commit to. This made me trust ranking metrics over thresholded
   ones for everything user-facing.
4. **DASH-only YouTube streams break OpenCV silently.** They look
   fine to `yt-dlp` metadata and only fail at the first
   `cap.read()`. Adding a one-frame OpenCV probe to the verifier
   turned a "mystery zero-frame run" into a clear up-front
   rejection — easily the highest reliability-per-line-of-code fix
   in the project.
5. **Profiling changed the design.** Adding `tracemalloc` and
   `psutil` reporting was meant as a recruiter-facing nicety, but
   the per-stage breakdown immediately exposed that frame
   *sampling* dominated runtime, not the model. That reordered
   the optimization plan from "bigger backbone?" to "smarter
   sampling."

---

## Collaboration with AI during this project

I built GlitchVision with heavy use of an AI coding assistant
(Anthropic's Claude, via Claude Code) for design discussions,
boilerplate generation, code review, and documentation drafting.
A few honest notes on how that went:

### One time the AI's suggestion was helpful

When I was deciding how to score per-frame anomalies in the
**reference** mode, I had drafted a single-centroid baseline:
embed every reference frame, average them, and score each
candidate frame by distance to the centroid. The AI pushed back,
pointing out that a real reference capture is multi-modal —
menus, cutscenes, combat, and idle gameplay all live in
different parts of the embedding space — and a single centroid
would falsely flag *any* underrepresented mode as anomalous. It
suggested a kNN-to-reference scorer with cosine distance and an
explicit `k` knob.

That suggestion shaped the final architecture. The reference
kNN scorer ended up being the strongest ranking model
(`ROC-AUC = 0.993`) **and** the fastest at scoring time
(`0.003 s`), and `k` is now the one knob in that mode that has a
clean, explainable meaning. The single-centroid version would
almost certainly have made the README's design-decisions section
much harder to defend.

### One time the AI's suggestion was flawed

While building the gameplay benchmark, I asked the AI to add a
"smarter" filter to remove non-gameplay frames (menus, loading
screens, fade-to-blacks) before scoring. It produced an
aggressive multi-rule filter combining low edge density, low
color variance, low pixel entropy, *and* a hard "near-uniform
frame" rule. On paper it looked clean. In practice it threw out
real gameplay — dark indoor scenes, stylized low-contrast art
shots, and any moment with a clear sky in frame all matched the
"static / low-variance" rule and were silently dropped. Coverage
of legitimate gameplay collapsed and the resulting reference
bank was biased toward bright, busy combat scenes, which then
made calm exploration footage look "anomalous" to the kNN
scorer.

I rolled the change back to a much narrower heuristic
(low-variance **and** low-edge **and** consecutive-near-identical
**only**, with a conservative threshold), and added the explicit
warning in the README that menu detection is *intentionally
limited*, because a false-positive filter is worse than no filter
at all. The lesson — which I want to remember for future projects
— is that AI suggestions to "improve a filter" need to be
evaluated on the population of things the filter will *wrongly
reject*, not just on a sample of things it correctly removes.

### How I worked with the AI overall

I treated the assistant as a fast, opinionated, occasionally
overconfident pair-programmer. It was excellent for unblocking
boilerplate (CLI argument parsing, dataclass scaffolding,
docstrings, error messages), for proposing design alternatives
under explicit trade-off prompts, and for drafting README and
docs prose I could then sharpen. It was less reliable on
problems that required intuition about the specific data
distribution — gameplay footage is unusual enough that
"reasonable defaults" suggested by the model often needed
empirical correction. I learned to trust it for *structure* and
*coverage*, and to verify it for *judgment*.

---

## Maintenance and contact

- **Owner.** Bijay (project author).
- **Issues.** Open a GitHub issue on this repository.
- **Update cadence.** This card is updated whenever a benchmark
  re-run materially changes the headline numbers, when a new
  scoring mode lands, or when a new bias or misuse vector is
  identified.
