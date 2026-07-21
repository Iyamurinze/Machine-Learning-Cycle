# Malaria Cell Classification — End-to-End ML Pipeline

Detecting *Plasmodium falciparum* parasites in segmented red blood cell images,
from raw data through to a deployed, load-tested, retrainable service.

<!-- TODO: fill in before submission -->
| | |
|---|---|
| **Live dashboard** | `TODO_UI_URL` |
| **API + Swagger docs** | `TODO_API_URL/docs` |
| **Video demo** | `TODO_YOUTUBE_URL` |
| **Repository** | `TODO_GITHUB_URL` |

---

## Contents

- [The problem](#the-problem)
- [What this project does](#what-this-project-does)
- [Results](#results)
- [Data insights](#data-insights)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Running the pipeline](#running-the-pipeline)
- [Deployment](#deployment)
- [Flood request simulation](#flood-request-simulation)
- [API reference](#api-reference)
- [Design decisions](#design-decisions)

---

## The problem

Malaria killed an estimated **597,000 people in 2023**, overwhelmingly in
sub-Saharan Africa. The diagnostic gold standard is unchanged in a century: a
trained microscopist examines a Giemsa-stained blood smear and counts parasites
by eye. It is accurate in expert hands and unreliable outside them — 20–30
minutes per slide, years of training required, and accuracy that degrades with
technician fatigue.

This project automates the per-cell decision: given one segmented red blood
cell, is it **Parasitized** or **Uninfected**?

## What this project does

1. **Acquires** the NIH Malaria dataset (27,558 expert-annotated images) with a
   single script — no Kaggle account needed.
2. **Analyses** it, deriving interpretable features from raw pixels and ranking
   them by effect size.
3. **Trains** a compact custom CNN with regularization, hyperparameter tuning,
   early stopping and LR scheduling.
4. **Evaluates** on 5,512 held-out images across six metrics.
5. **Serves** the model through a FastAPI application.
6. **Presents** it in a Streamlit dashboard with uptime, visualizations,
   prediction and retraining.
7. **Retrains** on user-uploaded data, continuing from the deployed model.
8. **Load-tests** the deployment across 1, 2 and 4 containers.

---

## Results

<!-- TODO: replace with the figures printed by the notebook -->

### Model performance on 5,512 held-out test images

| Metric | Score |
|---|---|
| **Accuracy** | **95.19%** |
| Precision | 92.44% |
| Recall | 98.44% |
| F1 score | 95.34% |
| ROC-AUC | 99.00% |
| Loss | 0.1753 |

Reported per class, from the clinical perspective where "positive" means a
detected infection:

| | Parasite detection |
|---|---|
| Precision | 98.33% |
| Recall | 91.94% |

Trained for 8 epochs (best epoch 7, restored by early stopping) with
`learning_rate=5e-4, dropout=0.3, l2=1e-4`.

The test set is exactly balanced (2,756 per class), so accuracy is a meaningful
headline figure rather than a misleading one.

**Recall is the metric that matters clinically.** A missed malaria case can
progress to cerebral malaria within 24 hours; a false alarm costs one
confirmatory test. The model is evaluated with that asymmetry in mind, and
per-class parasite detection figures are reported separately in the notebook.

### Model characteristics

| | |
|---|---|
| Architecture | Custom CNN, 3 convolutional blocks |
| Parameters | 297,121 (1.13 MB) |
| Input | 64×64×3 |
| Training data | 22,046 images |
| Format | `models/malaria_cnn.keras` |

---

## Data insights

Full analysis in [`notebook/malaria_classification.ipynb`](notebook/malaria_classification.ipynb).

The NIH images ship as raw pixels with **no accompanying metadata** — there is
no CSV of patient age or parasite density to plot. So rather than visualise
metadata that does not exist, the analysis *derives* features from the pixels,
each corresponding to something a microscopist actually looks for. Separation is
quantified with **Cohen's *d*** (0.2 small, 0.5 medium, 0.8 large).

Measured on 1,500 images per class:

| Feature | Parasitized | Uninfected | Cohen's *d* | Effect |
|---|---|---|---|---|
| Texture variation (`intensity_std`) | 9.8113 | 6.4440 | **1.17** | large |
| Dark-pixel ratio | 0.0085 | 0.0003 | **0.95** | large |
| Mean saturation | 0.2622 | 0.1996 | 0.64 | medium |
| Mean intensity | 164.10 | 169.62 | −0.58 | medium |
| Cell area ratio | 0.7102 | 0.7247 | −0.24 | small |

### 1. Texture variation — infection is legible as disorder

A healthy red blood cell is a smooth biconcave disc filled uniformly with
haemoglobin; its pixel intensities cluster tightly. When *Plasmodium* invades it
adds a chromatin body, digests haemoglobin into dark hemozoin crystals, and
disrupts the cytoplasm — so brightness scatters far more widely.

**The story:** the parasitized distribution is not merely shifted but visibly
**right-skewed with a long tail** — heavier parasite load, more chaotic
interior. This is the largest effect in the dataset and it explains why a CNN is
the right tool: convolutions detect local texture, which *is* the signal, while
a model fed only average colour would discard it.

### 2. Dark-pixel ratio — a ~30× relative gap

The parasite's physical footprint, measured as the fraction of the cell markedly
darker than **that cell's own median** rather than a global cutoff. That detail
is what makes the feature survive real data: slides vary in staining strength
and illumination, and a fixed threshold would flag every under-exposed slide as
infected — a failure mode that would surface as a whole clinic reading positive.

**The story:** roughly **28× more dark pixels** in parasitized cells, the widest
relative separation measured. On a log scale it becomes clear that a large share
of uninfected cells contain *exactly zero* dark pixels. This is closer to a
presence/absence test than a continuous measurement.

### 3. Stain uptake — the cleanest mechanism, the weakest signal

Giemsa stain binds nucleic acid. A mature red blood cell is one of the few cells
in the body with **no nucleus**, so it carries no DNA, takes up little stain, and
stays a flat desaturated pink. *Plasmodium* brings its own genome and the cell
turns purple.

**The story, and the honest caveat:** the mechanism is the clearest of the three
but the separation is the weakest (*d* = 0.57 against 1.12 for texture). Stain
uptake also depends on how long the slide sat in the bath and how fresh the
reagent was — variance with nothing to do with the patient. **A classifier
relying on colour alone would be measuring laboratory procedure as much as
pathology**, and would degrade on slides from a different clinic.

### A documented negative result

Cell area was expected to help — infection distorts and enlarges cells. It
barely does (*d* = −0.24, small, and the wrong sign for the hypothesis: infected
cells came out marginally *smaller*). The NIH pipeline
**segmented and cropped each cell tightly**,
normalising away the very size variation the feature was meant to capture. The
signal was destroyed upstream, before publication. It is reported rather than
quietly dropped, because it demonstrates that preprocessing decisions silently
remove information — and that effect sizes are worth computing *before*
modelling, not after.

---

## Project structure

```
Machine-Learning-Cycle/
│
├── README.md
├── requirements.txt
├── Dockerfile                 # API image
├── Dockerfile.ui              # Streamlit image
├── docker-compose.yml         # local stack + scaling for the load test
├── nginx.conf                 # load balancer across API replicas
├── render.yaml                # Render deployment blueprint
│
├── notebook/
│   └── malaria_classification.ipynb
│
├── src/
│   ├── data_acquisition.py    # download, extract, stratified split
│   ├── preprocessing.py       # datasets, augmentation, feature extraction
│   ├── model.py               # architecture, training, retraining
│   └── prediction.py          # inference
│
├── api/
│   └── main.py                # FastAPI service
│
├── ui/
│   └── app.py                 # Streamlit dashboard
│
├── locust/
│   ├── locustfile.py
│   └── README.md              # load test procedure
│
├── data/
│   ├── train/                 # 22,046 images (gitignored, script-generated)
│   ├── test/                  # 5,512 images (gitignored)
│   └── uploads/               # staged retraining data
│
└── models/
    ├── malaria_cnn.keras
    └── metadata.json          # training history and metrics
```

**Note on `data/`.** The dataset is ~1 GB across raw and split copies, well past
what GitHub accepts, so the images are gitignored. They are fully reproducible
with one command (below) — nothing is lost.

---

## Setup

Requires **Python 3.10** and, for the containerised path, Docker.

```bash
git clone TODO_GITHUB_URL
cd Machine-Learning-Cycle

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Download the data

```bash
python src/data_acquisition.py
```

Streams 353 MB directly from the NIH host — **no Kaggle account or API token
required** — then extracts and performs a stratified 80/20 split. Takes a few
minutes. Verify:

```
test/Parasitized    2,756
test/Uninfected     2,756
train/Parasitized  11,023
train/Uninfected   11,023
TOTAL              27,558
```

---

## Running the pipeline

### Train the model

Open and run the notebook end to end:

```bash
jupyter notebook notebook/malaria_classification.ipynb
```

Or headlessly:

```bash
python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=7200 \
    notebook/malaria_classification.ipynb
```

This writes `models/malaria_cnn.keras` and `models/metadata.json`.

### Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

Swagger UI at <http://localhost:8000/docs>.

### Start the dashboard

```bash
export API_URL=http://localhost:8000
streamlit run ui/app.py
```

At <http://localhost:8501>.

### Predict from the command line

```bash
python src/prediction.py data/test/Parasitized/<any-file>.png
```

### Everything at once, in containers

```bash
docker compose up --build
```

Dashboard on `:8501`, API on `:8080` (through the load balancer).

---

## Deployment

Deployed on **Render** from `render.yaml` (New → Blueprint → point at this
repository). Two services are created: the API and the dashboard, with the
dashboard's `API_URL` wired to the API automatically.

**Free-tier behaviour:** instances sleep after 15 minutes idle and cold-start in
roughly 30–60 seconds, most of it TensorFlow importing. Wake the service before
demoing or load-testing.

### Evaluating the model in production

The deployed service exposes its own evaluation trail:

- `GET /metrics` returns the live model's metrics *and* the full history across
  every retraining, so drift is visible rather than assumed.
- `GET /status` reports uptime, model version, and staged upload counts.
- The dashboard's **Overview** page charts metrics by model version — each
  retrain appends a point, making regressions obvious.

Metrics are always measured against the same held-out test set, so figures
remain comparable across versions.

---

## Flood request simulation

Full procedure in [`locust/README.md`](locust/README.md).

```bash
docker compose up -d --scale api=1
locust -f locust/locustfile.py --host http://localhost:8080 \
    --users 100 --spawn-rate 10 --run-time 2m --headless \
    --html locust/reports/report_1container.html
```

Repeat with `--scale api=2` and `--scale api=4`.

The load test **asserts prediction correctness**, not just HTTP 200 — a
container serving a broken or half-loaded model is recorded as a failure, which
a liveness-only test would score as perfectly healthy.

### Results

**Setup:** 100 concurrent users, spawning at 10/second, 2 minutes per run, all
traffic through nginx. Measured on an Intel Core i7-8569U @ 2.80 GHz —
**4 physical cores** (8 logical), 16 GB RAM, macOS 15.6.1. Each replica capped
at 1 CPU with TensorFlow pinned to a single thread.

| Containers | Requests | HTTP failures | Median | p95 | p99 | Requests/s |
|---|---|---|---|---|---|---|
| 1 | 2,341 | **0** | 3,000 ms | 3,900 ms | 4,300 ms | 19.6 |
| 2 | 4,257 | **0** | 640 ms | 1,500 ms | 1,900 ms | **35.7** |
| 4 | 4,116 | **0** | **280 ms** | 3,500 ms | 5,900 ms | 34.7 |

`/predict` alone — the endpoint that actually runs the CNN:

| Containers | Predictions | Median | p95 |
|---|---|---|---|
| 1 | 1,448 | 2,900 ms | 3,700 ms |
| 2 | 2,630 | 510 ms | 1,300 ms |
| 4 | 2,554 | **270 ms** | 2,900 ms |

### Interpretation

**Zero HTTP failures at every scale.** No timeouts, no 502s, no dropped
connections. The failure counts Locust reports (25 / 34 / 29) are entirely
*model misclassifications* — the load test asserts that the returned class
matches the image's true label, so the model's own ~1.5% error rate on the
request pool registers as "failures". A liveness-only test would have reported
a flat 0% and told you nothing about whether the service was still correct
under load. Notably, ~75% of those misses are `said Uninfected, expected
Parasitized` — false negatives, consistent with parasite recall (91.94%) being
the model's weaker metric.

**1 → 2 containers: near-linear gain.** Throughput rises 82% (19.6 → 35.7 rps)
and median latency drops 79% (3,000 → 640 ms). One container at 1 CPU is
comprehensively saturated by ~40 offered rps; the second replica roughly halves
the queue.

**2 → 4 containers: median improves, throughput plateaus, tail degrades.**
Median latency halves again (640 → 280 ms) but throughput is flat (35.7 → 34.7
rps) and p99 nearly triples (1,900 → 5,900 ms). This is the host running out of
physical cores. Four replicas at 1 CPU each claim the entire 4-core machine,
leaving nothing for nginx, Docker, and the Locust process itself. Requests that
get scheduled promptly are served fast — hence the excellent median — but
requests that arrive when all cores are busy wait considerably longer, which is
exactly what a degrading tail with a flat throughput ceiling looks like.

**The practical conclusion:** on this hardware ~35 rps is the ceiling, and it is
a property of the host rather than the application. Two containers is the
efficient operating point; the fourth replica buys a better median at the cost
of predictability. On a machine with more cores the plateau would move out
proportionally.

> **A methodological note.** An earlier version of `nginx.conf` used a
> conventional `upstream { server api:8000; }` block. nginx resolves that
> hostname *once at startup* and pins to a single replica IP, so all three runs
> silently hit the same container and produced near-identical numbers that
> looked like "scaling gives no benefit". The tell was container memory: three
> of four replicas sat at 42 MB, having never loaded the model. The fix is a
> `resolver` directive plus a **variable** in `proxy_pass`, which forces
> re-resolution. If you adapt this setup, do not inline the hostname back into
> `proxy_pass` — it reintroduces the bug, and it fails silently.

---

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Service metadata |
| `GET` | `/health` | Liveness probe |
| `GET` | `/status` | Uptime, model version, staged uploads |
| `GET` | `/metrics` | Live model metrics + history |
| `GET` | `/visualizations` | Derived feature distributions |
| `POST` | `/predict` | Classify one image |
| `POST` | `/predict/batch` | Classify several images |
| `POST` | `/upload` | Stage labelled data for retraining |
| `GET` | `/uploads` | Staged data counts |
| `POST` | `/retrain` | Trigger retraining |
| `GET` | `/retrain/status` | Retraining progress |

### Example

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@data/test/Parasitized/example.png"
```

```json
{
  "prediction": "Parasitized",
  "confidence": 0.9873,
  "probabilities": {"Parasitized": 0.9873, "Uninfected": 0.0127},
  "threshold": 0.5,
  "latency_ms": 41.2
}
```

### The retraining flow

1. **Upload** — a ZIP containing `Parasitized/` and `Uninfected/` folders, or
   individual images with an explicit label. Archive entries are validated on
   extraction rather than blanket-extracted, which enforces the label structure
   and prevents a malicious archive writing outside its destination.
2. **Stage** — images are saved to `data/uploads/<class>/`.
3. **Preprocess** — the uploaded images go through the same resize, rescale and
   augmentation pipeline as the original training data.
4. **Retrain** — the saved model is **loaded as a pre-trained starting point**
   and fine-tuned at a reduced learning rate (1e-4 rather than 1e-3). Training
   from scratch would discard everything already learned; hitting loaded weights
   with the original learning rate would wash it out.
5. **Evaluate & swap** — the retrained model is scored on the same held-out test
   set, metrics are appended to `models/metadata.json`, and the in-memory model
   is hot-swapped without a restart.

Retraining runs on a **background thread**. It takes minutes; holding the HTTP
connection open would time out behind most cloud load balancers and block the
worker from serving predictions the whole time.

By default the staged uploads are **mixed with a sample of the original
training data** (750 images per class). Fine-tuning on a small new batch alone
causes catastrophic forgetting — the model overfits the handful of new images
and loses the baseline distribution. Replaying the *entire* 22,046-image
training set would be the safest option but would make one retrain as expensive
as training from scratch, defeating the point of fine-tuning; a few thousand
replayed images anchors the distribution at a fraction of the cost.

### Measured retraining run

80 new labelled images uploaded as a ZIP, then retrained for 3 epochs at
`lr=1e-4`:

| | v1 (from notebook) | v2 (after retrain) |
|---|---|---|
| Accuracy | 95.19% | **96.39%** |
| F1 score | 95.34% | **96.43%** |
| Loss | 0.1753 | **0.1441** |
| ROC-AUC | 99.00% | 99.13% |

**Wall-clock time: 1 minute 57 seconds**, start to finish, including
evaluation on the full 5,512-image test set and hot-swapping the live model.
Both versions are retained in `models/metadata.json` and charted by model
version on the dashboard's Overview page.

---

## Design decisions

**Why a custom CNN rather than a fine-tuned ImageNet backbone.** ImageNet
features encode object parts assembling into wheels and faces; segmented
single-cell microscopy shares almost none of that structure. ResNet expects
224×224, so a 130 px cell would have to be upsampled, inventing information the
sensor never captured. And ResNet50 is ~25M parameters against 297K here —
**84× larger** — which would make retraining-inside-a-request impossible. The
custom network reaches high accuracy anyway, leaving little for a heavier model
to recover.

**Why 64×64 input.** Tested visually in the notebook: at 64 px the parasite's
chromatin dot is still clearly visible; at 32 px the smaller ones smear into the
cytoplasm. 64 px is where resolution is sufficient but training stays fast
enough to retrain on CPU-only cloud hardware.

**Why rescaling lives inside the model.** The `Rescaling` layer is part of the
model graph, so the API hands over raw `[0, 255]` pixels and *cannot forget* to
normalise. A train/serve normalisation mismatch is among the most common and
hardest-to-spot production ML bugs; this makes it structurally impossible.

**Why global average pooling instead of flatten.** A flatten after the final
8×8×128 block would feed 8,192 values into the dense layer — ~524K parameters in
one layer, more than the entire current model. Global average pooling reduces
that to 128 values and 8K parameters, and is the single biggest reason the model
generalises.

**Why the UI holds no model.** The dashboard is a pure API client. That keeps
the UI image at ~400 MB instead of 1.5 GB, and means the load test measures the
API rather than Streamlit.

---

## Acknowledgements

Dataset: [NIH / U.S. National Library of Medicine — Malaria Cell Images](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html).
Cells were segmented from Giemsa-stained thin blood smears collected at
Chittagong Medical College Hospital, Bangladesh, and annotated by an expert
slide reader at the Mahidol-Oxford Tropical Medicine Research Unit.

Rajaraman et al. (2018), *Pre-trained convolutional neural networks as feature
extractors toward improved malaria parasite detection in thin blood smear
images*, PeerJ 6:e4568.
