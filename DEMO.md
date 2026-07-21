# Video demo walkthrough

The rubric awards full marks for *"clear and user-friendly demonstration of the
prediction and retraining process, with camera on."* Both must appear, and your
face must be visible. Target 5–8 minutes.

## Before you hit record

```bash
# 1. Start the API
uvicorn api.main:app --port 8000

# 2. Start the dashboard (separate terminal)
export API_URL=http://localhost:8000
streamlit run ui/app.py
```

Checklist:

- [ ] Camera on, face visible in the corner
- [ ] Microphone tested — poor sound loses marks
- [ ] `data/uploads/` is empty, so the staged count starts at 0
- [ ] Have `demo_images.txt` open — it lists images the model classifies
      correctly with the highest confidence
- [ ] Close IntelliJ and anything heavy; a laggy screen recording looks bad
- [ ] Screen resolution scaled up so code and numbers are readable on video

---

## The script

### 1. Introduce the problem (~45s)

Say what the project does and why it matters. Something like:

> "Malaria killed around 597,000 people in 2023, mostly in sub-Saharan Africa.
> Diagnosis still depends on a trained microscopist examining a blood smear by
> eye — 20 to 30 minutes per slide. This project automates the per-cell
> decision: given one red blood cell, is it infected or not."

Mention the dataset: **NIH, 27,558 expert-annotated images, perfectly balanced.**

### 2. Dashboard overview — uptime and metrics (~45s)

Open the **Overview** page. Point out:

- The service is online and its **uptime**
- **Model version** — note it says v2, because it has already been retrained once
- The six evaluation metrics
- The **metrics-by-version chart**, showing v1 → v2 improvement

Say explicitly: *"this is the production evaluation trail — every retraining
appends a point, so regressions would be visible immediately."*

### 3. Data insights — the three features (~90s)

Open **Data insights**, press **Compute feature distributions**.

Walk through all three, and give the interpretation, not just the chart:

**Texture variation** — *"a healthy cell is a smooth uniform disc; a parasite
adds a chromatin body and disturbs the cytoplasm, so intensities scatter. This
is the largest effect in the dataset, Cohen's d of 1.17."*

**Dark-pixel ratio** — *"this is the parasite's physical footprint. Infected
cells have about 28 times more dark pixels. And crucially it's measured against
each cell's own median, not a fixed threshold — otherwise an under-exposed
slide would read as a whole clinic full of infected patients."*

**Stain uptake** — *"Giemsa stain binds DNA. A mature red blood cell has no
nucleus, so it takes up little stain. The parasite brings its own genome, so
the cell turns purple. Cleanest biology of the three, but the weakest
separation — which is why relying on colour alone would be measuring lab
procedure as much as pathology."*

If you have time, mention the **negative result**: cell area didn't work,
because NIH's own cropping normalised the size difference away.

### 4. Prediction — REQUIRED (~90s)

Open **Predict**.

Do this **twice**, once per class, using files from `demo_images.txt`:

1. Upload a **Parasitized** image → say the true label out loud *before*
   clicking, then click **Classify**
2. Point at the result: predicted class, confidence, the probability bars,
   and the latency in milliseconds
3. Repeat with an **Uninfected** image

The listed demo images classify correctly at ~100% confidence, so this will
land cleanly on camera.

**Optional but impressive:** upload one of the borderline cells and show a
low-confidence result, then explain that the model's errors concentrate on
early-stage infections where the parasite is only a few pixels across — and
that the notebook's error analysis confirmed this.

### 5. Upload and retrain — REQUIRED (~2min)

Open **Upload & retrain**.

1. Expand **"Build a test archive from the existing test set"**, set 25 images
   per class, click **Generate archive**, then **Download**
2. Switch to the **Bulk ZIP archive** tab and upload that file
3. Point out the staged count going from 0 to 50 — *"this is the data being
   saved for retraining"*
4. Set epochs to 3, keep **"Mix in the original training data"** checked, and
   explain why: *"training only on the new batch would cause catastrophic
   forgetting — the model would overfit these 50 images and lose the baseline"*
5. Click **Start retraining**
6. While it runs, explain what is happening: *"it loads the currently deployed
   model as a pre-trained starting point and fine-tunes it at a tenth of the
   original learning rate, rather than starting from scratch"*
7. Press **Refresh status** until it completes — takes about 2 minutes
8. Show the new metrics, then go back to **Overview** and show the version
   chart now has a third point

### 6. Wrap up (~30s)

Show quickly:

- The **API docs** at `/docs` (Swagger UI)
- The **notebook**, scrolling through the evaluation section
- The **Locust report** with the container scaling results

---

## What loses marks

| Mistake | Cost |
|---|---|
| Camera off | Drops from 5 to 3 |
| Poor audio | Drops from 5 to 3 |
| Only showing prediction, not retraining | Drops from 5 to 3 |
| Only showing retraining, not prediction | Drops from 5 to 3 |
| No video at all | 1 |

Both processes must be visible, end to end, with your camera on.
