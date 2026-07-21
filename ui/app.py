"""Streamlit dashboard for the malaria cell classification pipeline.

Four pages, matching the four things the deployed solution has to do:
    Overview   — service uptime, live model version, evaluation metrics
    Insights   — derived feature distributions and what they mean
    Predict    — classify a single uploaded cell image
    Retrain    — stage bulk labelled data and trigger retraining

The dashboard is a pure client of the FastAPI service; it holds no model of
its own. That separation is deliberate — it means the UI container stays
small, and the load test measures the API rather than Streamlit.
"""

from __future__ import annotations

import io
import os
import zipfile

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

def _resolve_api_url() -> str:
    """Normalise the configured API address into a usable base URL.

    Render's `fromService` wiring injects a bare `host:port` with no scheme,
    which `requests` rejects outright — the dashboard would report the API
    unreachable on every page. Rather than depend on whoever sets the
    environment variable getting the scheme right, missing schemes are filled
    in here: loopback gets http, anything else gets https.
    """
    raw = os.getenv("API_URL", "http://localhost:8000").strip().rstrip("/")

    if "://" not in raw:
        host = raw.split(":", 1)[0]
        local = host in {"localhost", "127.0.0.1", "0.0.0.0"}
        raw = f"{'http' if local else 'https'}://{raw}"

    # A host:443 suffix is redundant once the scheme says https, and some
    # proxies dislike the explicit port.
    if raw.startswith("https://") and raw.endswith(":443"):
        raw = raw[: -len(":443")]

    return raw


API_URL = _resolve_api_url()
REQUEST_TIMEOUT = 30

# Categorical slots 1 and 2 from the reference palette, validated for
# colour-vision deficiency against both the light and dark chart surfaces.
# Every chart also carries a legend and direct labels, so class identity is
# never communicated by colour alone.
COLOR_PARASITIZED = "#2a78d6"
COLOR_UNINFECTED = "#008300"
CLASS_COLORS = {
    "Parasitized": COLOR_PARASITIZED,
    "Uninfected": COLOR_UNINFECTED,
}

GRID_COLOR = "rgba(128,128,128,0.20)"
AXIS_COLOR = "rgba(128,128,128,0.65)"

FEATURE_LABELS = {
    "intensity_std": "Texture variation (intensity std. dev.)",
    "dark_pixel_ratio": "Dark-pixel ratio (parasite chromatin)",
    "mean_saturation": "Mean saturation (Giemsa stain uptake)",
    "mean_intensity": "Mean intensity (overall brightness)",
    "cell_area_ratio": "Cell area ratio (fraction of frame)",
}

st.set_page_config(
    page_title="Malaria Cell Classifier",
    page_icon=":material/biotech:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Presentation layer.
#
# Icons come from Streamlit's built-in Material Symbols (`:material/name:`)
# rather than emoji. Emoji render differently on every platform, carry a
# cartoon tone at odds with a diagnostic tool, and cannot be colour-matched to
# the interface. Material Symbols are vectors that inherit the current text
# colour and stay legible at small sizes.
CUSTOM_CSS = """
<style>
  /* Tighten the default vertical rhythm — Streamlit ships very loose. */
  .block-container { padding-top: 2.5rem; max-width: 1400px; }

  h1 { font-weight: 650; letter-spacing: -0.02em; }
  h2 { font-weight: 600; letter-spacing: -0.01em; margin-top: 0.4rem; }
  h3 { font-weight: 600; font-size: 1.05rem; }

  /* Metric cards: a hairline border reads as structure without the visual
     weight of a filled panel. */
  div[data-testid="stMetric"] {
    background: color-mix(in srgb, currentColor 3%, transparent);
    border: 1px solid color-mix(in srgb, currentColor 12%, transparent);
    border-radius: 10px;
    padding: 0.85rem 1rem;
  }
  div[data-testid="stMetricLabel"] p {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    opacity: 0.65;
  }
  div[data-testid="stMetricValue"] {
    font-size: 1.55rem;
    font-weight: 620;
    letter-spacing: -0.01em;
  }

  /* Status pill: a dot plus a word, so state is never colour-alone. */
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.86rem;
    font-weight: 550;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    border: 1px solid color-mix(in srgb, currentColor 18%, transparent);
  }
  .status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    box-shadow: 0 0 0 3px color-mix(in srgb, currentColor 18%, transparent);
  }
  .dot-live  { background: #12a150; color: #12a150; }
  .dot-down  { background: #d03b3b; color: #d03b3b; }
  .dot-idle  { background: #9a9a9a; color: #9a9a9a; }

  /* Prediction verdict card. */
  .verdict {
    border-radius: 12px;
    padding: 1.15rem 1.3rem;
    border: 1px solid;
    margin-bottom: 0.5rem;
  }
  .verdict-title {
    font-size: 1.35rem; font-weight: 650;
    letter-spacing: -0.01em; margin: 0 0 0.15rem 0;
  }
  .verdict-sub { font-size: 0.85rem; opacity: 0.75; margin: 0; }
  .verdict-positive {
    border-color: color-mix(in srgb, #d03b3b 45%, transparent);
    background: color-mix(in srgb, #d03b3b 10%, transparent);
  }
  .verdict-negative {
    border-color: color-mix(in srgb, #12a150 45%, transparent);
    background: color-mix(in srgb, #12a150 10%, transparent);
  }

  /* Sidebar nav: quieter label, roomier hit targets. */
  section[data-testid="stSidebar"] .stRadio label { padding: 0.15rem 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def status_pill(label: str, state: str = "live") -> str:
    """Render a status pill. `state` is one of live / down / idle."""
    return (
        f'<span class="status-pill">'
        f'<span class="status-dot dot-{state}"></span>{label}</span>'
    )


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_get(path: str, quiet: bool = False, **kwargs):
    """GET from the API, returning None on failure.

    `quiet` suppresses the inline error banner, for callers that render their
    own failure state — the sidebar health check shows a status pill, and a
    stacked error message underneath it would be redundant noise.
    """
    try:
        response = requests.get(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as error:
        if not quiet:
            st.error(
                f"Could not reach the API at `{API_URL}{path}` — {error}",
                icon=":material/cloud_off:",
            )
        return None


def api_post(path: str, **kwargs):
    """POST to the API, surfacing the server's error detail when there is one."""
    try:
        response = requests.post(f"{API_URL}{path}", timeout=300, **kwargs)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            st.error(f"{response.status_code} — {detail}")
            return None
        return response.json()
    except requests.exceptions.RequestException as error:
        st.error(f"Could not reach the API at `{API_URL}{path}` — {error}")
        return None


def style_axes(figure: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    """Apply consistent, recessive chart chrome.

    Backgrounds are transparent and grid/axis colours are mid-grey alphas so a
    single figure reads correctly against both the light and dark Streamlit
    themes without maintaining two palettes.
    """
    figure.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title=x_title,
        yaxis_title=y_title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=AXIS_COLOR, size=12),
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="closest",
        height=380,
    )
    figure.update_xaxes(gridcolor=GRID_COLOR, zeroline=False, linecolor=AXIS_COLOR)
    figure.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, linecolor=AXIS_COLOR)
    return figure


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_overview() -> None:
    st.title("Malaria Cell Classifier")
    st.caption(
        "Detecting *Plasmodium* parasites in segmented red blood cell images — "
        "NIH dataset, 27,558 images, balanced across two classes."
    )

    status = api_get("/status")
    if status is None:
        st.warning(
            f"The API is not responding. Start it with "
            f"`uvicorn api.main:app --reload` or check `API_URL` (currently "
            f"`{API_URL}`).",
            icon=":material/cloud_off:",
        )
        return

    online = status["status"] == "online"

    st.markdown("### Service status")
    st.markdown(
        status_pill("Online" if online else "Offline", "live" if online else "down"),
        unsafe_allow_html=True,
    )
    st.write("")

    col1, col2, col3 = st.columns(3)
    col1.metric("Uptime", status["uptime_human"])
    col2.metric("Model version", f"v{status['model_version']}")
    col3.metric("Model file", "Loaded" if status["model_available"] else "Missing")

    staged = status.get("staged_uploads", {})
    if sum(staged.values()):
        st.info(
            f"**{sum(staged.values())} image(s) staged for retraining** — "
            + ", ".join(f"{count} {label}" for label, count in staged.items()),
            icon=":material/inventory_2:",
        )

    st.divider()
    st.subheader("Live model performance")

    payload = api_get("/metrics")
    if payload is None or not payload.get("metrics"):
        st.warning("No evaluation metrics recorded yet. Train the model first.")
        return

    metrics = payload["metrics"]
    st.caption(
        f"Recorded {payload['recorded_at']} from the `{payload['event']}` event, "
        "measured on the held-out test set of 5,512 images."
    )

    columns = st.columns(6)
    display_order = [
        ("accuracy", "Accuracy", True),
        ("precision", "Precision", True),
        ("recall", "Recall", True),
        ("f1_score", "F1 score", True),
        ("auc", "ROC-AUC", True),
        ("loss", "Loss", False),
    ]
    for column, (key, label, as_percent) in zip(columns, display_order):
        value = metrics.get(key)
        if value is None:
            column.metric(label, "—")
        elif as_percent:
            column.metric(label, f"{value:.2%}")
        else:
            column.metric(label, f"{value:.4f}")

    history = payload.get("history", [])
    if len(history) > 1:
        st.divider()
        st.subheader("Metrics across retrainings")
        st.caption(
            "How the model has moved each time it was retrained on newly "
            "uploaded data. This is the production evaluation trail."
        )

        frame = pd.DataFrame(
            [
                {
                    "Version": f"v{row['version']}",
                    "Event": row["event"],
                    **{
                        key: row["metrics"].get(key)
                        for key in ("accuracy", "precision", "recall", "f1_score")
                    },
                }
                for row in history
            ]
        )

        figure = go.Figure()
        for key, color in (
            ("accuracy", COLOR_PARASITIZED),
            ("f1_score", COLOR_UNINFECTED),
        ):
            figure.add_trace(
                go.Scatter(
                    x=frame["Version"],
                    y=frame[key],
                    name=key.replace("_", " ").title(),
                    mode="lines+markers+text",
                    text=[f"{value:.1%}" if value else "" for value in frame[key]],
                    textposition="top center",
                    line=dict(color=color, width=2),
                    marker=dict(size=9, color=color),
                )
            )
        st.plotly_chart(
            style_axes(figure, "Test metrics by model version", "", "Score"),
            use_container_width=True,
        )
        st.dataframe(frame, use_container_width=True, hide_index=True)


def page_insights() -> None:
    st.title("Data insights")
    st.caption(
        "The NIH images ship as raw pixels with no accompanying metadata, so "
        "these features are computed from the pixels themselves. Each one is a "
        "measurable property of a cell that a clinician would also look for "
        "down a microscope."
    )

    sample_size = st.slider(
        "Images sampled per class", min_value=100, max_value=1000, value=400, step=100
    )

    if st.button("Compute feature distributions", type="primary"):
        st.session_state.pop("insights", None)
        with st.spinner("Extracting features from the dataset…"):
            st.session_state["insights"] = api_get(
                "/visualizations", params={"per_class": sample_size}
            )

    payload = st.session_state.get("insights")
    if payload is None:
        st.info("Press **Compute feature distributions** to analyse the dataset.")
        return

    frame = pd.DataFrame(payload["records"])
    summary = payload["summary"]

    st.success(
        f"Analysed {payload['sample_size']:,} images "
        f"({payload['per_class']} per class)."
    )

    def separation(feature: str) -> float:
        """Cohen's d — mean gap in pooled standard deviations."""
        parasitized = frame[frame.label == "Parasitized"][feature]
        uninfected = frame[frame.label == "Uninfected"][feature]
        pooled = ((parasitized.var() + uninfected.var()) / 2) ** 0.5
        if pooled == 0:
            return 0.0
        return float((parasitized.mean() - uninfected.mean()) / pooled)

    # --- Feature 1 -------------------------------------------------------
    st.divider()
    st.subheader("1. Texture variation — the strongest single signal")

    figure = go.Figure()
    for label, color in CLASS_COLORS.items():
        figure.add_trace(
            go.Violin(
                y=frame[frame.label == label]["intensity_std"],
                name=label,
                line_color=color,
                fillcolor=color,
                opacity=0.55,
                box_visible=True,
                meanline_visible=True,
                points=False,
            )
        )
    st.plotly_chart(
        style_axes(
            figure,
            "Within-cell intensity variation by class",
            "",
            "Std. dev. of pixel intensity",
        ),
        use_container_width=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Parasitized (mean)", f"{summary['intensity_std']['Parasitized']['mean']:.2f}"
    )
    col2.metric(
        "Uninfected (mean)", f"{summary['intensity_std']['Uninfected']['mean']:.2f}"
    )
    col3.metric("Cohen's d", f"{separation('intensity_std'):.2f}", "large effect")

    st.markdown(
        "**What this tells us.** A healthy red blood cell is a smooth, "
        "uniform disc — its pixels cluster tightly around one brightness. "
        "Once a parasite invades, it introduces a chromatin body and disturbs "
        "the surrounding cytoplasm, so intensities scatter much more widely. "
        "This is the largest effect of any feature measured here, and it is "
        "why a convolutional network does well on this dataset: **texture "
        "heterogeneity, not colour, is the dominant cue.**"
    )

    # --- Feature 2 -------------------------------------------------------
    st.divider()
    st.subheader("2. Dark-pixel ratio — the parasite's physical footprint")

    figure = go.Figure()
    for label, color in CLASS_COLORS.items():
        figure.add_trace(
            go.Box(
                y=frame[frame.label == label]["dark_pixel_ratio"],
                name=label,
                marker_color=color,
                line_color=color,
                boxpoints="outliers",
                marker=dict(size=4, opacity=0.5),
            )
        )
    st.plotly_chart(
        style_axes(
            figure,
            "Fraction of cell markedly darker than its own median",
            "",
            "Dark-pixel ratio",
        ),
        use_container_width=True,
    )

    parasitized_mean = summary["dark_pixel_ratio"]["Parasitized"]["mean"]
    uninfected_mean = summary["dark_pixel_ratio"]["Uninfected"]["mean"]
    ratio = parasitized_mean / uninfected_mean if uninfected_mean else float("inf")

    col1, col2, col3 = st.columns(3)
    col1.metric("Parasitized (mean)", f"{parasitized_mean:.4%}")
    col2.metric("Uninfected (mean)", f"{uninfected_mean:.4%}")
    col3.metric("Ratio", f"{ratio:.0f}×", "parasitized vs uninfected")

    st.markdown(
        "**What this tells us.** This is the widest *relative* gap in the "
        f"dataset — parasitized cells carry roughly **{ratio:.0f}× more dark "
        "pixels** than uninfected ones. Those dark pixels are the parasite's "
        "chromatin, physically present inside the cell. Measuring darkness "
        "against each cell's *own* median rather than a fixed cutoff is what "
        "makes this hold up: slides vary in exposure and staining strength, "
        "and a global threshold would mistake a dim slide for a sick patient. "
        "Note the uninfected box sits almost flat against zero — healthy cells "
        "contain essentially nothing this dark."
    )

    # --- Feature 3 -------------------------------------------------------
    st.divider()
    st.subheader("3. Stain uptake — chemistry made visible")

    figure = go.Figure()
    for label, color in CLASS_COLORS.items():
        figure.add_trace(
            go.Histogram(
                x=frame[frame.label == label]["mean_saturation"],
                name=label,
                marker_color=color,
                opacity=0.6,
                nbinsx=45,
            )
        )
    figure.update_layout(barmode="overlay")
    st.plotly_chart(
        style_axes(
            figure,
            "Distribution of mean colour saturation",
            "Mean saturation",
            "Number of cells",
        ),
        use_container_width=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Parasitized (mean)",
        f"{summary['mean_saturation']['Parasitized']['mean']:.3f}",
    )
    col2.metric(
        "Uninfected (mean)",
        f"{summary['mean_saturation']['Uninfected']['mean']:.3f}",
    )
    col3.metric("Cohen's d", f"{separation('mean_saturation'):.2f}", "medium effect")

    st.markdown(
        "**What this tells us.** Giemsa stain binds to DNA. A mature red blood "
        "cell has ejected its nucleus and carries no DNA of its own, so it "
        "takes up little stain and stays a flat, desaturated pink. A parasite "
        "brings its own genome, the stain binds to it, and the cell turns "
        "visibly purple. The two distributions overlap far more than the "
        "previous two features, which is the honest caveat: **stain intensity "
        "alone would misclassify a lot of cells**, and it is only reliable in "
        "combination with texture and darkness."
    )

    # --- Combined --------------------------------------------------------
    st.divider()
    st.subheader("Why the model works: the features combined")

    figure = go.Figure()
    for label, color in CLASS_COLORS.items():
        subset = frame[frame.label == label]
        figure.add_trace(
            go.Scatter(
                x=subset["intensity_std"],
                y=subset["dark_pixel_ratio"],
                name=label,
                mode="markers",
                marker=dict(
                    size=6,
                    color=color,
                    opacity=0.55,
                    line=dict(width=1, color="rgba(255,255,255,0.55)"),
                ),
            )
        )
    st.plotly_chart(
        style_axes(
            figure,
            "Texture variation against dark-pixel ratio",
            "Std. dev. of pixel intensity",
            "Dark-pixel ratio",
        ),
        use_container_width=True,
    )

    st.markdown(
        "**The story.** Neither feature separates the classes cleanly on its "
        "own, but plotted together the two clouds pull apart: uninfected cells "
        "collapse into a tight low-variation, near-zero-darkness corner, while "
        "parasitized cells spread up and to the right. This is exactly the "
        "structure a CNN exploits — and it is also why the overlap region "
        "never disappears entirely. Those borderline cells are early-stage "
        "infections with very small parasites, and they are where the model's "
        "remaining errors live."
    )

    st.divider()
    st.subheader("Full feature summary")
    summary_rows = [
        {
            "Feature": FEATURE_LABELS.get(feature, feature),
            "Parasitized": round(values["Parasitized"]["mean"], 4),
            "Uninfected": round(values["Uninfected"]["mean"], 4),
            "Cohen's d": round(separation(feature), 2),
        }
        for feature, values in summary.items()
    ]
    frame_summary = pd.DataFrame(summary_rows).sort_values(
        "Cohen's d", key=lambda column: column.abs(), ascending=False
    )
    st.dataframe(frame_summary, use_container_width=True, hide_index=True)
    st.caption(
        "Cohen's d is the class mean gap in pooled standard deviations: "
        "0.2 small, 0.5 medium, 0.8 large. Cell area ratio is included as a "
        "deliberate negative result — the NIH pipeline crops every cell "
        "tightly, which normalises away the size variation it was meant to "
        "capture."
    )


def fetch_sample_bytes(label: str, filename: str) -> bytes | None:
    """Download one bundled example image from the API."""
    try:
        response = requests.get(
            f"{API_URL}/samples/{label}/{filename}", timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException:
        return None


def render_prediction(
    image_bytes: bytes,
    filename: str,
    true_label: str | None = None,
    require_click: bool = False,
) -> None:
    """Classify one image and render the verdict, metrics and probabilities.

    Extracted so the result can be rendered immediately beneath whichever
    input the user actually used. Rendering it once at the bottom of the page
    meant a click on an example scrolled the answer below the file uploader.
    """
    col_image, col_result = st.columns([1, 1.4])

    with col_image:
        st.image(image_bytes, caption=filename, width=260)
        if true_label:
            st.caption(f"True label: **{true_label}**")

    with col_result:
        if require_click and not st.button(
            "Classify",
            type="primary",
            use_container_width=True,
            icon=":material/biotech:",
        ):
            return

        with st.spinner("Running inference…"):
            result = api_post(
                "/predict", files={"file": (filename, image_bytes, "image/png")}
            )

        if result is None:
            return

        label = result["prediction"]
        confidence = result["confidence"]
        infected = label == "Parasitized"

        verdict_class = "verdict-positive" if infected else "verdict-negative"
        verdict_sub = (
            "Parasite detected in this cell" if infected else "No parasite detected"
        )

        st.markdown(
            f'<div class="verdict {verdict_class}">'
            f'<p class="verdict-title">{label}</p>'
            f'<p class="verdict-sub">{verdict_sub}</p>'
            f"</div>",
            unsafe_allow_html=True,
        )

        col_conf, col_lat = st.columns(2)
        col_conf.metric("Confidence", f"{confidence:.2%}")
        col_lat.metric("Latency", f"{result['latency_ms']:.0f} ms")

        probabilities = result["probabilities"]
        figure = go.Figure()
        for class_name, probability in probabilities.items():
            figure.add_trace(
                go.Bar(
                    x=[probability],
                    y=[class_name],
                    orientation="h",
                    name=class_name,
                    marker=dict(color=CLASS_COLORS[class_name], cornerradius=4),
                    text=[f"{probability:.1%}"],
                    textposition="outside",
                )
            )
        figure.update_xaxes(range=[0, 1.15], tickformat=".0%")
        st.plotly_chart(
            style_axes(figure, "Class probabilities", "Probability", ""),
            use_container_width=True,
        )

        if confidence < 0.75:
            st.warning(
                "Low confidence. This cell sits near the decision boundary, "
                "likely an early-stage infection with a small parasite or an "
                "artefact on the slide.",
                icon=":material/help:",
            )


def page_predict() -> None:
    st.title("Predict")
    st.caption("Classify a single segmented red blood cell image.")

    # Example cells served by the API. Without these, a visitor to the hosted
    # app has nothing to classify: the deployed container carries no dataset,
    # and telling someone to open a local folder is useless in a browser.
    payload = api_get("/samples", quiet=True)
    sample_list = (payload or {}).get("samples", [])

    if sample_list:
        st.markdown("#### Try an example")
        st.caption(
            "Real held-out cells the model never saw during training. "
            "The true label is shown, so you can check the answer."
        )

        columns = st.columns(len(sample_list))
        for column, entry in zip(columns, sample_list):
            with column:
                thumbnail = fetch_sample_bytes(entry["label"], entry["filename"])
                # Fixed-height box so every thumbnail occupies the same
                # vertical space and the Classify buttons line up. The cells
                # are cropped individually and differ slightly in aspect
                # ratio, which otherwise staggers the whole row.
                with st.container(height=150, border=False):
                    if thumbnail:
                        st.image(thumbnail, use_container_width=True)

                # The true label is deliberately not shown here. Revealing it
                # before classification gives the answer away; it appears with
                # the result instead, where it serves as verification.
                if st.button(
                    "Classify",
                    key=f"sample_{entry['label']}_{entry['filename']}",
                    use_container_width=True,
                    icon=":material/play_arrow:",
                ):
                    st.session_state["sample_pick"] = entry
                    st.session_state.pop("upload_pick", None)

        # Result renders here, directly under the examples.
        picked = st.session_state.get("sample_pick")
        if picked:
            st.divider()
            sample_bytes = fetch_sample_bytes(picked["label"], picked["filename"])
            if sample_bytes:
                render_prediction(
                    sample_bytes, picked["filename"], true_label=picked["label"]
                )

        st.divider()

    st.markdown("#### Or upload your own")
    uploaded = st.file_uploader(
        "Cell image", type=["png", "jpg", "jpeg"], accept_multiple_files=False
    )

    if uploaded is not None:
        st.session_state.pop("sample_pick", None)
        st.divider()
        render_prediction(uploaded.getvalue(), uploaded.name)
    elif not sample_list:
        st.info(
            "Upload a segmented red blood cell image to classify it.",
            icon=":material/upload_file:",
        )


def page_retrain() -> None:
    st.title("Upload data & retrain")
    st.caption(
        "Stage new labelled images, then retrain the deployed model on them. "
        "Retraining continues from the current model rather than starting "
        "from scratch, so it adapts to the new data without discarding what "
        "it already learned."
    )

    uploads = api_get("/uploads")
    if uploads is None:
        return

    st.subheader("Staged data")
    col1, col2, col3 = st.columns(3)
    col1.metric("Parasitized", uploads["staged"].get("Parasitized", 0))
    col2.metric("Uninfected", uploads["staged"].get("Uninfected", 0))
    col3.metric(
        "Total",
        uploads["total"],
        f"need {uploads['minimum_required']}",
    )

    st.divider()
    st.subheader("1. Upload new data")

    tab_zip, tab_images = st.tabs(["Bulk ZIP archive", "Individual images"])

    with tab_zip:
        st.markdown(
            "Upload a `.zip` containing `Parasitized/` and `Uninfected/` "
            "folders. Labels are read from the folder names."
        )
        archive = st.file_uploader("ZIP archive", type=["zip"], key="zip_upload")

        if archive is not None and st.button("Upload archive", type="primary"):
            with st.spinner("Uploading and unpacking…"):
                result = api_post(
                    "/upload",
                    files={
                        "file": (archive.name, archive.getvalue(), "application/zip")
                    },
                )
            if result:
                st.success(result["message"])
                st.json(result["saved"])
                st.rerun()

        with st.expander("Don't have a ZIP? Download a ready-made one"):
            st.markdown(
                "Builds a labelled archive from cells bundled with the API — "
                "real held-out images the model was never trained on. Download "
                "it, then upload it above to run the retraining flow."
            )

            per_class = st.number_input(
                "Images per class", min_value=10, max_value=25, value=25, step=5
            )

            # Served by the API rather than assembled from the local
            # filesystem. The deployed dashboard ships no dataset, so reading
            # data/test/ here would fail on the hosted app.
            archive_bytes = None
            try:
                response = requests.get(
                    f"{API_URL}/samples/archive",
                    params={"per_class": int(per_class)},
                    timeout=REQUEST_TIMEOUT,
                )
                if response.ok:
                    archive_bytes = response.content
                    image_count = response.headers.get("X-Image-Count", "?")
                else:
                    st.error(f"Could not build the archive — {response.status_code}")
            except requests.exceptions.RequestException as error:
                st.error(f"Could not reach the API — {error}")

            if archive_bytes:
                st.download_button(
                    f"Download retrain_batch.zip ({image_count} images)",
                    data=archive_bytes,
                    file_name="retrain_batch.zip",
                    mime="application/zip",
                    icon=":material/download:",
                    use_container_width=True,
                )

    with tab_images:
        label = st.selectbox("Label for these images", list(CLASS_COLORS))
        images = st.file_uploader(
            "Images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="image_upload",
        )

        if images and st.button("Upload images", type="primary"):
            progress = st.progress(0.0)
            uploaded_count = 0

            for index, image in enumerate(images, start=1):
                result = api_post(
                    "/upload",
                    files={"file": (image.name, image.getvalue(), "image/png")},
                    data={"label": label},
                )
                if result:
                    uploaded_count += 1
                progress.progress(index / len(images))

            st.success(f"Uploaded {uploaded_count} of {len(images)} image(s).")
            st.rerun()

    st.divider()
    st.subheader("2. Trigger retraining")

    col_epochs, col_lr = st.columns(2)
    epochs = col_epochs.slider("Epochs", 1, 20, 5)
    learning_rate = col_lr.select_slider(
        "Learning rate",
        options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        value=1e-4,
        format_func=lambda value: f"{value:.0e}",
    )
    include_base = st.checkbox(
        "Mix in the original training data",
        value=True,
        help=(
            "Recommended. Training only on a small new batch causes the model "
            "to forget the original distribution."
        ),
    )

    if not uploads["ready_to_retrain"]:
        st.warning(
            f"Upload at least {uploads['minimum_required']} images before "
            f"retraining. Currently staged: {uploads['total']}."
        )

    if st.button(
        "Start retraining",
        type="primary",
        icon=":material/model_training:",
        disabled=not uploads["ready_to_retrain"],
        use_container_width=True,
    ):
        result = api_post(
            "/retrain",
            json={
                "epochs": int(epochs),
                "learning_rate": float(learning_rate),
                "include_base_data": bool(include_base),
            },
        )
        if result:
            st.success(result["message"])

    st.divider()
    st.subheader("Retraining status")

    col_refresh, _ = st.columns([1, 4])
    if col_refresh.button("Refresh status"):
        st.rerun()

    job = api_get("/retrain/status")
    if job is None:
        return

    state = job["status"]
    if state == "running":
        st.markdown(status_pill("Retraining in progress", "live"),
                    unsafe_allow_html=True)
        st.write("")
        st.info(job["message"], icon=":material/hourglass_top:")
        st.caption(f"Started {job['started_at']}. Press **Refresh status** to update.")
    elif state == "completed":
        st.markdown(status_pill("Completed", "live"), unsafe_allow_html=True)
        st.write("")
        st.success(job["message"], icon=":material/task_alt:")
        if job.get("metrics"):
            columns = st.columns(len(job["metrics"]))
            for column, (key, value) in zip(columns, job["metrics"].items()):
                column.metric(key.replace("_", " ").title(), f"{value:.4f}")
    elif state == "failed":
        st.markdown(status_pill("Failed", "down"), unsafe_allow_html=True)
        st.write("")
        st.error(job["message"], icon=":material/error:")
    else:
        st.markdown(status_pill("Idle", "idle"), unsafe_allow_html=True)
        st.write("")
        st.caption("No retraining has been run in this session yet.")


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

PAGES = {
    "Overview": page_overview,
    "Data insights": page_insights,
    "Predict": page_predict,
    "Upload & retrain": page_retrain,
}

PAGE_CAPTIONS = {
    "Overview": "Uptime and live model metrics",
    "Data insights": "Feature analysis of the dataset",
    "Predict": "Classify a single cell image",
    "Upload & retrain": "Add data and retrain the model",
}


def main() -> None:
    with st.sidebar:
        st.markdown("#### Malaria Cell Classifier")
        st.caption("Diagnostic screening pipeline")
        st.divider()

        choice = st.radio(
            "Page",
            list(PAGES),
            captions=[PAGE_CAPTIONS[name] for name in PAGES],
            label_visibility="collapsed",
        )

        st.divider()

        health = api_get("/health", quiet=True)
        st.markdown(
            status_pill(
                "API reachable" if health else "API unreachable",
                "live" if health else "down",
            ),
            unsafe_allow_html=True,
        )
        st.caption(f"`{API_URL}`")

        st.divider()
        st.caption(
            "NIH dataset — 27,558 segmented red blood cell images, "
            "expert-annotated and class-balanced."
        )

    PAGES[choice]()


if __name__ == "__main__":
    main()
