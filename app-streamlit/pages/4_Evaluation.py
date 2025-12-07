import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from utils import initialize_workspace

initialize_workspace()

st.set_page_config(
    layout="wide",
    page_title="Model Evaluation",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <div class=\"hero-section\">
        <h1>Evaluation Studio</h1>
        <p>Benchmark generative models on fidelity, diversity, and downstream utility.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style=\"margin-top: 0.75rem; margin-bottom: 0.75rem; display: flex; gap: 1.5rem; flex-wrap: wrap;\">
        <div style=\"flex: 2; min-width: 260px; background: #ffffff; border-radius: 12px; padding: 1.0rem 1.25rem; border: 1px solid #e0e6ed; box-shadow: 0 4px 12px rgba(15,23,42,0.06);\">
            <div style=\"font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; font-weight: 600; margin-bottom: 0.35rem;\">Quick start</div>
            <ol style=\"margin: 0; padding-left: 1.2rem; font-size: 0.95rem; color: #0f172a; line-height: 1.7;\">
                <li>Review F1, FID, Inception Score, and LPIPS for all models.</li>
                <li>Use the charts to see trade‑offs by metric.</li>
            </ol>
        </div>
        <div style=\"flex: 1; min-width: 220px; background: #ffffff; border-radius: 12px; padding: 1.0rem 1.25rem; border: 1px solid #e0e6ed; box-shadow: 0 4px 12px rgba(15,23,42,0.06);\">
            <div style=\"font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; font-weight: 600; margin-bottom: 0.35rem;\">Metrics</div>
            <ul style=\"margin: 0; padding-left: 1.2rem; font-size: 0.9rem; color: #0f172a; line-height: 1.6;\">
                <li><strong>F1</strong> – classifier performance on real vs. synthetic.</li>
                <li><strong>FID</strong> – distribution distance to real images.</li>
                <li><strong>IS</strong> – image quality & diversity proxy.</li>
                <li><strong>LPIPS</strong> – perceptual similarity distance.</li>
            </ul>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Load evaluation data from log_eval folder (classification overall_metrics.json only)
log_root = Path.cwd().parent / "log_eval"

model_runs = {
    "cgan": log_root / "cgan" / "20251125_192908",
    "conditional_diffusion": log_root / "conditional_diffusion" / "20251125_211432",
    "prebuilt_diffusion": log_root / "prebuilt_diffusion" / "20251125_193016",
    "prebuilt_gan": log_root / "prebuilt_gan" / "20251125_192951",
}

model_mapping = {
    "cgan": "Conditional GAN",
    "conditional_diffusion": "Conditional Diffusion",
    "prebuilt_diffusion": "Probability Diffusion",
    "prebuilt_gan": "Pre-built GAN",
}

records = []

for key, run_dir in model_runs.items():
    summary_path = run_dir / "image_quality" / "summary.json"

    try:
        if summary_path.exists():
            summary = pd.read_json(summary_path, typ="series")

            records.append(
                {
                    "model_key": key,
                    "Model": model_mapping.get(key, key),
                    "F1 Score (%)": float(summary.get("classification", {}).get("f1", float("nan"))),
                    "FID Score (↓)": float(summary.get("fid", float("nan"))),
                    "Inception Score (↑)": float(summary.get("is", {}).get("mean", float("nan")) if isinstance(summary.get("is"), dict) else summary.get("is_mean", float("nan"))),
                    "LPIPS (↓)": float(summary.get("lpips", float("nan"))),
                }
            )
    except Exception as e:
        st.warning(f"Failed to load metrics for {key}: {e}")

if records:
    df_real = pd.DataFrame.from_records(records)
    df_metrics = df_real[["Model", "F1 Score (%)", "FID Score (↓)", "Inception Score (↑)", "LPIPS (↓)"]]
else:
    st.error("No evaluation runs found in log_eval. Please run the evaluation script first.")
    df_metrics = pd.DataFrame()

st.subheader("Quantitative metrics")

st.dataframe(df_metrics.style.format({
    "F1 Score (%)": "{:.2f}",
    "FID Score (↓)": "{:.4f}",
    "Inception Score (↑)": "{:.4f}",
    "LPIPS (↓)": "{:.4f}"
}), use_container_width=True)

def _styled_bar(metric_col, ylabel, palette=None):
    if df_metrics.empty:
        return None
    colors = palette or ['#0f172a', '#0369a1', '#16a34a', '#ea580c']
    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    ax.bar(df_metrics["Model"], df_metrics[metric_col], color=colors[: len(df_metrics)])
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(df_metrics["Model"], rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    return fig

col1, col2 = st.columns(2)

with col1:
    st.markdown("### F1 score")
    fig = _styled_bar("F1 Score (%)", "F1 Score (%)")
    if fig is not None:
        st.pyplot(fig)

with col2:
    st.markdown("### FID score")
    fig = _styled_bar("FID Score (↓)", "FID score (lower is better)")
    if fig is not None:
        st.pyplot(fig)

col3, col4 = st.columns(2)

with col3:
    st.markdown("### Inception score")
    fig = _styled_bar("Inception Score (↑)", "Inception score (higher is better)")
    if fig is not None:
        st.pyplot(fig)

with col4:
    st.markdown("### LPIPS")
    fig = _styled_bar("LPIPS (↓)", "LPIPS (lower is better)")
    if fig is not None:
        st.pyplot(fig)

st.markdown("---")

st.subheader("Visual assessment")
st.info("Side‑by‑side image comparison between models is planned for a future release.")
