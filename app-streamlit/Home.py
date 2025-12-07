import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import base64

from utils import initialize_workspace

# Initialize workspace path and imports
initialize_workspace()

st.set_page_config(
    layout="wide",
    page_title="Diffusion Model Training Platform",
    page_icon="🎨",
    initial_sidebar_state="expanded"
)

# Header & footer are rendered by the router (streamlit_app.py) when using st.navigation
# CSS is loaded globally from styles/app.css in the router
# Fallback: load CSS here if page is run directly
try:
    css_path = os.path.join(os.path.dirname(__file__), "styles", "app.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass


# CSS is loaded from styles/app.css

def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


sps_logo = ""
sg_logo = ""



# Platform metrics for diffusion models
total_training_images = 35346
training_epochs_completed = 100
active_models = 4

# Build metrics list for diffusion platform
metrics_data = [
    {
        "label": "Training Images",
        "value": (f"{total_training_images:,}" if total_training_images not in (None, 0) else "0"),
        "delta": "",
        "context": "HAM10000 + ISIC 2018",
    },
    {
        "label": "Training Epochs",
        "value": (f"{training_epochs_completed:,}" if training_epochs_completed not in (None, 0) else "0"),
        "delta": "",
        "context": "completed",
    },
    {
        "label": "Active Models",
        "value": (f"{active_models:,}" if active_models not in (None, 0) else "0"),
        "delta": "",
        "context": "CGAN, Conditional Diffusion, LoRA",
    },
]

# persist to session for smoother navigation
st.session_state['home_metrics'] = {'overview': metrics_data}


session_metrics = st.session_state.get("home_metrics") if "home_metrics" in st.session_state else None
if isinstance(session_metrics, list) and session_metrics:
    metrics_data = session_metrics
elif isinstance(session_metrics, dict):
    overview_metrics = session_metrics.get("overview")
    if isinstance(overview_metrics, list) and overview_metrics:
        metrics_data = overview_metrics

metric_cards = []
for metric in metrics_data:
    label = str(metric.get("label", "")).strip()
    value = str(metric.get("value", "")).strip()
    delta_text = str(metric.get("delta", "")).strip()
    context_text = str(metric.get("context", "")).strip()
    
    trend_html = ""
    if delta_text:
        trend_class = "trend-up" if delta_text.startswith("+") else "trend-down"
        trend_icon = "↑" if delta_text.startswith("+") else "↓"
        trend_html = f'<span class="{trend_class}">{trend_icon} {delta_text}</span>'
    
    metric_cards.append(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        # f'<div class="metric-trend">{trend_html} <span style="opacity: 0.7">{context_text}</span></div>'
        f'</div>'
    )

hero_html = f"""
<div class="hero-container">
    <div class="hero-bg-pattern"></div>
    <div class="hero-content">
        <div>
            <h1 class="hero-title">Medical Image Synthesis Platform</h1>
            <p class="hero-description">
                State-of-the-art diffusion models for generating high-quality synthetic medical images.
                Train conditional models, fine-tune with LoRA, and generate diverse skin lesion images for research and augmentation.
            </p>
        </div>
        <div class="metrics-grid" style="display: flex; flex-direction: column; gap: 1rem; align-items: stretch;">
            {''.join(metric_cards)}
        </div>
    </div>
</div>
"""

st.markdown(hero_html, unsafe_allow_html=True)

navigation_cards = [
    {
        "title": "Model Training",
        "description": "Train conditional diffusion models, CGANs, and fine-tune pre-built models with LoRA. Monitor training progress, loss curves, and checkpoint management.",
        "button": "Open Training Dashboard",
        "page": "pages/1_Training.py",
        "key": "nav_training_btn",
    },
    {
        "title": "Image Generation",
        "description": "Generate synthetic medical images with trained models. Control lesion types, apply conditional guidance, and export high-quality samples for dataset augmentation.",
        "button": "Launch Generator",
        "page": "pages/2_Generation.py",
        "key": "nav_generation_btn",
    },
    {
        "title": "Dataset Analysis",
        "description": "Explore HAM10000 and ISIC datasets with interactive visualizations. Analyze class distributions, image statistics, and metadata insights.",
        "button": "View Dataset Explorer",
        "page": "pages/3_Dataset_Analysis.py",
        "key": "nav_dataset_btn",
    },
    {
        "title": "Model Evaluation",
        "description": "Evaluate model performance with FID scores, inception scores, and visual quality metrics. Compare generated samples against real images.",
        "button": "Open Evaluation Suite",
        "page": "pages/4_Evaluation.py",
        "key": "nav_eval_btn",
    },
]

nav_columns = st.columns(len(navigation_cards))
for column, card in zip(nav_columns, navigation_cards):
    with column:
        st.markdown(
            f"""
            <div class="nav-card">
                <h4 class="nav-title">{card['title']}</h4>
                <p class="nav-desc">{card['description']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Button outside the card div to work with Streamlit's native button
        if st.button(card["button"], key=card["key"]):
            st.switch_page(card["page"])

