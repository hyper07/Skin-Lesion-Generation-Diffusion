import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from utils import initialize_workspace

initialize_workspace()

st.set_page_config(
    layout="wide",
    page_title="Dataset Analysis",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <div class="hero-section">
        <h1>Dataset Intelligence</h1>
        <p>Inspect class balance and coverage across HAM10000 and ISIC training data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="margin-top: 0.75rem; margin-bottom: 0.75rem; display: flex; gap: 1.5rem; flex-wrap: wrap;">
        <div style="flex: 2; min-width: 260px; background: #ffffff; border-radius: 12px; padding: 1.0rem 1.25rem; border: 1px solid #e0e6ed; box-shadow: 0 4px 12px rgba(15,23,42,0.06);">
            <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; font-weight: 600; margin-bottom: 0.35rem;">Why it matters</div>
            <p style="margin: 0; font-size: 0.95rem; color: #0f172a; line-height: 1.7;">
                Understand how real data is distributed before augmenting with synthetic cohorts.
                Spot minority classes, confirm coverage, and track shifts over time.
            </p>
        </div>
        <div style="flex: 1; min-width: 220px; background: #ffffff; border-radius: 12px; padding: 1.0rem 1.25rem; border: 1px solid #e0e6ed; box-shadow: 0 4px 12px rgba(15,23,42,0.06);">
            <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; font-weight: 600; margin-bottom: 0.35rem;">Quick view</div>
            <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.9rem; color: #0f172a; line-height: 1.6;">
                <li>Toggle between HAM10000, ISIC 2018, or combined.</li>
                <li>Review class counts and imbalance.</li>
                <li>Preview raw metadata for QA.</li>
            </ul>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container():
    left, right = st.columns([1.4, 1])

    with left:
        st.markdown("""<h2 style='margin-top: 1.0rem;'>Dataset selection</h2>""", unsafe_allow_html=True)
        dataset_name = st.selectbox(
            "Dataset",
            ["HAM10000", "ISIC 2018", "Combined Training Dataset"],
            index=0,
            help="Switch between individual datasets or the combined training pool.",
        )


st.markdown("---")

# Define paths
ham_metadata_path = project_root / "dataset" / "HAM10000_metadata.csv"
isic_metadata_path = project_root / "dataset" / "ISIC_metadata.csv"

# Define mappings
ham_dx_map = {
    'nv': 'Nevus',
    'mel': 'Melanoma (HAM)',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratosis',
    'vasc': 'Vascular lesion',
    'df': 'Dermatofibroma',
    'bkl': 'Benign keratosis (HAM)'
}

bcn_diagnosis_map = {
    'Nevus': 'Nevus',
    'Melanoma, NOS': 'Melanoma (BCN)',
    'Melanoma metastasis': 'Melanoma metastasis',
    'Basal cell carcinoma': 'Basal cell carcinoma',
    'Seborrheic keratosis': 'Seborrheic keratosis',
    'Solar or actinic keratosis': 'Actinic keratosis',
    'Squamous cell carcinoma, NOS': 'Squamous cell carcinoma',
    'Scar': 'Scar',
    'Solar lentigo': 'Solar lentigo',
    'Dermatofibroma': 'Dermatofibroma'
}

# Load data functions
@st.cache_data
def load_ham_data():
    if not ham_metadata_path.exists():
        return None
    try:
        df = pd.read_csv(ham_metadata_path)
        # Map diagnosis
        if 'dx' in df.columns:
            df['diagnosis_class'] = df['dx'].map(ham_dx_map)
        return df
    except Exception as e:
        st.error(f"Error loading HAM10000 metadata: {e}")
        return None

@st.cache_data
def load_isic_data():
    if not isic_metadata_path.exists():
        return None
    try:
        df = pd.read_csv(isic_metadata_path)
        # Map diagnosis
        if 'diagnosis_3' in df.columns:
            df['diagnosis_class'] = df['diagnosis_3'].map(lambda x: bcn_diagnosis_map.get(x, x))
        return df
    except Exception as e:
        st.error(f"Error loading ISIC metadata: {e}")
        return None

# Load datasets
ham_df = load_ham_data()
isic_df = load_isic_data()

# Dataset statistics
st.subheader("Dataset overview")

col1, col2, col3 = st.columns(3)

if dataset_name == "HAM10000":
    if ham_df is not None:
        with col1:
            st.metric("Total Images", f"{len(ham_df):,}")
        with col2:
            st.metric("Classes", f"{ham_df['diagnosis_class'].nunique()}")
        with col3:
            st.metric("Image Size", "600x450")
            
        st.markdown("### Class Distribution")
        
        # Calculate distribution
        dist_df = ham_df['diagnosis_class'].value_counts().reset_index()
        dist_df.columns = ['Class', 'Count']
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=dist_df, x='Count', y='Class', ax=ax, palette='viridis')
        ax.set_title("HAM10000 Class Distribution")
        
        # Add count labels
        for i, v in enumerate(dist_df['Count']):
            ax.text(v + 5, i, str(int(v)), va='center')
            
        st.pyplot(fig)
        
        st.markdown("### Raw Data Preview")
        st.dataframe(ham_df.head())
    else:
        st.error(f"HAM10000 metadata not found at {ham_metadata_path}")

elif dataset_name == "ISIC 2018":
    if isic_df is not None:
        with col1:
            st.metric("Total Images", f"{len(isic_df):,}")
        with col2:
            st.metric("Classes", f"{isic_df['diagnosis_class'].nunique()}")
        with col3:
            st.metric("Image Size", "Various")
            
        st.markdown("### Class Distribution")
        
        # Calculate distribution
        dist_df = isic_df['diagnosis_class'].value_counts().reset_index()
        dist_df.columns = ['Class', 'Count']
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=dist_df, x='Count', y='Class', ax=ax, palette='viridis')
        ax.set_title("ISIC 2018 (BCN20000) Class Distribution")
        
        # Add count labels
        for i, v in enumerate(dist_df['Count']):
            ax.text(v + 5, i, str(int(v)), va='center')
            
        st.pyplot(fig)
        
        st.markdown("### Raw Data Preview")
        st.dataframe(isic_df.head())
    else:
        st.error(f"ISIC metadata not found at {isic_metadata_path}")

elif dataset_name == "Combined Training Dataset":
    if ham_df is not None and isic_df is not None:
        # Combine counts
        ham_counts = ham_df['diagnosis_class'].value_counts()
        isic_counts = isic_df['diagnosis_class'].value_counts()
        
        combined_counts = ham_counts.add(isic_counts, fill_value=0).sort_values(ascending=False)
        total_images = len(ham_df) + len(isic_df)
        
        with col1:
            st.metric("Total Images", f"{total_images:,}")
        with col2:
            st.metric("Classes", f"{len(combined_counts)}")
        with col3:
            st.metric("Image Size", "Various")
            
        st.markdown("### Class Distribution")
        st.info("This dataset combines HAM10000 and BCN20000 datasets used for training the models.")
        
        dist_df = combined_counts.reset_index()
        dist_df.columns = ['Class', 'Count']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=dist_df, x='Count', y='Class', ax=ax, palette='viridis')
        ax.set_title("Combined Dataset Class Distribution")
        
        # Add count labels
        for i, v in enumerate(dist_df['Count']):
            ax.text(v + 50, i, str(int(v)), va='center')
            
        st.pyplot(fig)
        
        st.markdown("### Detailed Counts")
        st.dataframe(dist_df)
    else:
        st.error("One or both metadata files are missing.")

