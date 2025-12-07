import streamlit as st
import sys
import os
from pathlib import Path
import subprocess
import threading
import time

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from utils import initialize_workspace

initialize_workspace()

st.set_page_config(
    layout="wide",
    page_title="Model Training",
    page_icon="🎓",
    initial_sidebar_state="expanded"
)

# Load CSS
try:
    css_path = os.path.join(os.path.dirname(__file__), "..", "styles", "app.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

st.markdown(
    """
    <div class="hero-section">
        <h1>Training Orchestrator</h1>
        <p>Configure, launch, and monitor generative model training runs in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="margin-top: 0.75rem; margin-bottom: 0.75rem; display: flex; gap: 1.5rem; flex-wrap: wrap;">
        <div style="flex: 2; min-width: 260px; background: #ffffff; border-radius: 12px; padding: 1.0rem 1.25rem; border: 1px solid #e0e6ed; box-shadow: 0 4px 12px rgba(15,23,42,0.06);">
            <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; font-weight: 600; margin-bottom: 0.35rem;">Quick start</div>
            <ol style="margin: 0; padding-left: 1.2rem; font-size: 0.95rem; color: #0f172a; line-height: 1.7;">
                <li>Select a model family and training mode.</li>
                <li>Set core hyperparameters (batch size, epochs, LR).</li>
                <li>Point to your dataset and checkpoint destination.</li>
                <li>Copy the command and run it on your training node.</li>
            </ol>
        </div>
        <div style="flex: 1; min-width: 220px; background: #ffffff; border-radius: 12px; padding: 1.0rem 1.25rem; border: 1px solid #e0e6ed; box-shadow: 0 4px 12px rgba(15,23,42,0.06);">
            <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; font-weight: 600; margin-bottom: 0.35rem;">Model overview</div>
            <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.9rem; color: #0f172a; line-height: 1.6;">
                <li><strong>CGAN</strong> – adversarial image generator.</li>
                <li><strong>Cond. diffusion</strong> – high‑fidelity denoising model.</li>
                <li><strong>Prob. diffusion</strong> – diversity‑focused diffusion.</li>
                <li><strong>Pre‑built GAN</strong> – finetuning of a pretrained GAN.</li>
            </ul>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for training status
if 'training_active' not in st.session_state:
    st.session_state.training_active = False
if 'training_logs' not in st.session_state:
    st.session_state.training_logs = []

# # Training status indicator at the top
# status_col1, status_col2 = st.columns([3, 1])
# with status_col1:
#     st.markdown("### Training Configuration")
# with status_col2:
#     if st.session_state.training_active:
#         st.warning("Training in progress...")
#     else:
#         st.success("Ready to train")

st.markdown("---")

st.subheader("Configuration")

col_base1, col_base2 = st.columns(2)

with col_base1:
    model_type = st.selectbox(
        "Model family",
        ["Conditional GAN (CGAN)", "Conditional Diffusion Model", "Probability Diffusion Model", "Pre-built GAN"],
        help="Choose which generative model to train.",
    )

with col_base2:
    training_mode = st.radio(
        "Training mode",
        ["Train from scratch", "Fine-tune existing model"],
        help="Start fresh or continue from a checkpoint.",
        horizontal=True,
    )

col_hp1, col_hp2, col_hp3, col_hp4 = st.columns(4)

with col_hp1:
    batch_size = st.number_input(
        "Batch size",
        min_value=1,
        max_value=128,
        value=16,
        step=1,
        help="Number of samples per mini‑batch.",
    )

with col_hp2:
    epochs = st.number_input(
        "Epochs",
        min_value=1,
        max_value=1000,
        value=100,
        step=10,
        help="Total passes over the training set.",
    )

with col_hp3:
    learning_rate = st.number_input(
        "Learning rate",
        min_value=0.0,
        max_value=1.0,
        value=0.0002,
        step=0.0001,
        format="%.5f",
        help="Optimizer step size.",
    )

with col_hp4:
    image_size = st.selectbox(
        "Image size",
        [64, 128, 256],
        index=1,
        help="Input/output resolution.",
    )

st.markdown("---")

st.subheader("Model‑specific settings")

if model_type == "Conditional GAN (CGAN)":
    col1, col2 = st.columns(2)
    with col1:
        z_dim = st.number_input(
            "Latent Dimension (z_dim)",
            min_value=64,
            max_value=512,
            value=100,
            step=32,
            help="Dimension of the random noise vector"
        )
    with col2:
        d_lr = st.number_input(
            "Discriminator LR Multiplier",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Multiplier for discriminator learning rate"
        )

elif model_type == "Conditional Diffusion Model":
    col1, col2, col3 = st.columns(3)
    with col1:
        num_timesteps = st.number_input(
            "Number of Timesteps",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100,
            help="Diffusion process timesteps"
        )
    with col2:
        beta_start = st.number_input(
            "Beta Start",
            min_value=0.0,
            max_value=1.0,
            value=0.0001,
            step=0.0001,
            format="%.5f",
            help="Starting variance for diffusion"
        )
    with col3:
        beta_end = st.number_input(
            "Beta End",
            min_value=0.0,
            max_value=1.0,
            value=0.02,
            step=0.001,
            format="%.3f",
            help="Ending variance for diffusion"
        )

elif model_type == "Probability Diffusion Model":
    col1, col2, col3 = st.columns(3)
    with col1:
        num_timesteps = st.number_input(
            "Number of Timesteps",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100,
            help="Diffusion process timesteps"
        )
    with col2:
        beta_start = st.number_input(
            "Beta Start",
            min_value=0.0,
            max_value=1.0,
            value=0.0001,
            step=0.0001,
            format="%.5f",
            help="Starting variance for diffusion"
        )
    with col3:
        beta_end = st.number_input(
            "Beta End",
            min_value=0.0,
            max_value=1.0,
            value=0.02,
            step=0.001,
            format="%.3f",
            help="Ending variance for diffusion"
        )

elif model_type == "Pre-built GAN":
    col1, col2 = st.columns(2)
    with col1:
        z_dim = st.number_input(
            "Latent Dimension (z_dim)",
            min_value=64,
            max_value=512,
            value=128,
            step=32,
            help="Dimension of the random noise vector"
        )
    with col2:
        st.empty()  # Placeholder for alignment

st.markdown("---")

st.subheader("Data & output configuration")

col1, col2 = st.columns(2)

with col1:
    data_path = st.text_input(
        "Dataset path",
        value=str(project_root / "dataset"),
        help="Root folder containing training images/metadata.",
    )

with col2:
    output_dir = st.text_input(
        "Checkpoint directory",
        value=str(project_root / "checkpoints" / model_type.lower().replace(" ", "_").replace("(", "").replace(")", "")),
        help="Folder to write checkpoints, configs, and training samples.",
    )

if training_mode == "Fine-tune existing model":
    checkpoint_path = st.text_input(
        "Existing checkpoint (optional)",
        value="",
        help="Path to a .pt file to resume or fine‑tune from.",
        placeholder="e.g., checkpoints/cgan/best_model.pt",
    )
else:
    checkpoint_path = ""

st.markdown("---")

# Training Configuration Summary
st.subheader("Configuration summary")
summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.markdown(f"""
    **Model Configuration**
    - Model: {model_type}
    - Mode: {training_mode}
    - Image Size: {image_size}x{image_size}
    """)

with summary_col2:
    st.markdown(f"""
    **Training Settings**
    - Batch Size: {batch_size}
    - Epochs: {epochs}
    - Learning Rate: {learning_rate}
    """)

with summary_col3:
    if model_type == "Conditional GAN (CGAN)":
        st.markdown(f"""
        **Model Parameters**
        - Latent Dim: {z_dim}
        - D LR Multiplier: {d_lr}
        """)
    elif model_type == "Conditional Diffusion Model":
        st.markdown(f"""
        **Diffusion Parameters**
        - Timesteps: {num_timesteps}
        - Beta: [{beta_start}, {beta_end}]
        """)
    elif model_type == "Probability Diffusion Model":
        st.markdown(f"""
        **Diffusion Parameters**
        - Timesteps: {num_timesteps}
        - Beta: [{beta_start}, {beta_end}]
        """)
    elif model_type == "Pre-built GAN":
        st.markdown(f"""
        **Model Parameters**
        - Latent Dim: {z_dim}
        """)

st.markdown("---")

st.subheader("Training controls")

btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])

with btn_col1:
    start_button = st.button(
        "Start Training", 
        type="primary", 
        use_container_width=True, 
        disabled=st.session_state.training_active
    )

with btn_col2:
    stop_button = st.button(
        "Stop Training", 
        type="secondary", 
        use_container_width=True, 
        disabled=not st.session_state.training_active
    )

with btn_col3:
    if st.button("Reset Configuration", use_container_width=True):
        st.rerun()

if start_button:
    st.session_state.training_active = True
    st.session_state.training_logs = []
    # Build command based on model type
    if model_type == "Conditional GAN (CGAN)":
        script_path = project_root / "train" / "train_cgan.py"
        cmd = [
            "python", str(script_path),
            "--batch-size", str(batch_size),
            "--epochs", str(epochs),
            "--lr", str(learning_rate),
            "--image-size", str(image_size),
            "--z-dim", str(z_dim),
            "--output-dir", output_dir,
        ]
        if checkpoint_path:
            cmd.extend(["--checkpoint", checkpoint_path])
    elif model_type == "Conditional Diffusion Model":
        script_path = project_root / "train" / "train_diffusion.py"
        cmd = [
            "python", str(script_path),
            "--batch-size", str(batch_size),
            "--epochs", str(epochs),
            "--lr", str(learning_rate),
            "--image-size", str(image_size),
            "--num-timesteps", str(num_timesteps),
            "--output-dir", output_dir,
        ]
        if checkpoint_path:
            cmd.extend(["--checkpoint", checkpoint_path])
    elif model_type == "Probability Diffusion Model":
        script_path = project_root / "pb_diffusion" / "train_diffusion.py"
        cmd = [
            "python", str(script_path),
            "--batch-size", str(batch_size),
            "--epochs", str(epochs),
            "--lr", str(learning_rate),
            "--image-size", str(image_size),
            "--num-timesteps", str(num_timesteps),
            "--output-dir", output_dir,
        ]
        if checkpoint_path:
            cmd.extend(["--checkpoint", checkpoint_path])
    elif model_type == "Pre-built GAN":
        script_path = project_root / "pre-builtGAN_finetune" / "train_prebuilt_gan.py"
        cmd = [
            "python", str(script_path),
            "--batch-size", str(batch_size),
            "--epochs", str(epochs),
            "--lr", str(learning_rate),
            "--image-size", str(image_size),
            "--z-dim", str(z_dim),
            "--output-dir", output_dir,
        ]
        if checkpoint_path:
            cmd.extend(["--checkpoint", checkpoint_path])
    st.info(f"Starting training with command: `{' '.join(cmd)}`")
    st.warning("Note: Training will run in the background. Check the terminal for detailed logs.")
    st.code(' '.join(cmd), language='bash')
    st.info("Copy and run this command in your terminal to start training.")

if stop_button:
    st.session_state.training_active = False
    st.warning("Training stopped by user")

st.markdown("---")

st.subheader("Training progress & metrics")

# Placeholder for metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Current Epoch", "0/0", help="Current epoch / Total epochs")
with col2:
    st.metric("Training Loss", "N/A", help="Current training loss")
with col3:
    st.metric("Time Elapsed", "0:00:00", help="Time since training started")
with col4:
    st.metric("ETA", "N/A", help="Estimated time remaining")

st.markdown("---")

st.subheader("Training logs")

st.text_area(
    "Logs Output",
    value="\n".join(st.session_state.training_logs) if st.session_state.training_logs else "No logs yet. Start training to see logs.",
    height=300,
    disabled=True,
    key="training_logs_area"
)

st.markdown("---")

# Additional Information Sections
info_col1, info_col2 = st.columns(2)

with info_col1:
    with st.expander("Training Tips", expanded=False):
        st.markdown("""
        ### General Tips
        - Start with smaller batch sizes if you encounter memory issues
        - Use larger batch sizes for stable training on powerful hardware
        - Monitor the loss curves to detect overfitting
        - Save checkpoints frequently to resume training if interrupted
        
        ### Model-Specific Tips
        
        **Conditional GAN (CGAN)**:
        - Balance generator and discriminator training with the LR multiplier
        - Typical training: 100-200 epochs for good results
        - Watch for mode collapse (all images look similar)
        
        **Conditional Diffusion Model**:
        - Requires more training time than GANs
        - Increase inference steps for better quality at test time
        - More timesteps = better quality but slower training
        
        **Probability Diffusion Model**:
        - Probabilistic approach for improved sample diversity
        - Good for capturing complex data distributions
        - Similar training considerations to Conditional Diffusion
        
        **Pre-built GAN**:
        - Benefits from fine-tuning on domain-specific data
        - Generally converges faster than training from scratch
        - Good starting point for limited data scenarios
        """)

with info_col2:
    with st.expander("Checkpoint Management", expanded=False):
        st.markdown("""
        ### Checkpoint Locations
        - **CGAN**: `checkpoints/cgan/`
        - **Diffusion**: `checkpoints/conditional_diffusion/`
        - **Probability Diffusion**: `checkpoints/probability_diffusion/`
        - **Pre-built GAN**: `checkpoints/prebuilt_gan/`
        
        ### Files Saved
        - `best_model.pt` or `generator_final.pt`: Best performing model
        - `checkpoint_epoch_N.pt`: Periodic checkpoints
        - `training_config.json`: Training configuration
        - `samples/`: Generated sample images during training
        """)

with st.expander("Advanced Options", expanded=False):
    st.markdown("""
    ### Environment Variables
    Make sure to set `DATA_DIR` in your `.env` file:
    ```
    DATA_DIR=/path/to/your/dataset
    ```
    
    ### GPU/MPS Acceleration
    The training scripts automatically detect and use:
    - **Mac M1/M2/M3**: Metal Performance Shaders (MPS)
    - **NVIDIA GPU**: CUDA
    - **CPU**: Fallback to CPU training
    
    ### Monitoring Training
    For real-time monitoring, run the training command in your terminal:
    ```bash
    python train/train_cgan.py --batch-size 16 --epochs 100
    ```
    """)
