import streamlit as st
import sys
import os
from pathlib import Path
import torch
from PIL import Image
import io
import json

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from utils import initialize_workspace

initialize_workspace()

st.set_page_config(
    layout="wide",
    page_title="Image Generation",
    page_icon="🖼️",
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
        <h1>AI Skin Lesion Studio</h1>
        <p>Generate high‑quality synthetic dermatology images in a few clicks.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="margin-top: 0.75rem; margin-bottom: 0.5rem; display: flex; gap: 1.5rem; flex-wrap: wrap;">
        <div style="flex: 2; min-width: 260px; background: #ffffff; border-radius: 12px; padding: 1.0rem 1.25rem; border: 1px solid #e0e6ed; box-shadow: 0 4px 12px rgba(15,23,42,0.06);">
            <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; font-weight: 600; margin-bottom: 0.35rem;">Quick start</div>
            <ol style="margin: 0; padding-left: 1.2rem; font-size: 0.95rem; color: #0f172a; line-height: 1.7;">
                <li>Select a model and disease.</li>
                <li>Choose how many images to generate.</li>
                <li>Optionally adjust quality / speed parameters.</li>
                <li>Click <strong>Generate images</strong>.</li>
                <li>Review and download the generated samples.</li>
            </ol>
        </div>
        <div style="flex: 1; min-width: 220px; background: #ffffff; border-radius: 12px; padding: 1.0rem 1.25rem; border: 1px solid #e0e6ed; box-shadow: 0 4px 12px rgba(15,23,42,0.06);">
            <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; font-weight: 600; margin-bottom: 0.35rem;">Model overview</div>
            <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.9rem; color: #0f172a; line-height: 1.6;">
                <li><strong>CGAN</strong> – fast grids for quick augmentation.</li>
                <li><strong>Diffusion</strong> – slower, higher‑fidelity samples.</li>
                <li><strong>Prob. diffusion</strong> – diverse individual samples.</li>
                <li><strong>Pre‑built GAN</strong> – pretrained baseline generator.</li>
            </ul>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Lightweight help strip
help_tab, classes_tab, tips_tab, compare_tab = st.tabs([
    "About models",
    "Disease classes",
    "Tips",
    "Compare",
])

with help_tab:
    st.markdown(
        """
        <div style="background:#ffffff;border-radius:12px;padding:0.85rem 1rem;border:1px solid #e2e8f0;font-size:0.9rem;">
            <strong>Models at a glance</strong><br/>
            • <strong>CGAN</strong> – fast grids for quick augmentation.<br/>
            • <strong>Conditional diffusion</strong> – higher fidelity, slower per image.<br/>
            • <strong>Probability diffusion</strong> – diverse single images per class.<br/>
            • <strong>Pre‑built GAN</strong> – pretrained baseline on skin lesions.
        </div>
        """,
        unsafe_allow_html=True,
    )

with classes_tab:
    st.markdown(
        """
        <div style="background:#ffffff;border-radius:12px;padding:0.85rem 1rem;border:1px solid #e2e8f0;font-size:0.85rem;max-height:210px;overflow-y:auto;">
            <strong>Supported lesion classes (13)</strong>
            <ul style="margin:0.35rem 0 0 1.1rem; padding:0;">
                <li>Actinic keratosis</li>
                <li>Basal cell carcinoma</li>
                <li>Benign keratosis (HAM)</li>
                <li>Dermatofibroma</li>
                <li>Melanoma (BCN)</li>
                <li>Melanoma (HAM)</li>
                <li>Melanoma metastasis</li>
                <li>Nevus</li>
                <li>Scar</li>
                <li>Seborrheic keratosis</li>
                <li>Solar lentigo</li>
                <li>Squamous cell carcinoma</li>
                <li>Vascular lesion</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tips_tab:
    st.markdown(
        """
        <div style="background:#ffffff;border-radius:12px;padding:0.85rem 1rem;border:1px solid #e2e8f0;font-size:0.9rem;">
            <strong>Generation tips</strong>
            <ul style="margin:0.35rem 0 0 1.1rem; padding:0;">
                <li>Start with 30–50 diffusion steps; increase only if needed.</li>
                <li>Lower batch size if you see memory issues.</li>
                <li>Keep default checkpoints for production‑like behaviour.</li>
                <li>Use higher latent dimension only when you need more diversity.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with compare_tab:
    st.markdown(
        """
        <div style="background:#ffffff;border-radius:12px;padding:0.85rem 1rem;border:1px solid #e2e8f0;font-size:0.85rem;">
            <table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
                <thead>
                    <tr style="border-bottom:1px solid #e5e7eb;">
                        <th style="text-align:left;padding:0.25rem 0.35rem;">Feature</th>
                        <th style="text-align:left;padding:0.25rem 0.35rem;">CGAN</th>
                        <th style="text-align:left;padding:0.25rem 0.35rem;">Cond. diff.</th>
                        <th style="text-align:left;padding:0.25rem 0.35rem;">Prob. diff.</th>
                        <th style="text-align:left;padding:0.25rem 0.35rem;">Pre‑built GAN</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="padding:0.25rem 0.35rem;">Speed</td>
                        <td style="padding:0.25rem 0.35rem;">Fast</td>
                        <td style="padding:0.25rem 0.35rem;">Slow</td>
                        <td style="padding:0.25rem 0.35rem;">Slow</td>
                        <td style="padding:0.25rem 0.35rem;">Fast</td>
                    </tr>
                    <tr style="background:#f9fafb;">
                        <td style="padding:0.25rem 0.35rem;">Quality</td>
                        <td style="padding:0.25rem 0.35rem;">Good</td>
                        <td style="padding:0.25rem 0.35rem;">Excellent</td>
                        <td style="padding:0.25rem 0.35rem;">Excellent</td>
                        <td style="padding:0.25rem 0.35rem;">Good</td>
                    </tr>
                    <tr>
                        <td style="padding:0.25rem 0.35rem;">Diversity</td>
                        <td style="padding:0.25rem 0.35rem;">Medium</td>
                        <td style="padding:0.25rem 0.35rem;">High</td>
                        <td style="padding:0.25rem 0.35rem;">Very high</td>
                        <td style="padding:0.25rem 0.35rem;">Medium</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Initialize session state for generated images
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []

# Disease classes
DISEASE_CLASSES = {
    "Actinic keratosis": 0,
    "Basal cell carcinoma": 1,
    "Benign keratosis (HAM)": 2,
    "Dermatofibroma": 3,
    "Melanoma (BCN)": 4,
    "Melanoma (HAM)": 5,
    "Melanoma metastasis": 6,
    "Nevus": 7,
    "Scar": 8,
    "Seborrheic keratosis": 9,
    "Solar lentigo": 10,
    "Squamous cell carcinoma": 11,
    "Vascular lesion": 12
}

st.markdown("---")

with st.container():
    st.markdown("""<h2 style='margin-top: 1.5rem;'>Configuration</h2>""", unsafe_allow_html=True)

    left, right = st.columns([1.3, 1])

    with left:
        st.markdown("""
        <div style="background: #ffffff; border-radius: 16px; padding: 1.25rem 1.5rem; border: 1px solid #e2e8f0; box-shadow: 0 8px 18px rgba(15,23,42,0.04); margin-bottom: 1rem;">
            <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; font-weight: 700; margin-bottom: 0.25rem;">Step 1</div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="font-size: 1.1rem; font-weight: 600; color: #0f172a;">Model & disease</div>
                <span style="font-size: 0.8rem; color: #64748b;">Required</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            model_type = st.selectbox(
                "Model",
                ["Conditional GAN (CGAN)", "Conditional Diffusion Model", "Probability Diffusion Model", "Pre-built GAN"],
                help="Choose which generative model to use for image synthesis",
            )

        with col2:
            selected_disease = st.selectbox(
                "Disease class",
                list(DISEASE_CLASSES.keys()),
                help="Type of skin lesion to generate",
            )

        st.markdown("""
        <div style="background: #ffffff; border-radius: 16px; padding: 1.25rem 1.5rem; border: 1px solid #e2e8f0; box-shadow: 0 8px 18px rgba(15,23,42,0.04); margin-top: 0.5rem;">
            <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; font-weight: 700; margin-bottom: 0.25rem;">Step 2</div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                <div style="font-size: 1.1rem; font-weight: 600; color: #0f172a;">Sampling</div>
                <span style="font-size: 0.8rem; color: #64748b;">Performance vs. quality</span>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            num_samples = st.slider(
                "Number of images",
                min_value=1,
                max_value=16,
                value=4,
                help="Total synthetic images to create",
            )

        with col2:
            batch_size = st.slider(
                "Batch size",
                min_value=1,
                max_value=16,
                value=4,
                help="Images per generation batch (memory vs. speed)",
            )

        col1, col2 = st.columns(2)

        with col1:
            if model_type == "Conditional Diffusion Model" or model_type == "Probability Diffusion Model":
                num_inference_steps = st.slider(
                    "Inference steps",
                    min_value=10,
                    max_value=100,
                    value=50,
                    step=10,
                    help="More steps → higher fidelity, slower generation",
                )
            elif model_type == "Pre-built GAN":
                img_size = st.selectbox(
                    "Image size",
                    [64, 128, 256],
                    index=1,
                    help="Spatial resolution of generated images",
                )
            else:
                st.caption("GAN backends generate without configurable diffusion steps.")

        with col2:
            if model_type == "Pre-built GAN":
                z_dim = st.slider(
                    "Latent dimension",
                    min_value=64,
                    max_value=512,
                    value=128,
                    step=64,
                    help="Controls diversity of generated samples",
                )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div style="background: #0f172a; border-radius: 16px; padding: 1.25rem 1.5rem; color: #e5e7eb; box-shadow: 0 10px 25px rgba(15,23,42,0.45);">
            <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; color: #38bdf8; font-weight: 700; margin-bottom: 0.25rem;">Session snapshot</div>
            <div style="font-size: 1.0rem; font-weight: 600; margin-bottom: 0.75rem;">Current configuration</div>
        """, unsafe_allow_html=True)

        st.markdown(
            f"""
            <ul style='list-style: none; padding-left: 0; margin: 0 0 0.75rem 0; font-size: 0.9rem;'>
                <li><span style='color:#94a3b8;'>Model</span>: <strong>{model_type}</strong></li>
                <li><span style='color:#94a3b8;'>Disease</span>: <strong>{selected_disease}</strong></li>
                <li><span style='color:#94a3b8;'>Images</span>: <strong>{num_samples}</strong></li>
                <li><span style='color:#94a3b8;'>Batch size</span>: <strong>{batch_size}</strong></li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

        if model_type == "Conditional Diffusion Model" or model_type == "Probability Diffusion Model":
            st.markdown(
                f"""
                <div style='font-size: 0.85rem; color: #cbd5f5; margin-bottom: 0.5rem;'>
                    Diffusion steps: <strong>{num_inference_steps}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif model_type == "Pre-built GAN":
            st.markdown(
                f"""
                <div style='font-size: 0.85rem; color: #cbd5f5; margin-bottom: 0.5rem;'>
                    Image size: <strong>{img_size}px</strong><br/>
                    Latent dim: <strong>{z_dim}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

        checkpoint_path = st.text_input(
            "Checkpoint override (optional)",
            value="",
            placeholder="Leave blank to use the default production checkpoint",
            help="Point to a specific .pt file for experiments.",
        )

        st.caption("Tip: keep defaults for production‑ready presets unless you know you need to override them.")

st.markdown("---")

# Configuration Summary
st.subheader("Generation Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.markdown(f"""
    **Model Configuration**
    - Model: {model_type}
    - Disease: {selected_disease}
    """)

with summary_col2:
    st.markdown(f"""
    **Generation Settings**
    - Number of Images: {num_samples}
    - Batch Size: {batch_size}
    """)

with summary_col3:
    if model_type == "Conditional Diffusion Model" or model_type == "Probability Diffusion Model":
        st.markdown(f"""
        **Diffusion Parameters**
        - Inference Steps: {num_inference_steps}
        """)
    elif model_type == "Pre-built GAN":
        st.markdown(f"""
        **GAN Parameters**
        - Image Size: {img_size}
        - Latent Dim: {z_dim}
        """)
    else:
        st.markdown(f"""
        **Model Type**
        - {model_type}
        """)

st.markdown("---")

# Generation Controls
st.subheader("Generate Images")

btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])

with btn_col1:
    generate_button = st.button("Generate Images", type="primary", use_container_width=True)

with btn_col2:
    if st.button("Clear Results", use_container_width=True):
        st.session_state.generated_images = []
        st.rerun()

with btn_col3:
    if st.button("Batch Generate", use_container_width=True, disabled=True):
        st.info("Batch generation coming soon!")

st.markdown("---")


if generate_button:
    with st.spinner(f"Generating images using {model_type}..."):
        try:
            import subprocess
            import tempfile
            import glob
            from datetime import datetime
            
            # Create temporary output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_output_dir = str(project_root / "temp" / f"generation_{timestamp}")
            
            # Ensure temp directory exists
            Path(temp_output_dir).mkdir(parents=True, exist_ok=True)
            
            # Prepare command based on model type
            if model_type == "Conditional GAN (CGAN)":
                script_path = str(project_root / "generate" / "generate_cgan.py")
                cmd = [
                    sys.executable, script_path,
                    "--checkpoint", checkpoint_path or str(project_root / "checkpoints" / "cgan" / "cgan128_final.pt"),
                    "--num_samples", str(num_samples),
                    "--class_id", str(DISEASE_CLASSES[selected_disease]),
                    "--output", f"{temp_output_dir}/cgan_generated.png"
                ]
                
            elif model_type == "Conditional Diffusion Model":
                script_path = str(project_root / "generate" / "generate_diffusion.py")
                cmd = [
                    sys.executable, script_path,
                    "--checkpoint", checkpoint_path or str(project_root / "checkpoints" / "conditional_diffusion" / "diffusion_final.pt"),
                    "--class_id", str(DISEASE_CLASSES[selected_disease]),
                    "--num_samples", str(num_samples),
                    "--num_inference_steps", str(num_inference_steps),
                    "--output", f"{temp_output_dir}/diffusion_generated.png"
                ]
                
            elif model_type == "Probability Diffusion Model":
                script_path = str(project_root / "generate" / "generate_prebuilt_diffusion.py")
                cmd = [
                    sys.executable, script_path,
                    "--checkpoint", checkpoint_path or str(project_root / "checkpoints" / "prebuilt_diffusion" / "prebuilt_diffusion_epoch_88_final.pt"),
                    "--class_id", str(DISEASE_CLASSES[selected_disease]),
                    "--num_samples", str(num_samples),
                    "--num_classes", str(len(DISEASE_CLASSES)),
                    "--output", temp_output_dir
                ]
                
            elif model_type == "Pre-built GAN":
                script_path = str(project_root / "generate" / "generate_prebuilt_gan.py")
                cmd = [
                    sys.executable, script_path,
                    "--checkpoint", checkpoint_path or str(project_root / "checkpoints" / "prebuilt_gan" / "G_final.pt"),
                    "--num_classes", str(len(DISEASE_CLASSES)),
                    "--samples_per_class", str(num_samples),
                    "--class_id", str(DISEASE_CLASSES[selected_disease]),
                    "--img_size", str(img_size),
                    "--z_dim", str(z_dim),
                    "--output", f"{temp_output_dir}/prebuilt_gan_generated.png"
                ]
            
            # Run the generation script
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
            
            if result.returncode == 0:
                # Generation Section
                st.subheader("Generated Images")

                st.success(f"✅ Generated {num_samples} images of {selected_disease}")
                
                # Find generated images
                if model_type == "Probability Diffusion Model":
                    # Probability diffusion creates individual files in subdirectories
                    image_pattern = f"{temp_output_dir}/**/class_{DISEASE_CLASSES[selected_disease]}/*.png"
                    image_files = glob.glob(image_pattern, recursive=True)
                else:
                    # Other models create a single grid image
                    image_pattern = f"{temp_output_dir}/*.png"
                    image_files = glob.glob(image_pattern)

                # Debug: List all files in temp directory
                if Path(temp_output_dir).exists():
                    all_files = list(Path(temp_output_dir).rglob("*"))
                
                if image_files:
                    # Save image info to JSON
                    image_info = {
                        "model_type": model_type,
                        "disease": selected_disease,
                        "num_images": len(image_files),
                        "images": image_files,
                        "timestamp": timestamp
                    }
                    
                    json_path = Path(temp_output_dir) / "generation_info.json"
                    with open(json_path, 'w') as f:
                        json.dump(image_info, f, indent=2)
                    
                    st.success(f"📄 Saved generation info to {json_path}")
                    
                    # Display images - handle both individual images and grids
                    if len(image_files) == 1:
                        # Single grid image
                        st.image(image_files[0], caption=f"Generated {model_type} samples for {selected_disease}", use_container_width=True)
                        
                        # Download button for grid
                        with open(image_files[0], "rb") as f:
                            st.download_button(
                                label="💾 Download Grid",
                                data=f,
                                file_name=f"{model_type.replace(' ', '_')}_{selected_disease.replace(' ', '_')}_grid.png",
                                mime="image/png",
                                key="download_grid"
                            )
                    else:
                        # Multiple individual images - display in up to 4 columns
                        num_cols = min(4, len(image_files))
                        cols = st.columns(num_cols)
                        
                        for i, img_path in enumerate(image_files):
                            col_idx = i % num_cols
                            with cols[col_idx]:
                                try:
                                    st.image(img_path, caption=f"{selected_disease} #{i+1}", use_container_width=True)
                                except Exception as e:
                                    st.error(f"❌ Failed to display {Path(img_path).name}: {str(e)}")
                                
                                # Download button for each image
                                with open(img_path, "rb") as f:
                                    st.download_button(
                                        label="💾 Download",
                                        data=f,
                                        file_name=f"{selected_disease.replace(' ', '_')}_{i+1}.png",
                                        mime="image/png",
                                        key=f"download_{i}"
                                    )
                else:
                    st.warning("Images were generated but could not be found for display.")
                    
            else:
                st.error(f"Generation failed: {result.stderr}")
                if result.stdout:
                    st.text("STDOUT:")
                    st.code(result.stdout)
                    
        except Exception as e:
            st.error(f"Error generating images: {str(e)}")
            st.exception(e)

