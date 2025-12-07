import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.graph_objects as go
from datetime import datetime


def create_data_overview_section(df, data_source):
    """
    Create and display the data overview section with basic statistics, data structure, and preview.

    Args:
        df: The dataframe to analyze
        data_source: String describing the data source
    """
    st.markdown("---")
    st.markdown("## Data Overview")

    st.markdown(f"**Data Source:** {data_source}")

    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rows", f"{len(df):,}")

    with col2:
        st.metric("Total Columns", len(df.columns))

    with col3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.metric("Numeric Columns", len(numeric_cols))

    with col4:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        st.metric("Categorical Columns", len(categorical_cols))

    # Data types and missing values overview
    st.markdown("### Data Structure Overview")

    try:
        from function_collection import overview
        overview_df = overview(df)

        # Display overview table with row numbers
        overview_df_display = overview_df.reset_index()
        st.dataframe(
            overview_df_display,
            use_container_width=True,
            column_config={
                "dtype": st.column_config.TextColumn("Data Type"),
                "n_unique": st.column_config.NumberColumn("Unique Values", format="%d"),
                "missing_%": st.column_config.NumberColumn("Missing %", format="%.2f%%")
            }
        )

    except Exception as e:
        st.error(f"Error generating overview: {e}")

    # Sample data preview
    st.markdown("### Data Preview")

    col1, col2 = st.columns([3, 1])

    n_rows = st.slider("Number of rows to preview", 5, 100, 10)

    # Add row numbers
    preview_df = df.head(n_rows).copy()
    st.dataframe(preview_df, use_container_width=True)


def create_model_performance_section():
    """
    Create and display the model performance metrics section with comparison table.
    """
    st.markdown("### Model Performance Metrics")
    st.markdown("View performance metrics and statistics for all anomaly detection models.")

    # Define path to JSON summary files
    json_dir = "aml/result/json"

    # List of model summary files
    model_files = {
        "ECOD": "ecod_summary.json",
        "HBOS": "hbos_summary.json",
        "Isolation Forest": "isolation_forest_summary.json",
        "PCA + HBOS": "pca_hbos_summary.json",
        "PCA + Isolation Forest": "pca_isolation_forest_summary.json",
        "HBOS & (PCA+IF)": "hbos_pca_isolation_forest_summary.json",
        "PCA": "pca_outlier_detection_summary.json"
    }

    # Load all available model summaries
    model_summaries = {}
    available_models = []

    for model_name, filename in model_files.items():
        filepath = os.path.join(json_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    model_summaries[model_name] = json.load(f)
                    available_models.append(model_name)
            except Exception as e:
                st.warning(f"⚠️ Could not load {model_name} summary: {e}")

    if not available_models:
        st.warning("⚠️ No model performance summaries found. Please run the anomaly detection models first.")
        st.info("Expected location: `aml/result/json/*_summary.json`")
    else:
        st.success(f"✅ Found {len(available_models)} model performance summaries")

        st.markdown("## Model Performance Comparison")

        comparison_data = []
        for model_name in available_models:
            summary = model_summaries[model_name]
            comparison_data.append({
                "Model": model_name,
                "Total Transactions": summary.get("total_transactions", 0),
                "Outliers Detected": summary.get("outliers", {}).get("count", 0),
                "Outlier Rate (%)": round(summary.get("outliers", {}).get("rate", 0) * 100, 2),
                "Execution Time (s)": round(summary.get("performance", {}).get("total_time_seconds", 0), 2),
                "Avg Score": round(summary.get("anomaly_scores", {}).get("mean", 0), 4),
                "Score Std": round(summary.get("anomaly_scores", {}).get("std", 0), 4),
                "Score Min": round(summary.get("anomaly_scores", {}).get("min", 0), 4),
                "Score Max": round(summary.get("anomaly_scores", {}).get("max", 0), 4),
                "Threshold 95%": round(summary.get("thresholds", {}).get("threshold_95", 0), 4),
            })

        comparison_df = pd.DataFrame(comparison_data)

        # Add row numbers
        comparison_df_display = comparison_df.copy()

        # Display comparison table
        st.dataframe(
            comparison_df_display,
            use_container_width=True,
            column_config={
                "Model": st.column_config.TextColumn("Model"),
                "Total Transactions": st.column_config.NumberColumn("Total Transactions", format="%d"),
                "Outliers Detected": st.column_config.NumberColumn("Outliers Detected", format="%d"),
                "Outlier Rate (%)": st.column_config.NumberColumn("Outlier Rate (%)", format="%.2f%%"),
                "Execution Time (s)": st.column_config.NumberColumn("Execution Time (s)", format="%.2f"),
                "Avg Score": st.column_config.NumberColumn("Avg Score", format="%.4f"),
                "Score Std": st.column_config.NumberColumn("Score Std", format="%.4f"),
                "Score Min": st.column_config.NumberColumn("Score Min", format="%.4f"),
                "Score Max": st.column_config.NumberColumn("Score Max", format="%.4f"),
                "Threshold 95%": st.column_config.NumberColumn("Threshold 95%", format="%.4f"),
            }
        )


def create_score_histogram(scores, model_name, score_col_name):
    """
    Create a histogram plot for anomaly scores with threshold lines.

    Args:
        scores: Series of anomaly scores
        model_name: Name of the model for the plot title
        score_col_name: Name of the score column for display

    Returns:
        plotly figure object
    """
    # Calculate thresholds
    threshold_90 = scores.quantile(0.90)
    threshold_95 = scores.quantile(0.95)

    # Create bins for the histogram
    hist_data, bin_edges = np.histogram(scores, bins=200)

    # Create separate data for each threshold range
    low_scores = scores[scores < threshold_90]
    medium_scores = scores[(scores >= threshold_90) & (scores < threshold_95)]
    high_scores = scores[scores >= threshold_95]

    fig = go.Figure()

    # Add bars for low scores (default color)
    if len(low_scores) > 0:
        low_hist, _ = np.histogram(low_scores, bins=bin_edges)
        fig.add_trace(go.Bar(
            x=bin_edges[:-1],
            y=low_hist,
            width=np.diff(bin_edges),
            marker_color='#4ecdc4',
            opacity=0.7,
            name='Normal Scores',
            showlegend=False
        ))

    # Add bars for medium scores (orange)
    if len(medium_scores) > 0:
        medium_hist, _ = np.histogram(medium_scores, bins=bin_edges)
        fig.add_trace(go.Bar(
            x=bin_edges[:-1],
            y=medium_hist,
            width=np.diff(bin_edges),
            marker_color='orange',
            opacity=0.7,
            name='High Risk (90th percentile)',
            showlegend=False
        ))

    # Add bars for high scores (red)
    if len(high_scores) > 0:
        high_hist, _ = np.histogram(high_scores, bins=bin_edges)
        fig.add_trace(go.Bar(
            x=bin_edges[:-1],
            y=high_hist,
            width=np.diff(bin_edges),
            marker_color='red',
            opacity=0.7,
            name='Critical Risk (95th percentile)',
            showlegend=False
        ))

    # Add mean line
    mean_val = scores.mean()
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Mean: {mean_val:.3f}",
        annotation_position="top",
        annotation_y=0.95
    )

    # Add 95% threshold (Critical)
    fig.add_vline(
        x=threshold_95,
        line_dash="solid",
        line_color="red",
        line_width=3,
        annotation_text=f"Critical (95%): {threshold_95:.3f}",
        annotation_position="top right",
        annotation_y=0.85
    )

    # Add 90% threshold (High)
    fig.add_vline(
        x=threshold_90,
        line_dash="dot",
        line_color="orange",
        line_width=2,
        annotation_text=f"High (90%): {threshold_90:.3f}",
        annotation_position="top left",
        annotation_y=0.85
    )

    fig.update_layout(
        title=f"{model_name} Anomaly Score Distribution",
        xaxis_title="Anomaly Score",
        yaxis_title="Frequency",
        height=400,
        barmode='overlay',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def create_model_visualizations_section(trained_df):
    """
    Create and display the model visualizations section with interactive histograms.

    Args:
        trained_df: The trained dataframe with anomaly scores
    """
    # Model Visualizations with Tabs
    st.markdown("---")
    st.markdown("## Model Visualizations")

    # Define image paths for each model combination
    image_dir = "aml/result/image"

    # Check if trained_df is available for interactive visualizations
    if trained_df is not None and len(trained_df) > 0:

        # Define model score columns mapping
        model_score_columns = {
            "HBOS": ["hbos_anomaly_score"],
            "PCA + Isolation Forest": ["pca_isolation_forest_score"],
            "HBOS & PCA + Isolation Forest": ["hbos_anomaly_score", "pca_isolation_forest_score", "pca_hbos_score"]
        }

        # Create tabs for the three model combinations
        tab_hbos, tab_pca_if, tab_hbos_pca_if = st.tabs(["HBOS", "PCA + IF", "HBOS & PCA + IF"])

        with tab_hbos:

            hbos_cols = model_score_columns["HBOS"]
            available_hbos_cols = [col for col in hbos_cols if col in trained_df.columns]

            if available_hbos_cols:
                # Create histogram for HBOS scores with conditional coloring
                score_col = available_hbos_cols[0]
                scores = trained_df[score_col].dropna()

                fig = create_score_histogram(scores, "HBOS", score_col)
                st.plotly_chart(fig, use_container_width=True)

                # Display basic statistics
                st.markdown("**Score Statistics:**")
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                with stats_col1:
                    st.metric("Mean", f"{scores.mean():.4f}")
                with stats_col2:
                    st.metric("Std Dev", f"{scores.std():.4f}")
                with stats_col3:
                    st.metric("Min", f"{scores.min():.4f}")
                with stats_col4:
                    st.metric("Max", f"{scores.max():.4f}")
            else:
                st.warning("⚠️ HBOS score columns not found in the dataset")

        with tab_pca_if:

            pca_if_cols = model_score_columns["PCA + Isolation Forest"]
            available_pca_if_cols = [col for col in pca_if_cols if col in trained_df.columns]

            if available_pca_if_cols:
                # Create histogram for PCA + Isolation Forest scores with conditional coloring
                score_col = available_pca_if_cols[0]
                scores = trained_df[score_col].dropna()

                fig = create_score_histogram(scores, "PCA + Isolation Forest", score_col)
                st.plotly_chart(fig, use_container_width=True)

                # Display basic statistics
                st.markdown("**Score Statistics:**")
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                with stats_col1:
                    st.metric("Mean", f"{scores.mean():.4f}")
                with stats_col2:
                    st.metric("Std Dev", f"{scores.std():.4f}")
                with stats_col3:
                    st.metric("Min", f"{scores.min():.4f}")
                with stats_col4:
                    st.metric("Max", f"{scores.max():.4f}")
            else:
                st.warning("⚠️ PCA + Isolation Forest score columns not found in the dataset")

        with tab_hbos_pca_if:

            combined_cols = model_score_columns["HBOS & PCA + Isolation Forest"]
            available_combined_cols = [col for col in combined_cols if col in trained_df.columns]

            # Check if the aggregated score is available
            aggregated_score_col = 'hbos_pca_isolation_forest_score'
            if aggregated_score_col in trained_df.columns:
                scores = trained_df[aggregated_score_col].dropna()

                fig = create_score_histogram(scores, "HBOS & PCA + Isolation Forest (Aggregated)", aggregated_score_col)
                st.plotly_chart(fig, use_container_width=True)

                # Display basic statistics
                st.markdown("**Score Statistics:**")
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                with stats_col1:
                    st.metric("Mean", f"{scores.mean():.4f}")
                with stats_col2:
                    st.metric("Std Dev", f"{scores.std():.4f}")
                with stats_col3:
                    st.metric("Min", f"{scores.min():.4f}")
                with stats_col4:
                    st.metric("Max", f"{scores.max():.4f}")
            else:
                st.warning("⚠️ Aggregated score column not found in the dataset")