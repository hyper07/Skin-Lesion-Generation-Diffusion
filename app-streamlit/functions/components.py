import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode
from functions.downloaderCSV import download_button

def create_transaction_table(filtered_df, display_columns, display_columns_detail, thresholds=None, n_rows=50, key="preview_table"):
    """
    Create and display an AgGrid table for transaction data with grouped columns and custom styling.

    Args:
        filtered_df: The filtered dataframe to display
        df: The original full dataframe (for threshold calculations)
        display_columns: List of original column names
        display_columns_detail: List of display column names
        n_rows: Number of rows per page (default: 50)
        thresholds: Dictionary with model-specific thresholds from get_transaction_counts()

    Returns:
        grid_response: The AgGrid response object
    """

    total_transactions = len(filtered_df)
    # Prepare data for AgGrid - work directly with filtered_df to avoid copies
    if 'hbos_pca_if_decile_rank' in filtered_df.columns:
        available_display_columns = [col for col in display_columns if col in filtered_df.columns]
        preview_df = filtered_df[available_display_columns].copy()
        preview_df['hbos_pca_if_decile_rank'] = filtered_df['hbos_pca_if_decile_rank']
        cols_to_keep = [col for col in preview_df.columns if col != 'hbos_pca_if_decile_rank']
        preview_df = preview_df[cols_to_keep + ['hbos_pca_if_decile_rank']]
    else:
        available_display_columns = [col for col in display_columns if col in filtered_df.columns]
        preview_df = filtered_df[available_display_columns]

    # Create mapping from original column names to display labels
    column_mapping = dict(zip(display_columns, display_columns_detail))
    if 'hbos_pca_if_decile_rank' in preview_df.columns:
        column_mapping['hbos_pca_if_decile_rank'] = 'Rank'

    # Format score columns to 2 decimal places
    score_columns = [col for col in preview_df.columns if 'score' in col.lower()]
    for col in score_columns:
        if col in preview_df.columns:
            preview_df[col] = pd.to_numeric(preview_df[col], errors='coerce').round(2)

    # Round CURRENCY_AMOUNT to 2 decimals
    if 'CURRENCY_AMOUNT' in preview_df.columns:
        preview_df['CURRENCY_AMOUNT'] = pd.to_numeric(preview_df['CURRENCY_AMOUNT'], errors='coerce').astype(float).round(2)

    # Format DATE_KEY to show date only
    if 'DATE_KEY' in preview_df.columns:
        preview_df['DATE_KEY'] = pd.to_datetime(preview_df['DATE_KEY'], errors='coerce').dt.strftime('%Y-%m-%d')
        preview_df['DATE_KEY'] = preview_df['DATE_KEY'].fillna('')

    # Rename columns to use display labels
    preview_df = preview_df.rename(columns=column_mapping)

    # Get score columns for rendering
    original_score_columns = [col for col in filtered_df.columns if 'score' in col.lower()]
    score_columns_renamed = [column_mapping.get(col, col) for col in original_score_columns if col in column_mapping]
    allowed_score_names = ['HBOS Score', 'PCA IF Score', 'HBOS PCA IF Score']
    score_columns_renamed = [col for col in score_columns_renamed if col in allowed_score_names]

    # Configure AgGrid options
    gb = GridOptionsBuilder.from_dataframe(preview_df)
    gb.configure_selection('single')
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=n_rows)
    gb.configure_side_bar()
    gb.configure_default_column(
        resizable=True,
        sortable=True,
        filterable=True,
        editable=False,
        menu=True
    )

    # Create grouped column definitions
    column_defs = []

    # Transaction Info Group
    transaction_children = []
    if 'Transaction Key' in preview_df.columns:
        transaction_children.append({
            "field": "Transaction Key",
            "width": 100,
            "headerName": "KEY",
            "pinned": "left",
            "filter": "agTextColumnFilter"
        })
    if 'Date' in preview_df.columns:
        transaction_children.append({
            "field": "Date",
            "width": 100,
            "headerName": "Date",
            "pinned": "left",
            "filter": "agTextColumnFilter"
        })
    if 'Amount' in preview_df.columns:
        transaction_children.append({
            "field": "Amount",
            "width": 150,
            "cellStyle": {"textAlign": "right"},
            "pinned": "left",
            "cellRenderer": JsCode("function(params) { if (params.value != null && params.value !== undefined) { var val = parseFloat(params.value); return '' + val.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2}); } return ''; }")
        })
    if transaction_children:
        column_defs.append({
            "headerName": "Transaction Info",
            "children": transaction_children,
            "headerGroupClass": "ag-header-group-cell-center"
        })

    # Byorder Info Group
    byorder_children = []
    if 'ByordType' in preview_df.columns:
        byorder_children.append({
            "field": "ByordType",
            "headerName": "Type",
            "width": 100,
            "filter": "agTextColumnFilter"
        })
    if 'ByordID' in preview_df.columns:
        byorder_children.append({
            "field": "ByordID",
            "headerName": "ID",
            "width": 100,
            "filter": "agTextColumnFilter"
        })
    if byorder_children:
        column_defs.append({
            "headerName": "Byorder Info",
            "children": byorder_children,
            "headerGroupClass": "ag-header-group-cell-center"
        })

    # Beneficiary Info Group
    beneficiary_children = []
    if 'BeneType' in preview_df.columns:
        beneficiary_children.append({
            "field": "BeneType",
            "headerName": "Type",
            "width": 100,
            "filter": "agTextColumnFilter"
        })
    if 'BeneID' in preview_df.columns:
        beneficiary_children.append({
            "field": "BeneID",
            "headerName": "ID",
            "width": 100,
            "filter": "agTextColumnFilter"
        })
    if beneficiary_children:
        column_defs.append({"headerName": "Beneficiary Info", "children": beneficiary_children})

    # Mechanism column
    if 'Mechanism' in preview_df.columns:
        column_defs.append({
            "field": "Mechanism",
            "width": 150,
            "filter": "agTextColumnFilter"
        })

    # Anomaly Scores Group with colored dots
    score_children = []

    for col in score_columns_renamed:
        if col in preview_df.columns:
            score_child = {
                "field": col,
                "width": 100,
                "cellStyle": {"textAlign": "left"},
                "type": ["numericColumn", "numberColumnFilter"]
            }

            # Add colored dot renderer
            original_score_col = [k for k, v in column_mapping.items() if v == col][0] if col in column_mapping.values() else None
            if original_score_col and original_score_col in filtered_df.columns:
                # Use model-specific thresholds if available, otherwise fall back to global percentiles
                if thresholds:
                    # Map display names to threshold keys
                    threshold_mapping = {
                        'HBOS Score': ('hbos_anomaly_score_high_risk_threshold', 'hbos_anomaly_score_critical_threshold'),
                        'PCA IF Score': ('pca_isolation_forest_score_high_risk_threshold', 'pca_isolation_forest_score_critical_threshold'),
                        'HBOS PCA IF Score': ('hbos_pca_isolation_forest_score_high_risk_threshold', 'hbos_pca_isolation_forest_score_critical_threshold')
                    }
                    
                    if col in threshold_mapping:
                        high_risk_key, critical_key = threshold_mapping[col]
                        if high_risk_key in thresholds and critical_key in thresholds:
                            high_risk_threshold = round(thresholds[high_risk_key], 2)
                            critical_threshold = round(thresholds[critical_key], 2)
                        else:
                            # Fallback to global percentiles if specific thresholds not available
                            series_all = pd.to_numeric(filtered_df[original_score_col], errors='coerce').dropna()
                            if len(series_all) > 0:
                                high_risk_threshold = round(float(np.percentile(series_all, 90.0)), 2)
                                critical_threshold = round(float(np.percentile(series_all, 95.0)), 2)
                            else:
                                continue
                    else:
                        # Fallback for unmapped columns
                        series_all = pd.to_numeric(filtered_df[original_score_col], errors='coerce').dropna()
                        if len(series_all) > 0:
                            high_risk_threshold = round(float(np.percentile(series_all, 90.0)), 2)
                            critical_threshold = round(float(np.percentile(series_all, 95.0)), 2)
                        else:
                            continue
                else:
                    # Fallback to global percentiles if no thresholds provided
                    series_all = pd.to_numeric(filtered_df[original_score_col], errors='coerce').dropna()
                    if len(series_all) > 0:
                        high_risk_threshold = round(float(np.percentile(series_all, 90.0)), 2)
                        critical_threshold = round(float(np.percentile(series_all, 95.0)), 2)
                    else:
                        continue

                # Apply the colored dot renderer with the calculated thresholds
                _dot_renderer_template = """
                    class MissionResultRenderer {
                      init(params) {
                        const value = params && params.value != null ? parseFloat(params.value) : null;
                        const wrapper = document.createElement('span');
                        wrapper.style.display = 'inline-flex';
                        wrapper.style.alignItems = 'center';
                        if (value === null || isNaN(value)) { this.eGui = wrapper; return; }
                        const roundedValue = parseFloat(value.toFixed(2));
                        let color = '#17a2b8';
                        if (roundedValue >= __CRIT__) { color = '#dc3545'; }
                        else if (roundedValue >= __HIGH__) { color = '#ffc107'; }
                        const dot = document.createElement('span');
                        dot.style.display = 'inline-block';
                        dot.style.width = '12px';
                        dot.style.height = '12px';
                        dot.style.borderRadius = '50%';
                        dot.style.background = color;
                        dot.style.marginRight = '8px';
                        const text = document.createElement('span');
                        text.textContent = value.toFixed(2);
                        wrapper.appendChild(dot);
                        wrapper.appendChild(text);
                        this.eGui = wrapper;
                      }
                      getGui() { return this.eGui; }
                      refresh(params) { return false; }
                    }
                """
                dot_renderer = JsCode(_dot_renderer_template.replace("__CRIT__", str(critical_threshold)).replace("__HIGH__", str(high_risk_threshold)))
                score_child["cellRenderer"] = dot_renderer
                score_child["valueFormatter"] = None
                score_child["headerName"] = col.upper().replace('HBOS PCA', 'HBOS&PCA').replace(' SCORE', '').replace(' ', '+')

            score_children.append(score_child)

    if score_children:
        column_defs.append({"headerName": "Anomaly Scores", "children": score_children, "cellStyle": {"textAlign": "center"}})

    # Add Rank column if exists
    if 'Rank' in preview_df.columns:
        column_defs.append({
            "field": "Rank",
            "width": 60,
            "filter": "agTextColumnFilter",
            "pinned": "right"
        })
    # Add Rank column if exists
    if 'KEY_INDEX' in preview_df.columns:
        column_defs.append({
            "hide": True,
            "field": "KEY_INDEX",
            "width": 100,
            "filter": "agTextColumnFilter",
        })
    # Build grid options
    grid_options = gb.build()
    grid_options['columnDefs'] = column_defs

    # Configure row selection
    grid_options['rowSelection'] = {
        'mode': 'singleRow',
        'checkboxes': False,
        'enableClickSelection': True,
    }

    # Configure row styling for selected rows
    get_row_style_js = JsCode("""
        function(params) {
            if (params.node.isSelected()) {
                return {
                    'backgroundColor': '#e3f2fd',
                    'color': '#000000'
                };
            }
            return null;
        }
    """)
    grid_options['getRowStyle'] = get_row_style_js
    grid_options["pagination"] = True
    # Calculate total pages from the actual data being displayed
    total_pages = math.ceil(total_transactions / n_rows) if total_transactions > 0 else 1

    # Display AgGrid
    return AgGrid(
        preview_df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=True,
        height=400,
        default_page_size=25,
        paginationPageSize=total_pages,
        theme='bootstrap',
        title='Transaction Overview',
        allow_unsafe_jscode=True,
        key=key
    )


def create_risk_cards(critical_count, high_risk_count, normal_count, critical_pct, high_risk_pct, normal_pct,
                     critical_threshold_display, high_risk_range_display, normal_range_display):
    """
    Create and display risk categorization cards with counts and percentages.

    Args:
        critical_count: Number of critical risk transactions
        high_risk_count: Number of high risk transactions
        normal_count: Number of normal risk transactions
        critical_pct: Percentage of critical risk transactions
        high_risk_pct: Percentage of high risk transactions
        normal_pct: Percentage of normal risk transactions
        critical_threshold_display: Display string for critical threshold
        high_risk_range_display: Display string for high risk range
        normal_range_display: Display string for normal range
    """

    # Critical Risk Card
    st.markdown(f"""
    <div class="risk-card-container" style="border-left: 5px solid #dc3545;">
        <div class="risk-card-header">
            <span class="risk-card-label">Critical Risk</span>
            <span class="risk-card-value" style="color: #dc3545;">{critical_count:,}</span>
        </div>
        <div class="risk-card-details">
            <span style="font-weight: 600;">{critical_pct:.2f}%</span> of total | Score : >= {critical_threshold_display:.3f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # High Risk Card
    st.markdown(f"""
    <div class="risk-card-container" style="border-left: 5px solid #ffc107;">
        <div class="risk-card-header">
            <span class="risk-card-label">High Risk</span>
            <span class="risk-card-value" style="color: #ffc107;">{high_risk_count:,}</span>
        </div>
        <div class="risk-card-details">
            <span style="font-weight: 600;">{high_risk_pct:.2f}%</span> of total | Score: >= {high_risk_range_display:.3f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Normal Risk Card
    st.markdown(f"""
    <div class="risk-card-container" style="border-left: 5px solid #17a2b8;">
        <div class="risk-card-header">
            <span class="risk-card-label">Normal Risk</span>
            <span class="risk-card-value" style="color: #17a2b8;">{normal_count:,}</span>
        </div>
        <div class="risk-card-details">
            <span style="font-weight: 600;">{normal_pct:.2f}%</span> of total | Score: < {normal_range_display}
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_transaction_pattern_analysis(df, selected_row, selected_model, model_info, high_risk_percentile, format_currency):
    """
    Create and display transaction pattern analysis with cumulative chart and statistics.

    Args:
        df: The full dataframe
        selected_row: The selected transaction row
        selected_model: The selected model name
        model_info: Dictionary with model information
        high_risk_percentile: High risk percentile threshold
        format_currency: Function to format currency amounts
    """
    import plotly.graph_objects as go
    import pycountry

    # Check if required columns exist for the graph
    if not ('DATE_KEY' in df.columns and 'CURRENCY_AMOUNT' in df.columns and 'byorder_to_bene' in df.columns):
        st.info("Transaction pattern analysis requires DATE_KEY, CURRENCY_AMOUNT, and byorder_to_bene columns.")
        return

    # Get the byorder_to_bene from the selected transaction
    if isinstance(selected_row, pd.Series):
        selected_byorder_to_bene = selected_row.get('byorder_to_bene', None) if 'byorder_to_bene' in selected_row.index else None
    elif isinstance(selected_row, dict):
        selected_byorder_to_bene = selected_row.get('byorder_to_bene', None)
    else:
        selected_byorder_to_bene = None

    # Debug: Check if byorder_to_bene was found
    if selected_byorder_to_bene is None or pd.isna(selected_byorder_to_bene):
        st.warning(f"⚠️ Could not find 'byorder_to_bene' in selected row. Available columns: {', '.join(selected_row.index.tolist()[:10]) if isinstance(selected_row, pd.Series) else 'N/A'}")
        return

    if selected_byorder_to_bene is None:
        return

    score_col = model_info[selected_model]["column"]

    # Convert percentile to actual score threshold
    score_data_for_threshold = pd.to_numeric(df[score_col], errors='coerce').dropna()
    if len(score_data_for_threshold) == 0:
        st.warning("No valid score data available for threshold calculation")
        return

    high_risk_threshold = float(np.percentile(score_data_for_threshold, high_risk_percentile))

    # Filter transactions for this byorder_to_bene AND at/above high risk threshold
    if 'byorder_to_bene' not in df.columns:
        st.error("❌ 'byorder_to_bene' column not found in dataframe")
        return

    # Convert byorder_to_bene to string for comparison
    df_filter = df.copy()
    df_filter['byorder_to_bene'] = df_filter['byorder_to_bene'].astype(str)
    selected_byorder_to_bene_str = str(selected_byorder_to_bene)

    # Filter by byorder_to_bene
    byorder_bene_transactions = df_filter[df_filter['byorder_to_bene'] == selected_byorder_to_bene_str].copy()

    if len(byorder_bene_transactions) == 0:
        st.warning(f"⚠️ No transactions found for byorder_to_bene: {selected_byorder_to_bene_str}")
        return

    # Convert score to numeric and filter by threshold
    byorder_bene_transactions['score_numeric'] = pd.to_numeric(
        byorder_bene_transactions[score_col], errors='coerce'
    )
    high_risk_transactions = byorder_bene_transactions[
        byorder_bene_transactions['score_numeric'] >= high_risk_threshold
    ].copy()

    if len(high_risk_transactions) == 0:
        st.info(f"No similar transactions found for byorder_to_bene: {selected_byorder_to_bene_str} with {selected_model} score ≥ {high_risk_percentile:.1f}th percentile ({high_risk_threshold:.4f})")

        # Show all transactions for this byorder_to_bene (regardless of risk)
        if len(byorder_bene_transactions) > 0:
            st.markdown(f"#### All Transactions for Byorder-to-Beneficiary: {selected_byorder_to_bene_str}")
            st.info(f"Total transactions: {len(byorder_bene_transactions)} (none meet high risk threshold)")
        return

    # Convert DATE_KEY to datetime
    high_risk_transactions['DATE_KEY'] = pd.to_datetime(high_risk_transactions['DATE_KEY'])
    high_risk_transactions = high_risk_transactions.sort_values('DATE_KEY')

    # Calculate cumulative amount
    high_risk_transactions['CUMULATIVE_AMOUNT'] = high_risk_transactions['CURRENCY_AMOUNT'].cumsum()

    # Format amounts for hover display
    high_risk_transactions['CUMULATIVE_AMOUNT_FORMATTED'] = high_risk_transactions['CUMULATIVE_AMOUNT'].apply(format_currency)
    high_risk_transactions['CURRENCY_AMOUNT_FORMATTED'] = high_risk_transactions['CURRENCY_AMOUNT'].apply(format_currency)

    # Extract values from selected_byorder_to_bene
    split_values = selected_byorder_to_bene_str.split('_')
    byorder_id = split_values[0] if len(split_values) > 0 else "Unknown"
    bene_id = split_values[1] if len(split_values) > 1 else "Unknown"
    byorder_country = split_values[2] if len(split_values) > 2 else "999"
    bene_country = split_values[3] if len(split_values) > 3 else "999"
    mechanism = '_'.join(split_values[4:]) if len(split_values) > 4 else "Unknown"
    mechanism = mechanism.replace('_', ' ') if mechanism != "Unknown" else mechanism

    # Helper function to safely get country name
    def get_country_name(country_code):
        if country_code == '999':
            return "Unknown"
        try:
            country = pycountry.countries.get(alpha_3=country_code)
            return country.name if country else "Unknown"
        except (AttributeError, KeyError):
            return "Unknown"

    byorder_country_name = get_country_name(byorder_country)
    bene_country_name = get_country_name(bene_country)

    # Show summary statistics
    st.markdown("""
        <style>
        div[data-testid="stMetric"] {
            text-align: center;
        }
        div[data-testid="stMetric"] > div {
            justify-content: center;
        }
        div[data-testid="stMetricValue"] {
            text-align: center !important;
            justify-content: center !important;
        }
        div[data-testid="stMetricLabel"] {
            text-align: center !important;
            justify-content: center !important;
        }
        </style>
    """, unsafe_allow_html=True)

    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    with stats_col1:
        st.metric("Similar Transactions", f"{len(high_risk_transactions):,}")
    with stats_col2:
        total_amount = high_risk_transactions['CURRENCY_AMOUNT'].sum()
        st.metric("Total Amount", format_currency(total_amount))
    with stats_col3:
        avg_amount = high_risk_transactions['CURRENCY_AMOUNT'].mean()
        st.metric("Average Amount", format_currency(avg_amount))
    with stats_col4:
        max_amount = high_risk_transactions['CURRENCY_AMOUNT'].max()
        st.metric("Maximum Amount", format_currency(max_amount))

    # Create interactive line chart with Plotly
    fig = go.Figure()

    # Include all transactions in the main plot to keep the line continuous
    plot_transactions = high_risk_transactions.copy()
    selected_transaction_data = None

    # Get selected transaction key from session state
    selected_key = st.session_state.get('selected_rows', [{}])[0].get('Transaction Key') if st.session_state.get('selected_rows') else None

    if selected_key and 'TRANSACTION_KEY' in high_risk_transactions.columns:
        selected_mask = high_risk_transactions['TRANSACTION_KEY'] == selected_key
        if selected_mask.any():
            selected_transaction_data = high_risk_transactions[selected_mask].iloc[0]

    # Get scores as lists, handling missing columns
    hbos_scores = pd.to_numeric(plot_transactions['hbos_anomaly_score'], errors='coerce').fillna(0).tolist() if 'hbos_anomaly_score' in plot_transactions.columns else [0] * len(plot_transactions)
    pca_if_scores = pd.to_numeric(plot_transactions['pca_isolation_forest_score'], errors='coerce').fillna(0).tolist() if 'pca_isolation_forest_score' in plot_transactions.columns else [0] * len(plot_transactions)
    hbos_pca_iso_scores = pd.to_numeric(plot_transactions['hbos_pca_isolation_forest_score'], errors='coerce').fillna(0).tolist() if 'hbos_pca_isolation_forest_score' in plot_transactions.columns else [0] * len(plot_transactions)

    # Structure customdata
    customdata = list(zip(
        plot_transactions['TRANSACTION_KEY'] if 'TRANSACTION_KEY' in plot_transactions.columns else ['N/A'] * len(plot_transactions),
        plot_transactions['CURRENCY_AMOUNT_FORMATTED'],
        plot_transactions['CUMULATIVE_AMOUNT_FORMATTED'],
        [byorder_id] * len(plot_transactions),
        [byorder_country_name] * len(plot_transactions),
        [bene_id] * len(plot_transactions),
        [bene_country_name] * len(plot_transactions),
        [mechanism] * len(plot_transactions),
        hbos_scores,
        pca_if_scores,
        hbos_pca_iso_scores
    ))

    # Add scatter plot for cumulative transactions
    fig.add_trace(go.Scatter(
        x=plot_transactions['DATE_KEY'],
        y=plot_transactions['CUMULATIVE_AMOUNT'],
        mode='markers+lines',
        name='Cumulative Amount',
        marker=dict(
            size=10,
            color=plot_transactions['score_numeric'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title=f"{selected_model}<br>Score"),
            line=dict(width=1, color='white'),
            cmin=high_risk_threshold,
            cmax=high_risk_transactions['score_numeric'].max()
        ),
        line=dict(color='rgba(200, 50, 50, 0.5)', width=3),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br>' +
                     '<b>Transaction Key:</b> %{customdata[0]}<br>' +
                     '<b>Transaction Amount:</b> %{customdata[1]}<br>' +
                     '<b>Cumulative Amount:</b> %{customdata[2]}<br>' +
                     '<b>ByordID:</b> %{customdata[3]}<br>' +
                     '<b>Byorder Country:</b> %{customdata[4]}<br>' +
                     '<b>Bene ID:</b> %{customdata[5]}<br>' +
                     '<b>Bene Country:</b> %{customdata[6]}<br>' +
                     '<b>Mechanism:</b> %{customdata[7]}<br>' +
                     '<b>HBOS Score:</b> %{customdata[8]:.2f}<br>' +
                     '<b>PCA+IF Score:</b> %{customdata[9]:.2f}<br>' +
                     '<b>HBOS&PCA+IF Score:</b> %{customdata[10]:.2f}<br>' +
                     '<extra></extra>',
        customdata=customdata
    ))

    # Highlight the selected transaction if it exists in the chart
    if selected_transaction_data is not None:
        sel_trans = selected_transaction_data
        sel_hbos_score = pd.to_numeric(sel_trans.get('hbos_anomaly_score', 0), errors='coerce') if 'hbos_anomaly_score' in sel_trans.index else 0
        sel_pca_if_score = pd.to_numeric(sel_trans.get('pca_isolation_forest_score', 0), errors='coerce') if 'pca_isolation_forest_score' in sel_trans.index else 0
        sel_hbos_pca_iso_score = pd.to_numeric(sel_trans.get('hbos_pca_isolation_forest_score', 0), errors='coerce') if 'hbos_pca_isolation_forest_score' in sel_trans.index else 0

        sel_score_numeric = pd.to_numeric(sel_trans.get('score_numeric', 0), errors='coerce') if 'score_numeric' in sel_trans.index else 0

        sel_customdata = [[
            str(sel_trans.get('TRANSACTION_KEY', 'N/A')),
            str(sel_trans.get('CURRENCY_AMOUNT_FORMATTED', 'N/A')),
            str(sel_trans.get('CUMULATIVE_AMOUNT_FORMATTED', 'N/A')),
            str(byorder_id),
            str(byorder_country_name),
            str(bene_id),
            str(bene_country_name),
            str(mechanism),
            sel_hbos_score,
            sel_pca_if_score,
            sel_hbos_pca_iso_score
        ]]

        fig.add_trace(go.Scatter(
            x=[sel_trans['DATE_KEY']],
            y=[sel_trans['CUMULATIVE_AMOUNT']],
            mode='markers',
            name='Selected Transaction',
            marker=dict(
                size=10,
                symbol='star',
                color='gold',
                line=dict(width=1, color='gray'),
                opacity=1.0
            ),
            hovertemplate='<b>SELECTED TRANSACTION</b><br>' +
                         '<b>Date:</b> %{x|%Y-%m-%d}<br>' +
                         '<b>Transaction Key:</b> %{customdata[0]}<br>' +
                         '<b>Transaction Amount:</b> %{customdata[1]}<br>' +
                         '<b>Cumulative Amount:</b> %{customdata[2]}<br>' +
                         '<b>ByordID:</b> %{customdata[3]}<br>' +
                         '<b>Byorder Country:</b> %{customdata[4]}<br>' +
                         '<b>Bene ID:</b> %{customdata[5]}<br>' +
                         '<b>Bene Country:</b> %{customdata[6]}<br>' +
                         '<b>Mechanism:</b> %{customdata[7]}<br>' +
                         '<b>HBOS Score:</b> %{customdata[8]:.2f}<br>' +
                         '<b>PCA+IF Score:</b> %{customdata[9]:.2f}<br>' +
                         '<b>HBOS&PCA+ISO Score:</b> %{customdata[10]:.2f}<br>' +
                         '<extra></extra>',
            customdata=sel_customdata,
            showlegend=True
        ))

    # Update layout
    fig.update_layout(
        title=f'Model: {selected_model} <br>ByordID: {byorder_id} | Bene ID: {bene_id} | Byorder Country: {byorder_country_name} | Bene Country: {bene_country_name} | Mechanism: {mechanism}',
        xaxis_title='Date',
        yaxis_title='Cumulative Transaction Amount ($)',
        hovermode='closest',
        height=500,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.98,
            xanchor="left",
            x=-0.05
        )
    )

    # Format y-axis to show currency with B/M/K format
    max_cumulative = high_risk_transactions['CUMULATIVE_AMOUNT'].max()
    min_cumulative = high_risk_transactions['CUMULATIVE_AMOUNT'].min()

    if max_cumulative > 0:
        range_val = max_cumulative - min_cumulative
        if range_val > 0:
            num_ticks = 6
            tick_values = np.linspace(min_cumulative, max_cumulative, num_ticks)
            tick_labels = [format_currency(val) for val in tick_values]
            fig.update_yaxes(tickvals=tick_values, ticktext=tick_labels)
        else:
            fig.update_yaxes(tickformat='$,.0f')
    else:
        fig.update_yaxes(tickformat='$,.0f')

    st.plotly_chart(fig, use_container_width=True)

    # Free up memory after creating chart
    del high_risk_transactions, fig


def create_risk_time_series_plot(df, score_col, critical_threshold, high_risk_threshold):
    """
    Create and display a time series plot showing critical and high risk transactions over time.

    Args:
        df: The dataframe containing transaction data
        score_col: The column name containing anomaly scores
        critical_threshold: The threshold for critical risk transactions
        high_risk_threshold: The threshold for high risk transactions
    """
    # Check if DATE_KEY exists and score column is valid for time series
    if 'DATE_KEY' in df.columns and score_col in df.columns:
        # Cache key for time series data
        timeseries_cache_key = f"timeseries_{score_col}_{critical_threshold}_{high_risk_threshold}_{len(df)}_{df['DATE_KEY'].iloc[0] if len(df) > 0 else 'empty'}"
        
        if ('timeseries_cache_key' in st.session_state and st.session_state.timeseries_cache_key == timeseries_cache_key and
            'critical_daily' in st.session_state and 'high_risk_daily' in st.session_state):
            # Use cached time series data
            critical_daily = st.session_state.critical_daily
            high_risk_daily = st.session_state.high_risk_daily
        else:
            # Prepare data for time series using full dataset for meaningful trends
            df_with_scores = df.copy()  # Need full dataset for time series analysis
            df_with_scores[score_col] = pd.to_numeric(df_with_scores[score_col], errors='coerce')

            # Filter valid scores and dates
            valid_data = df_with_scores[
                (df_with_scores[score_col].notna()) &
                (df_with_scores['DATE_KEY'].notna())
            ].copy()

            if len(valid_data) > 0:
                # Convert DATE_KEY to datetime
                valid_data['DATE_KEY'] = pd.to_datetime(valid_data['DATE_KEY'], errors='coerce')
                valid_data = valid_data[valid_data['DATE_KEY'].notna()].copy()

                if len(valid_data) > 0:
                    # Group by date and count critical transactions
                    critical_daily = valid_data[
                        valid_data[score_col] >= critical_threshold
                    ].groupby(valid_data['DATE_KEY'].dt.date).size().reset_index(name='count')
                    critical_daily.columns = ['date', 'count']
                    critical_daily['date'] = pd.to_datetime(critical_daily['date'])
                    critical_daily = critical_daily.sort_values('date')
                    
                    # Group by date and count high risk transactions
                    high_risk_daily = valid_data[
                        (valid_data[score_col] >= high_risk_threshold) & 
                        (valid_data[score_col] < critical_threshold)
                    ].groupby(valid_data['DATE_KEY'].dt.date).size().reset_index(name='count')
                    high_risk_daily.columns = ['date', 'count']
                    high_risk_daily['date'] = pd.to_datetime(high_risk_daily['date'])
                    high_risk_daily = high_risk_daily.sort_values('date')
                    
                    # Cache the time series data
                    st.session_state.critical_daily = critical_daily
                    st.session_state.high_risk_daily = high_risk_daily
                    st.session_state.timeseries_cache_key = timeseries_cache_key
                else:
                    critical_daily = pd.DataFrame()
                    high_risk_daily = pd.DataFrame()
            else:
                critical_daily = pd.DataFrame()
                high_risk_daily = pd.DataFrame()
        
        # Create plot if we have data
        if not critical_daily.empty or not high_risk_daily.empty:
            # Create combined smooth line plot
            fig_combined = go.Figure()
            
            # Add high risk trace
            fig_combined.add_trace(go.Scatter(
                x=high_risk_daily['date'],
                y=high_risk_daily['count'],
                mode='lines',
                name='High Risk Transactions',
                line=dict(color='#ffc107', width=3, shape='spline'),
                fill='tozeroy',
                fillcolor='rgba(255, 193, 7, 0.2)',
                hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br>' +
                                '<b>High Risk Count:</b> %{y:,}<br>' +
                                '<extra></extra>'
            ))
            
            # Add critical trace
            fig_combined.add_trace(go.Scatter(
                x=critical_daily['date'],
                y=critical_daily['count'],
                mode='lines',
                name='Critical Risk Transactions',
                line=dict(color='#dc3545', width=3, shape='spline'),
                fill='tozeroy',
                fillcolor='rgba(220, 53, 69, 0.2)',
                hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br>' +
                                '<b>Critical Count:</b> %{y:,}<br>' +
                                '<extra></extra>'
            ))
            
            fig_combined.update_layout(
                title='Risk Transactions Over Time',
                yaxis_title='Number of Transactions',
                height=380,
                template='plotly_white',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=40, r=40, t=80, b=40)
            )
            
            st.plotly_chart(fig_combined, use_container_width=True)
        else:
            st.info("No valid data available for time series")
    else:
        st.info(f"Score column '{score_col}' not available for time series")
