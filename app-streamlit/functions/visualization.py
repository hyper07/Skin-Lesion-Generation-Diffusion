"""
Visualization and plotting utilities for AML data analysis
Contains functions for creating charts and plots from database data
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Tuple


"""
Visualization and plotting utilities for AML data analysis
Contains functions for creating charts and plots from database data
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def create_anomaly_score_distribution_plot_from_data(bin_data: pd.DataFrame,
                                                   score_column: str = 'hbos_pca_isolation_forest_score',
                                                   bins: int = 50, figsize: tuple = (12, 6)):
    """
    Create a bar plot showing the distribution of anomaly scores with specified number of bins
    Args:
        bin_data: DataFrame with histogram bin data from get_anomaly_score_histogram_bins
        score_column: Name of the score column being analyzed (for display purposes)
        bins: Number of bins for the histogram (for display purposes)
        figsize: Figure size for the plot (default: (12, 6))
    Returns:
        matplotlib figure object
    """
    if bin_data is None or len(bin_data) == 0:
        raise ValueError(f"No data found for score column '{score_column}'")

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create bars using bin centers and counts
    bars = ax.bar(bin_data['bin_center'], bin_data['count'],
                  width=(bin_data['bin_end'] - bin_data['bin_start']).iloc[0] * 0.9,
                  alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)

    # Customize the plot
    ax.set_title(f'Distribution of {score_column.replace("_", " ").title()} ({bins} bins)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Score Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels if there are many bins
    if bins > 20:
        ax.tick_params(axis='x', rotation=45)

    # Add some statistics as text
    total_count = bin_data['count'].sum()
    mean_score = (bin_data['bin_center'] * bin_data['count']).sum() / total_count
    ax.text(0.02, 0.98, f'Total Records: {total_count:,}\nMean Score: {mean_score:.4f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


def create_anomaly_score_distribution_plot(score_column: str = 'hbos_pca_isolation_forest_score',
                                         bins: int = 50, use_host: bool = False,
                                         table_name: str = "transactions", figsize: tuple = (12, 6)):
    """
    Create a bar plot showing the distribution of anomaly scores with specified number of bins
    This is a convenience function that gets data from database and creates the plot
    Args:
        score_column: Name of the score column to analyze (default: 'hbos_pca_isolation_forest_score')
        bins: Number of bins for the histogram (default: 50)
        use_host: If True, use localhost with host port (for connections from host machine)
        table_name: Name of the table to query (default: "transactions")
        figsize: Figure size for the plot (default: (12, 6))
    Returns:
        matplotlib figure object
    """
    # Import here to avoid circular imports
    from .database import get_anomaly_score_histogram_bins

    # Get histogram bin data
    bin_data = get_anomaly_score_histogram_bins(score_column, bins, use_host, table_name)

    # Create the plot
    return create_anomaly_score_distribution_plot_from_data(bin_data, score_column, bins, figsize)


def create_score_distribution_comparison_plot(score_stats_df: pd.DataFrame, figsize: tuple = (14, 8)):
    """
    Create a comparison plot showing distribution statistics for multiple score columns
    Args:
        score_stats_df: DataFrame from get_score_distribution() with stats for each score type
        figsize: Figure size for the plot (default: (14, 8))
    Returns:
        matplotlib figure object
    """
    if score_stats_df is None or len(score_stats_df) == 0:
        raise ValueError("No score distribution data provided")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Score Distribution Statistics Comparison', fontsize=16, fontweight='bold')

    score_types = score_stats_df['score_type'].unique()

    for i, score_type in enumerate(score_types):
        ax = axes[i//2, i%2]
        score_data = score_stats_df[score_stats_df['score_type'] == score_type]

        if len(score_data) > 0:
            row = score_data.iloc[0]

            # Create box plot style visualization
            stats_labels = ['Min', 'Q25', 'Median', 'Q75', 'Max']
            stats_values = [row['min'], row['q25'], row['median'], row['q75'], row['max']]

            # Plot the distribution box
            ax.plot([1]*len(stats_values), stats_values, 'o', color='blue', alpha=0.7)
            ax.plot([1, 1], [row['q25'], row['q75']], 'b-', linewidth=2)  # IQR box
            ax.plot([0.8, 1.2], [row['median'], row['median']], 'r-', linewidth=2)  # median line

            # Add whiskers
            ax.plot([1, 1], [row['min'], row['q25']], 'b--', alpha=0.5)
            ax.plot([1, 1], [row['q75'], row['max']], 'b--', alpha=0.5)

            ax.set_title(f'{score_type.replace("_", " ").title()}', fontsize=12)
            ax.set_ylabel('Score Value')
            ax.grid(True, alpha=0.3)

            # Add statistics text
            stats_text = f'Count: {row["count"]:,}\nMean: {row["mean"]:.4f}\nStd: {row["std"]:.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

    plt.tight_layout()
    return fig


def create_transaction_amount_histogram(amount_data: pd.Series, bins: int = 50,
                                     title: str = "Transaction Amount Distribution",
                                     figsize: tuple = (12, 6)):
    """
    Create a histogram for transaction amounts
    Args:
        amount_data: Series of transaction amounts
        bins: Number of bins for the histogram
        title: Plot title
        figsize: Figure size for the plot
    Returns:
        matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create histogram
    n, bins_edges, patches = ax.hist(amount_data, bins=bins, alpha=0.7, color='green', edgecolor='black')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Transaction Amount', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add statistics
    ax.text(0.02, 0.98, f'Total Transactions: {len(amount_data):,}\nMean: {amount_data.mean():.2f}\nMedian: {amount_data.median():.2f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


def create_risk_score_heatmap(score_data: pd.DataFrame, x_col: str, y_col: str,
                            value_col: str, figsize: tuple = (10, 8)):
    """
    Create a heatmap showing relationships between different risk scores
    Args:
        score_data: DataFrame with score columns
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        value_col: Column name for the values to display
        figsize: Figure size for the plot
    Returns:
        matplotlib figure object
    """
    # Create pivot table for heatmap
    pivot_data = score_data.pivot_table(values=value_col, index=y_col, columns=x_col, aggfunc='mean')

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(pivot_data, cmap='YlOrRd', aspect='auto')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(value_col.replace('_', ' ').title(), rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot_data.columns)))
    ax.set_yticks(np.arange(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.columns)
    ax.set_yticklabels(pivot_data.index)

    # Rotate x-axis labels
    ax.tick_params(axis='x', rotation=45)

    ax.set_title(f'{value_col.replace("_", " ").title()} Heatmap', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig
def create_score_distribution_comparison_plot(score_stats_df: pd.DataFrame, figsize: tuple = (14, 8)):
    """
    Create a comparison plot showing distribution statistics for multiple score columns
    Args:
        score_stats_df: DataFrame from get_score_distribution() with stats for each score type
        figsize: Figure size for the plot (default: (14, 8))
    Returns:
        matplotlib figure object
    """
    if score_stats_df is None or len(score_stats_df) == 0:
        raise ValueError("No score distribution data provided")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Score Distribution Statistics Comparison', fontsize=16, fontweight='bold')

    score_types = score_stats_df['score_type'].unique()

    for i, score_type in enumerate(score_types):
        ax = axes[i//2, i%2]
        score_data = score_stats_df[score_stats_df['score_type'] == score_type]

        if len(score_data) > 0:
            row = score_data.iloc[0]

            # Create box plot style visualization
            stats_labels = ['Min', 'Q25', 'Median', 'Q75', 'Max']
            stats_values = [row['min'], row['q25'], row['median'], row['q75'], row['max']]

            # Plot the distribution box
            ax.plot([1]*len(stats_values), stats_values, 'o', color='blue', alpha=0.7)
            ax.plot([1, 1], [row['q25'], row['q75']], 'b-', linewidth=2)  # IQR box
            ax.plot([0.8, 1.2], [row['median'], row['median']], 'r-', linewidth=2)  # median line

            # Add whiskers
            ax.plot([1, 1], [row['min'], row['q25']], 'b--', alpha=0.5)
            ax.plot([1, 1], [row['q75'], row['max']], 'b--', alpha=0.5)

            ax.set_title(f'{score_type.replace("_", " ").title()}', fontsize=12)
            ax.set_ylabel('Score Value')
            ax.grid(True, alpha=0.3)

            # Add statistics text
            stats_text = f'Count: {row["count"]:,}\nMean: {row["mean"]:.4f}\nStd: {row["std"]:.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

    plt.tight_layout()
    return fig


def create_transaction_amount_histogram(amount_data: pd.Series, bins: int = 50,
                                     title: str = "Transaction Amount Distribution",
                                     figsize: tuple = (12, 6)):
    """
    Create a histogram for transaction amounts
    Args:
        amount_data: Series of transaction amounts
        bins: Number of bins for the histogram
        title: Plot title
        figsize: Figure size for the plot
    Returns:
        matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create histogram
    n, bins_edges, patches = ax.hist(amount_data, bins=bins, alpha=0.7, color='green', edgecolor='black')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Transaction Amount', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add statistics
    ax.text(0.02, 0.98, f'Total Transactions: {len(amount_data):,}\nMean: {amount_data.mean():.2f}\nMedian: {amount_data.median():.2f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


def create_risk_score_heatmap(score_data: pd.DataFrame, x_col: str, y_col: str,
                            value_col: str, figsize: tuple = (10, 8)):
    """
    Create a heatmap showing relationships between different risk scores
    Args:
        score_data: DataFrame with score columns
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        value_col: Column name for the values to display
        figsize: Figure size for the plot
    Returns:
        matplotlib figure object
    """
    # Create pivot table for heatmap
    pivot_data = score_data.pivot_table(values=value_col, index=y_col, columns=x_col, aggfunc='mean')

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(pivot_data, cmap='YlOrRd', aspect='auto')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(value_col.replace('_', ' ').title(), rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot_data.columns)))
    ax.set_yticks(np.arange(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.columns)
    ax.set_yticklabels(pivot_data.index)

    # Rotate x-axis labels
    ax.tick_params(axis='x', rotation=45)

    ax.set_title(f'{value_col.replace("_", " ").title()} Heatmap', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig