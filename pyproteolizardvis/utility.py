import abc
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots


def calculate_statistics(clusters, noise):
    sum_intensity_clusters = clusters.groupby(['label'])['intensity'].sum().sum()
    sum_intensity_noise = noise.groupby(['label'])['intensity'].sum().sum()

    intensity_ratio = np.round(sum_intensity_clusters / sum_intensity_noise, 3)

    num_points_clusters = clusters.shape[0]
    num_points_noise = noise.shape[0]

    points_ratio = np.round(num_points_clusters / num_points_noise, 3)

    num_clusters = int(clusters.label.max())

    table_dict = {'Total': [num_points_clusters + num_points_noise,
                            sum_intensity_clusters + sum_intensity_noise, num_clusters],
                  'Cluster': [num_points_clusters, sum_intensity_clusters, num_clusters],
                  'Noise': [num_points_noise, sum_intensity_noise, 0],
                  'Ratio': [points_ratio, intensity_ratio, 0]}

    summary_table = pd.DataFrame(table_dict, index=['Points', 'Intensity', 'Clusters'])

    sum_table = clusters.groupby('label')

    return summary_table, sum_table


def get_discrete_color_swatches():
    """
    """
    return list(filter(lambda x: x.find('_') == -1, [attr for attr in dir(px.colors.qualitative)]))


def get_initial_histogram():
    """
    helper for generation of template vis, can be used for init or custom widget inspection
    """

    # create 4 histograms: cluster rt distribution, scan distibution, mz distribution, size distribution
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Cycle", "Scan", "m/z", "Size"))
    fig.add_trace(go.Histogram(x=np.random.normal(size=1000)), row=1, col=1)
    fig.add_trace(go.Histogram(x=np.random.normal(size=1000)), row=1, col=2)
    fig.add_trace(go.Histogram(x=np.random.normal(size=1000)), row=2, col=1)
    fig.add_trace(go.Histogram(x=np.random.normal(size=1000)), row=2, col=2)
    fig.update_layout(title_text="Cluster distributions", showlegend=False,
                      template="plotly_white")
    return fig
