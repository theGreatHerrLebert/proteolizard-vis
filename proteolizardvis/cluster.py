import abc
import pandas as pd
from abc import ABC

import hdbscan

from datetime import datetime

from sklearn.metrics import pairwise
from proteolizardalgo.clustering import cluster_precursors_hdbscan, cluster_precursors_dbscan

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from ipywidgets import widgets
from proteolizardvis.utility import calculate_statistics, get_initial_histogram, get_discrete_color_swatches, \
    calculate_mz_tick_spacing


class ClusterVisualizer(abc.ABC):
    """

    """

    def __init__(self, data, clusters=None, noise=None):
        """
        :param data:
        :param clusters:
        :param Noise:
        """
        self.data = data
        self.clusters = clusters
        self.noise = noise
        self.__create_widget()

    def __create_widget(self):
        """
        """
        #### ---- POINT CLOUD --- ####
        points = np.array([[1.0, 1.0, 1.0]])
        point_cloud = go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers',
                                   marker=dict(size=5, color='red', opacity=1))

        self.points_widget = go.FigureWidget(data=[point_cloud])
        self.points_widget.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                                         scene={'xaxis': {'title': 'X'},
                                                'yaxis': {'title': 'Y'},
                                                'zaxis': {'title': 'Z', 'dtick': 1}}, template="plotly_white")

        #### ---- VANILLA SCALING --- ####
        self.cycle_scaling = widgets.FloatSlider(value=-0.2, min=-2, max=2, step=0.1, description='cycle scaling',
                                                 continuous_update=False)

        self.scan_scaling = widgets.FloatSlider(value=0.4, min=-2, max=2, step=0.1, description='scan scaling',
                                                continuous_update=False)

        self.mz_scaling = widgets.FloatSlider(value=0.0, min=-5, max=5, step=0.1, description='mz scaling',
                                              continuous_update=False)

        self.resolution = widgets.IntSlider(value=70_000, min=5000,
                                            max=150_000, step=100, description='resolution:', continuous_update=False)

        self.scaling_controls = widgets.HBox(children=[self.cycle_scaling, self.scan_scaling, self.mz_scaling])
        self.cluster_settings = widgets.HBox()
        self.filter_noise = widgets.Checkbox(value=False, description='Remove noise', disabled=False, indent=False)

        self.cluster_button = widgets.Button(description='Cluster')
        self.cluster_button.on_click(self.on_cluster_clicked)

        #### ---- SAVE AND SCORE ---- ####
        self.save_button = widgets.Button(description='Save settings')
        self.save_button.on_click(self.on_save_clicked)

        self.score = widgets.Dropdown(options=['★', '★★', '★★★', '★★★★', '★★★★★'], value='★★★★★',
                                      description='Score', disabled=False)

        self.cluster_colors = widgets.Dropdown(options=get_discrete_color_swatches(), value='Alphabet',
                                               description='Cluster colors',
                                               disabled=False)

        #### ---- CLUSTER SUMMARY STATISTICS --- ####
        self.summary_widget = widgets.Output()

        with self.summary_widget:
            try:
                self.summary_widget.clear_output()
                display(pd.DataFrame({'A': [1], 'B': [1], 'C': [1], 'D': [1]}))
            except Exception as e:
                print(e)

        self.summary_plot_widget = go.FigureWidget(data=get_initial_histogram())
        self.summary_box = widgets.VBox(children=[self.summary_widget, self.summary_plot_widget])
        self.cluster_data = widgets.HBox(children=[self.points_widget, self.summary_box])
        self.filter_settings = widgets.HBox(children=[self.filter_noise, self.resolution])

        #### ---- BUILD WIDGET --- ####
        self.box = widgets.VBox(children=[self.cluster_settings,
                                          self.filter_settings,
                                          self.scaling_controls,
                                          widgets.HBox(children=[self.cluster_button, self.score,
                                                                 self.save_button, self.cluster_colors]),
                                          widgets.HBox(children=[self.points_widget, self.summary_box])])

    @abc.abstractmethod
    def display_widget(self):
        """
        """
        pass

    @abc.abstractmethod
    def on_cluster_clicked(self, change):
        """
        """
        pass

    @abc.abstractmethod
    def on_save_clicked(self, chage):
        """
        """
        pass

    @abc.abstractmethod
    def update_widget(self):
        """
        """
        pass

    def get_general_settings(self):
        """
        """

        score_to_int = {'★': 1, '★★': 2, '★★★': 3, '★★★★': 4, '★★★★★': 5}
        return {'resolution': self.resolution.value,
                'cylce-scale': self.cycle_scaling.value,
                'scan-scale': self.scan_scaling.value,
                'score': score_to_int[self.score.value]}


class DBSCANVisualizer(ClusterVisualizer, ABC):

    def __init__(self, data):
        super().__init__(data)
        self.min_samples = widgets.IntSlider(
            value=7, min=1, max=50, step=1,
            description='min samples:', continuous_update=False)

        self.epsilon = widgets.FloatSlider(
            value=1.7, min=0.5, max=8.0, step=0.1,
            description='epsilon:', continuous_update=True)

        self.metric = widgets.Dropdown(
            options=list(pairwise.PAIRWISE_DISTANCE_FUNCTIONS.keys()),
            value='euclidean', description='Metric:', disabled=False)

        self.cluster_controls = widgets.HBox(children=[self.min_samples, self.epsilon, self.metric])

        self.box.children = tuple(list(self.box.children)[:1] + [self.cluster_controls] + list(self.box.children)[1:])

    def display_widget(self):
        try:
            display(self.box)
        except Exception as e:
            print(e)

    def on_save_clicked(self, change):
        general_settings = self.get_general_settings()
        filter_settings = self.data.get_filter_settings()
        dbscan_settings = {'epsilon': self.epsilon.value,
                           'metric': self.metric.value,
                           'min_samples': self.min_samples.value,
                           'level': 'MS-I',
                           'algorithm': 'DBSCAN'}

        g = dict(filter_settings, **general_settings)
        g = dict(g, **dbscan_settings)
        table = pd.DataFrame(g, index=[0])
        print(table)
        print("save implememtation still pending.")

    def on_cluster_clicked(self, change):
        try:
            self.update_widget()
        except Exception as e:
            print(e)

    def update_widget(self):

        clustered_data = cluster_precursors_dbscan(self.data,
                                                   epsilon=self.epsilon.value,
                                                   min_samples=self.min_samples.value,
                                                   metric=self.metric.value,
                                                   cycle_scaling=self.cycle_scaling.value,
                                                   scan_scaling=self.scan_scaling.value,
                                                   resolution=self.resolution.value)

        self.clusters = clustered_data
        self.noise = self.clusters[self.clusters.label == -1]
        self.clusters = self.clusters[self.clusters.label != -1]

        data, summary_table = calculate_statistics(self.clusters, self.noise)

        with self.summary_widget:
            self.summary_widget.clear_output()
            display(data)

        self.__update_summary_subplots(self.summary_plot_widget, summary_table)

        color_dict = dict(list(enumerate(getattr(px.colors.qualitative, self.cluster_colors.value))))

        if self.filter_noise.value:
            clustered_data = clustered_data[clustered_data.label != -1]

        self.points_widget.data[0].x = clustered_data.cycle / np.power(2, self.cycle_scaling.value)
        self.points_widget.data[0].y = clustered_data.scan / np.power(2, self.scan_scaling.value)
        self.points_widget.data[0].z = clustered_data.mz
        self.points_widget.data[0].marker = dict(size=[2 if l == -1 else 3 for l in clustered_data.label.values],
                                                 color=['grey' if l == -1 else color_dict[l % len(color_dict)]
                                                        for l in clustered_data.label.values], opacity=0.8,
                                                 line=dict(width=0))

        self.points_widget.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene={'xaxis': {'title': 'Cycle'},
                                                                                 'yaxis': {'title': 'Scan'},
                                                                                 'zaxis': {'title': 'mz',
                                                                                           'dtick': 0.8}},
                                         template="plotly_white")

    def __update_summary_subplots(self, fig, summary_table):
        with fig.batch_update():
            fig.data[0].x = summary_table['cycle'].apply(lambda x: len(np.unique(x)))
            fig.data[1].x = summary_table['scan'].apply(lambda x: len(np.unique(x)))
            fig.data[2].x = summary_table['mz'].apply(lambda x: len(np.unique(x)))
            fig.data[3].x = summary_table['scan'].count()


class HDBSCANVisualizer(ClusterVisualizer, ABC):

    def __init__(self, data):
        super().__init__(data)

        ########## ------------------- ##########
        self.person = widgets.Text(placeholder='MyName',
                               description='Person:',
                               layout=widgets.Layout(height="auto", width="auto"))

        self.algorithm = widgets.Dropdown(
            options=['best'], value='best', description='algorithm:', disabled=False)

        self.alpha = widgets.FloatSlider(
            value=1, min=.1, max=5, step=.1, description='alpha:', continuous_update=False)

        self.approx_min_span_tree = widgets.Checkbox(
            value=True, description='approx tree', disabled=False, indent=False)

        self.gen_min_span_tree = widgets.Checkbox(
            value=True, description='generate tree', disabled=False, indent=False)

        self.use_probability = widgets.Checkbox(
            value=True, description='display probability by pointsize', disabled=False, indent=False
        )

        self.leaf_size = widgets.IntSlider(
            value=40, min=1, max=100, step=1, description='leaf size:', continuous_update=False)

        self.min_cluster_size = widgets.IntSlider(
            value=13, min=1, max=100, step=1, description='min cluster size:', continuous_update=False)

        self.min_samples = widgets.IntSlider(
            value=1, min=1, max=50, step=1, description='min samples:', continuous_update=False)

        self.metric = widgets.Dropdown(
            options=list(hdbscan.dist_metrics.METRIC_MAPPING.keys()), value='manhattan', description='Metric:',
            disabled=False)

        self.cluster_controls_1 = widgets.HBox(
            children=[self.algorithm, self.metric, self.gen_min_span_tree, self.approx_min_span_tree,
                      self.use_probability])
        self.cluster_controls_2 = widgets.HBox(
            children=[self.alpha, self.leaf_size, self.min_cluster_size, self.min_samples])

        self.cluster_settings = widgets.VBox(children=[self.person, self.cluster_controls_1, self.cluster_controls_2])

        self.box.children = tuple(list(self.box.children)[:1] + [self.cluster_settings] + list(self.box.children)[1:])

    def display_widget(self):
        try:
            display(self.box)
        except Exception as e:
            print(e)

    def on_save_clicked(self, change):
        general_settings = self.get_general_settings()
        filter_settings = self.data.get_filter_settings()

        hdbscan_settings = {
            'person': self.person.value,
            'algorithm': self.algorithm.value,
                            'alpha': self.alpha.value,
                            'approx-min-span-tree': self.approx_min_span_tree.value,
                            'gen-min-span-tree': self.gen_min_span_tree.value,
                            'leaf-size': self.leaf_size.value,
                            'min-cluster-size': self.min_cluster_size.value,
                            'min-samples': self.min_samples.value,
                            'metric': self.metric.value,
                            'cycle-scaling': self.cycle_scaling.value,
                            'scan-scaling': self.scan_scaling.value,
                            'resolution': self.resolution.value}

        g = dict(filter_settings, **general_settings)
        g = dict(g, **hdbscan_settings)
        table = pd.DataFrame(g, index=[0])

        try:
            now = datetime.now()
            current_time = now.strftime("%d-%m-%y-%H-%M-%S")
            table.to_csv(f'PRECURSOR-HDBSCAN-{current_time}.csv', index=False)

        except Exception as e:
            print(e)

    def on_cluster_clicked(self, change):
        try:
            self.update_widget()
        except Exception as e:
            print(e)

    def update_widget(self):

        clustered_data = cluster_precursors_hdbscan(self.data.filtered_data.get_precursor_coords3D().get_points(),
                                                    algorithm=self.algorithm.value,
                                                    alpha=self.alpha.value,
                                                    approx_min_span_tree=self.approx_min_span_tree.value,
                                                    gen_min_span_tree=self.gen_min_span_tree.value,
                                                    leaf_size=self.leaf_size.value,
                                                    min_cluster_size=self.min_cluster_size.value,
                                                    min_samples=self.min_samples.value,
                                                    metric=self.metric.value,
                                                    cycle_scaling=self.cycle_scaling.value,
                                                    scan_scaling=self.scan_scaling.value,
                                                    resolution=self.resolution.value,
                                                    mz_scaling=self.mz_scaling.value)

        self.clusters = clustered_data
        self.noise = self.clusters[self.clusters.label == -1]
        self.clusters = self.clusters[self.clusters.label != -1]

        data, summary_table = calculate_statistics(self.clusters, self.noise)

        with self.summary_widget:
            self.summary_widget.clear_output()
            display(data)

        self.__update_summary_subplots(self.summary_plot_widget, summary_table)

        color_dict = dict(list(enumerate(getattr(px.colors.qualitative, self.cluster_colors.value))))

        if self.filter_noise.value:
            clustered_data = clustered_data[clustered_data.label != -1]

        self.points_widget.data[0].x = clustered_data.cycle / np.power(2, self.cycle_scaling.value)
        self.points_widget.data[0].y = clustered_data.scan / np.power(2, self.scan_scaling.value)
        self.points_widget.data[0].z = clustered_data.mz

        up = self.use_probability.value
        bps = 3.5

        self.points_widget.data[0].marker = dict(
            size=[2 if l == -1 else bps * p * int(up) + bps * (1 - int(up)) for l, p in
                  zip(clustered_data.label.values, clustered_data.probability.values)],
            color=['grey' if l == -1 else color_dict[l % len(color_dict)]
                   for l in clustered_data.label.values], opacity=0.8,
            line=dict(width=0))

        tick_spacing = calculate_mz_tick_spacing(np.min(clustered_data.mz), np.max(clustered_data.mz))

        self.points_widget.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene={'xaxis': {'title': 'Cycle'},
                                                                                 'yaxis': {'title': 'Scan'},
                                                                                 'zaxis': {'title': 'mz',
                                                                                           'dtick': tick_spacing}},
                                         template="plotly_white")

    def __update_summary_subplots(self, fig, summary_table):
        with fig.batch_update():
            fig.data[0].x = summary_table['cycle'].apply(lambda x: len(np.unique(x)))
            fig.data[1].x = summary_table['scan'].apply(lambda x: len(np.unique(x)))
            fig.data[2].x = summary_table['mz'].apply(lambda x: len(np.unique(x)))
            fig.data[3].x = summary_table['scan'].count()
