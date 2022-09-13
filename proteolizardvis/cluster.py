import abc
import pandas as pd
from abc import ABC

import hdbscan

from datetime import datetime

from sklearn.metrics import pairwise
from proteolizardalgo.clustering import cluster_precursors_hdbscan, cluster_precursors_dbscan
from proteolizardalgo.hashing import TimsHasher

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from ipywidgets import widgets
from proteolizardvis.utility import calculate_statistics, get_initial_histogram, get_discrete_color_swatches, \
    calculate_mz_tick_spacing

from proteolizarddata.data import TimsFrame, TimsSlice


class ClusterVisualizer(abc.ABC):

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
                                          widgets.HBox(children=[self.cluster_button,
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
        return {'resolution': self.resolution.value,
                'cylce-scale': self.cycle_scaling.value,
                'scan-scale': self.scan_scaling.value,
                'score': score_to_int[self.score.value]}


class DBSCANVisualizer(ClusterVisualizer, abc.ABC):

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

    def on_cluster_clicked(self, change):
        try:
            self.update_widget()
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

    def update_widget(self):

        clustered_data = cluster_precursors_dbscan(self.data.filtered_data.get_precursor_points(),
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

        tick_spacing = calculate_mz_tick_spacing(np.min(clustered_data.mz), np.max(clustered_data.mz))

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
                                                                                           'dtick': tick_spacing}},
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


class TimsHasherVisualizer:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.hashed_data = None
        self.__create_widgets()

    def __create_widgets(self):
        self.num_trials = widgets.IntSlider(value=32, min=1, max=128, description='number trials')
        self.len_single_trial = widgets.IntSlider(value=20, min=1, max=64, description='length trial')
        self.hash_button = widgets.Button(description='Hash slice')
        self.hash_button.on_click(self.on_hash_clicked)

        self.resolution = widgets.IntSlider(value=1, min=0, max=4, description='resolution')
        self.len_window = widgets.IntSlider(value=10, min=3, max=100, description='length mz')

        self.__create_point_widget()

        self.hasher_settings_1 = widgets.HBox(children=[self.num_trials, self.len_single_trial,
                                                        self.hash_button])
        self.hasher_settings_2 = widgets.HBox(children=[self.resolution, self.len_window])

        self.controls = widgets.VBox(children=[self.hasher_settings_1, self.hasher_settings_2,
                                               self.point_box])

    def display_widgets(self):
        try:
            display(self.controls)

        except Exception as e:
            print(e)

    def __create_point_widget(self):
        points = np.array([[1.0, 1.0, 1.0, 1.0]])
        point_cloud = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(size=5, color=np.log(points[:, 3]), colorscale='geyser', opacity=1))

        self.points_widget = go.FigureWidget(data=[point_cloud])
        self.points_widget.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                                         scene={'xaxis': {'title': 'X'},
                                                'yaxis': {'title': 'Y'},
                                                'zaxis': {'title': 'Z', 'dtick': 1}},
                                         template="plotly_white")

        self.opacity_slider = widgets.FloatSlider(value=0.3, min=0.1, max=1, step=0.1, description='opacity:',
                                                  continuous_update=False)

        self.point_size_slider = widgets.FloatSlider(value=0.5, min=0.1, max=5.0, step=0.1, description='point size:',
                                                     continuous_update=False)

        self.color_scale = widgets.Dropdown(options=sorted(px.colors.named_colorscales()),
                                            value='geyser', description='color scale:', disabled=False)

        self.point_controls = widgets.HBox(children=[self.opacity_slider, self.point_size_slider, self.color_scale])

        self.update_button = widgets.Button(description='Update')
        self.update_button.on_click(self.on_update_clicked)

        self.point_box = widgets.VBox \
            (children=[self.point_controls, self.update_button, self.points_widget])

    def on_hash_clicked(self, change):
        precursor_frames = self.data_loader.filtered_data.get_precursor_frames()

        hasher = TimsHasher(trials=self.num_trials.value, len_trial=self.len_single_trial.value,
                            resolution=self.resolution.value, num_dalton=self.len_window.value)

        hashed_precs = []
        for frame in precursor_frames:
            try:
                hashed_precs.append(hasher(frame))
            except:
                pass

        self.hashed_data = TimsSlice(None, [f.frame_ptr for f in hashed_precs], [])

    def on_update_clicked(self, change):
        points = self.hashed_data.get_precursor_points().values
        f = np.sort(np.unique(points[:, 0]))
        f_idx = dict(np.c_[f, np.arange(f.shape[0])])

        self.points_widget.data[0].x = [f_idx[x] for x in points[:, 0]]
        self.points_widget.data[0].y = points[:, 1]
        self.points_widget.data[0].z = points[:, 3]
        self.points_widget.data[0].marker = dict(size=self.point_size_slider.value,
                                                 color=np.log(points[:, 4]),
                                                 colorscale=self.color_scale.value,
                                                 line=dict(width=0),
                                                 opacity=self.opacity_slider.value)

        tick_spacing = calculate_mz_tick_spacing(np.min(points[:, 3]), np.max(points[:, 3]))

        self.points_widget.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                                         scene={'xaxis': {'title': 'Rt-Index'},
                                                'yaxis': {'title': 'Mobility-Index'},
                                                'zaxis': {'title': 'm/z', 'dtick': tick_spacing}},
                                         template="plotly_white")
