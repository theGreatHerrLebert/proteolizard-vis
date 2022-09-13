import tensorflow as tf
import plotly.graph_objects as go
from ipywidgets import widgets
import numpy as np
import plotly.express as px

class TimsSurfaceVisualizer:
    def __init__(self, data_loader, data_filter):
        self.data_loader = data_loader
        self.data_filter = data_filter
        self.create_widgets()

    def __frame_to_rt(self):
        return dict(zip(self.data_loader.cursor.meta_data.Id, self.data_loader.cursor.meta_data.Time))

    def create_widgets(self):
        self.resolution = widgets.IntSlider(value=1, min=0, max=3, description='Resolution: ')
        self.normalization = widgets.Dropdown(options=['identity', 'sqrt', 'log'],
                                              value='identity', description='Normalize:')

        self.exclude_dim = widgets.Dropdown(options=['retention time', 'scan', 'mz'],
                                            value='mz', description='Exclude:')

        self.color_scale = widgets.Dropdown(options=sorted(px.colors.named_colorscales()),
                                            value='inferno', description='color scale:')

        self.controls = widgets.HBox(children=[self.resolution, self.normalization, self.exclude_dim, self.color_scale])

        self.surface = go.FigureWidget(data=[go.Surface(z=np.array([[0.0]]), colorscale='inferno')])

        self.surface.update_layout(title='', autosize=True, width=800, height=800, template='simple_white')

        self.display_button = widgets.Button(description='Display')

        self.display_button_box = widgets.HBox(children=[self.display_button])

        self.display_button.on_click(self.on_display_clicked)

        self.surface_box = widgets.HBox(children=[self.surface])

        self.widgets = widgets.VBox(children=[self.controls, self.display_button_box, self.surface_box])

    def on_display_clicked(self, change):
        sparse_tensor, f_min, f_max, scan_min, scan_max = self.data_filter.filtered_data.vectorize \
            (self.resolution.value).get_zero_indexed_sparse_tensor()

        if self.exclude_dim.value == 'mz':
            folded_tensor = tf.sparse.reduce_sum(sparse_tensor, axis=2)
            self.surface.data[0].z = folded_tensor

        elif self.exclude_dim.value == 'scan':
            folded_tensor = tf.sparse.reduce_sum(sparse_tensor, axis=1)
            self.surface.data[0].z = folded_tensor

        else:
            folded_tensor = tf.sparse.reduce_sum(sparse_tensor, axis=0)
            self.surface.data[0].z = folded_tensor

        if self.normalization.value == 'sqrt':
            self.surface.data[0].z = np.sqrt(self.surface.data[0].z)

        elif self.normalization.value == 'log':
            self.surface.data[0].z = np.log(self.surface.data[0].z + 1)

        self.surface.data[0].colorscale = self.color_scale.value

        self.surface.update_layout(
            scene = dict(xaxis_title='Ion mobility [1/K0]',
                         # xaxis = dict(tickvals=r_dt, ticktext=t_dt),
                         yaxis_title='Retention time [minutes]',
                         # yaxis = dict(tickvals=r_rt, ticktext=t_rt),
                         zaxis_title='Intensity',
                         zaxis = dict())
        )