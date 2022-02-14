import plotly.graph_objects as go
from ipywidgets import widgets
import numpy as np
import abc
import plotly.express as px


class ImsPointCloudVisualizer(abc.ABC):
    def __init__(self, data):
        self.data = data
        self.__create_widgets()

    def __create_widgets(self):
        points = np.array([[1.0, 1.0, 1.0, 1.0]])
        point_cloud = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(size=5, color=np.log(points[:, 3]), colorscale='inferno', opacity=1))

        self.points_widget = go.FigureWidget(data=[point_cloud])
        self.points_widget.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                                         scene={'xaxis': {'title': 'X'},
                                                'yaxis': {'title': 'Y'},
                                                'zaxis': {'title': 'Z', 'dtick': 1}},
                                         template="plotly_white")

        self.opacity_slider = widgets.FloatSlider(value=0.5, min=0.1, max=1, step=0.1, description='opacity:',
                                                  continuous_update=False)

        self.point_size_slider = widgets.FloatSlider(value=1, min=0.1, max=5.0, step=0.1, description='point size:',
                                                     continuous_update=False)

        self.color_scale = widgets.Dropdown(options=sorted(px.colors.named_colorscales()),
                                            value='inferno', description='color scale:', disabled=False)

        self.point_controls = widgets.HBox(children=[self.opacity_slider, self.point_size_slider, self.color_scale])

        self.update_button = widgets.Button(description='Update')
        self.update_button.on_click(self.on_update_clicked)

        self.box = widgets.VBox \
            (children=[self.point_controls, self.update_button, self.points_widget])

    @abc.abstractmethod
    def on_update_clicked(self, change):
        """
        """
        pass

    @abc.abstractmethod
    def display_widgets(self):
        """
        """
        pass
