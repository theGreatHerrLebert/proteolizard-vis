import abc
from ipywidgets import widgets


class MassSpecDataFilter(abc.ABC):
    def __init__(self, data_loader, filtered_data=None):
        self.data_loader = data_loader
        self.filtered_data = filtered_data
        self.__create_widgets()

    def __create_widgets(self):

        # mz controls
        self.mz_min = widgets.BoundedFloatText(value=500.0, min=0.0, max=2500.0, step=.1, description='m/z min:',
                                               disabled=False)
        self.mz_max = widgets.BoundedFloatText(value=510.0, min=1.0, max=2500.0, step=.1, description='m/z max:',
                                               disabled=False)
        # intensity controls
        self.intensity_min = widgets.IntSlider(value=150, min=0, max=1000, step=10, description='Min intensity:',
                                               continuous_update=False)

        # filter button
        self.filter_button = widgets.Button(description="Filter")
        self.filter_button.on_click(self.on_filter_clicked)
        self.filter_controls = widgets.HBox(children=[self.mz_min, self.mz_max, self.intensity_min])
        self.controls = widgets.VBox(children=[self.filter_controls, self.filter_button])

    def display_widgets(self):
        try:
            display(self.controls)
        except Exception as e:
            print(e)

    @abc.abstractmethod
    def on_filter_clicked(self, change):
        """
        """
        pass
    
    @abc.abstractmethod
    def get_data(self):
        """
        """
        pass

    @abc.abstractmethod
    def get_filter_settings(self):
        """
        """
        pass
