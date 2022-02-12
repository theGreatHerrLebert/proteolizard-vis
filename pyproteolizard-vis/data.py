import abc

from pyproteolizard.midia import MidiaExperiment, MidiaCursor
from ipywidgets import widgets


class DataLoader(abc.ABC):
    def __init__(self):
        self.data = None
        self.__create_widgets()

    def __create_widgets(self):
        self.dp = widgets.Text(placeholder='path/to/experiment.d',
                               description='Data path',
                               layout=widgets.Layout(height="auto", width="auto"))

        self.path_controls = widgets.VBox(children=[self.dp])

        self.rt_start = widgets.FloatSlider(value=20.0, min=0.0,
                                            max=46.0, step=0.5, description='Rt start:', continuous_update=True)

        self.rt_stop = widgets.FloatSlider(value=22.0, min=0.0, max=46.0, step=0.5, description='Rt stop:',
                                           continuous_update=True)

        self.load_button = widgets.Button(description="Load slice")

        self.load_button.on_click(self.on_load_clicked)

        self.load_controls = widgets.HBox(children=[self.rt_start, self.rt_stop, self.load_button])
        self.controls = widgets.VBox(children=[self.path_controls, self.load_controls])

    def display_widgets(self):
        try:
            display(self.controls)
        except Exception as e:
            print(e)

    @abc.abstractmethod
    def on_load_clicked(self, change):
        """
        """
        pass

    @abc.abstractmethod
    def get_data(self):
        """
        """
        pass


class MidiaDataLoader(DataLoader, abc.ABC):
    def __init__(self):
        super().__init__()
        self.ew = widgets.Text(
            placeholder='path/to/windows.h5',
            description='Window path',
            layout=widgets.Layout(height="auto", width="auto"))

        self.path_controls.children = tuple(list(self.path_controls.children) + [self.ew])

    def on_load_clicked(self, change):
        try:
            self.cursor = MidiaCursor(self.dp.value)
            self.exp = MidiaExperiment(self.dp.value, self.ew.value)
            self.data = self.cursor.get_slice_retention_time(self.rt_start.value * 60, self.rt_stop.value * 60)
        except Exception as e:
            print(e)

    def get_data(self):
        return self.data

