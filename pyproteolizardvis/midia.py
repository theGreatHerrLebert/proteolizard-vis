import abc
from data import DataLoader
from pyproteolizard.midia import MidiaExperiment, MidiaCursor


class MIDIADataLoader(DataLoader, abc.ABC):
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
