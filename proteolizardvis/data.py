import abc
import numpy as np
import plotly.graph_objects as go

from ipywidgets import widgets
from proteolizarddata.data import PyTimsDataHandleDDA


class DataLoader(abc.ABC):
    def __init__(self):
        self.data = None
        self.__create_widgets()

    def __create_widgets(self):
        self.dp = widgets.Text(value="",
                               placeholder='path/to/experiment.d',
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


class DDADataLoader(DataLoader):

    def __init__(self):
        super().__init__()

        # load button to create data handle
        self.__create_open_button()
        self.__create_tick_widget()

        # observe rt slider for plot updating
        self.rt_start.observe(self.__handle_slider_change)
        self.rt_stop.observe(self.__handle_slider_change)
        self.dp.value = '/media/hd01/CCSPred/M210115_004_Slot1-1_1_853.d'

        # using update counter to keep track of rt slider update cycles
        self.update_counter = 0

    def __create_open_button(self):
        self.open_button = widgets.Button(description='Open file')
        self.open_button_box = widgets.VBox(children=[self.open_button])
        self.open_button.on_click(self.__on_open_clicked)
        self.controls.children = tuple(
            list(self.controls.children)[:1] + [self.open_button_box] + list(self.controls.children)[1:])

    def __on_open_clicked(self, chage):
        try:
            self.cursor = PyTimsDataHandleDDA(self.dp.value)
            self.rt_start.max = self.cursor.meta_data.Time.values[-1] / 60
            self.rt_stop.max = self.cursor.meta_data.Time.values[-1] / 60
            self.f_idx = dict(np.c_[self.cursor.precursor_frames, np.arange(len(self.cursor.precursor_frames))])
            tic_data = self.cursor.meta_data[self.cursor.meta_data.MsMsType == 0].SummedIntensities.values

            self.tic.data[0].x = np.arange(len(tic_data))
            self.tic.data[0].y = tic_data / np.max(tic_data)

            vals, text = self.__get_vals_and_text(len(self.cursor.precursor_frames),
                                                  rt_start=self.cursor.meta_data.Time.values[0],
                                                  rt_stop=self.cursor.meta_data.Time.values[-1], num_ticks_rt=25)

            self.tic.update_layout(title=f'Total Intensity Count',
                                   xaxis_title='Time [Minutes]',
                                   xaxis=dict(tickvals=vals, ticktext=text),
                                   yaxis_title='Normalized Intensity',
                                   yaxis_range=[0, 1],
                                   template='plotly_white')

            prec_ids = self.cursor.rt_range_to_precursor_frame_ids(self.rt_start.value * 60,
                                                                   self.rt_stop.value * 60)
            self.tic.layout['shapes'] = ()
            self.tic.add_vrect(
                x0=self.f_idx[prec_ids[0]],
                x1=self.f_idx[prec_ids[-1]],
                line_width=1, fillcolor="green", opacity=0.3)

        except Exception as e:
            print(e)

    def __create_tick_widget(self):

        self.tic = go.FigureWidget(data=go.Scatter(x=np.arange(1000),
                                                   y=np.ones(1000)))

        self.tic.update_layout(title=f'Total Intensity Count',
                               xaxis_title='Time [Minutes]',
                               yaxis_title='Normalized Intensity',
                               yaxis_range=[0, 1],
                               template='plotly_white')

        self.tic_box = widgets.VBox(children=[self.tic])

        self.controls.children = tuple(list(self.controls.children) + [self.tic_box])

    def on_load_clicked(self, change):

        try:
            self.data = self.cursor.get_slice_rt_range(self.rt_start.value * 60, self.rt_stop.value * 60)

        except Exception as e:
            print(e)

    def get_data(self):
        return self.data

    def __handle_slider_change(self, change):

        if self.update_counter % 3 == 2:

            try:
                prec_ids = self.cursor.rt_range_to_precursor_frame_ids(self.rt_start.value * 60,
                                                                       self.rt_stop.value * 60)

                num_indices = len(list(self.f_idx.values()))
                binning = np.linspace(0, self.rt_stop.max, num_indices)

                x_start = np.argmin(np.abs(self.rt_start.value - binning))
                x_stop = np.argmin(np.abs(self.rt_stop.value - binning))

                self.tic.layout['shapes'] = ()
                self.tic.add_vrect(
                    x0=x_start,
                    x1=x_stop if x_start <= x_stop else x_start,
                    line_width=1, fillcolor="green", opacity=0.3)

            except Exception as e:
                pass

        # update counter to keep track of cycles
        self.update_counter += 1

    def __get_vals_and_text(self, num_rt, rt_start, rt_stop, num_ticks_rt):

        spacing_rt = int(num_rt / num_ticks_rt)
        range_rt = np.arange(0, num_rt, spacing_rt)

        rt_length = (rt_stop - rt_start) / 60

        spacing_rt = rt_length / num_ticks_rt

        text_rt = []
        for i, _ in enumerate(range_rt):
            s_rt = rt_start / 60 + (i * spacing_rt)
            text_rt.append(str(np.round(s_rt, 1)))

        return range_rt, text_rt
