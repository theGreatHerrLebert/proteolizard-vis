import abc


class MassSpecDataFilter(abc.ABC):
    def __init__(self, data_loader, filtered_data=None):
        self.data_loader = data_loader
        self.filtered_data = filtered_data
        self.__create_widgets()
        self.display_widgets()

    def __create_widgets(self):

        # mz controls
        self.mz_min = widgets.BoundedFloatText(value=500.0, min=0.0, max=2500.0, step=.1, description='m/z min:',
                                               disabled=False)
        self.mz_max = widgets.BoundedFloatText(value=510.0, min=1.0, max=2500.0, step=.1, description='m/z max:',
                                               disabled=False)
        # intensity controls
        self.intensity_min = widgets.IntSlider(value=200, min=0, max=500, step=10, description='Min intensity:',
                                               continuous_update=False)

        # filter button
        self.filter_button = widgets.Button(description="Filter Precursors")
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


class MidiaPrecursorScanFilter(MassSpecDataFilter, abc.ABC):
    def __init__(self, data_loader, filtered_data=None):
        self.filtered_data = filtered_data
        super().__init__(data_loader)

        self.sc_min = widgets.BoundedIntText(value=250, min=0, max=1000, step=1, description='scan min:',
                                             disabled=False)
        self.sc_max = widgets.BoundedIntText(value=350, min=1, max=1000, step=1, description='scan max:',
                                             disabled=False)

        self.scan_controls = widgets.HBox(children=[self.sc_min, self.sc_max])
        mz_controls = list(self.controls.children)
        self.controls.children = tuple(mz_controls[:1] + [self.scan_controls] + mz_controls[1:])

    def on_filter_clicked(self, change):
        midia_slice = self.data_loader.get_data()
        self.filtered_data = midia_slice.filter_precursors(
            scan_min=self.sc_min.value,
            scan_max=self.sc_max.value,
            mz_min=self.mz_min.value,
            mz_max=self.mz_max.value,
            intensity_min=self.intensity_min.value)
    
    def get_data(self):
        return self.filtered_data

