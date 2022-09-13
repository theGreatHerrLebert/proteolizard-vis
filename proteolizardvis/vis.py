import plotly.graph_objects as go
from ipywidgets import widgets

from proteolizardvis.data import DDADataLoader
from proteolizardvis.filter import DDAPrecursorFilter
from proteolizardvis.point import DDAPrecursorPointCloudVis
from proteolizardvis.surface import TimsSurfaceVisualizer
from proteolizardvis.cluster import DBSCANVisualizer, TimsHasherVisualizer


class TimsVisualizer:
    def __init__(self):
        self.data_loader = DDADataLoader()
        self.data_loader.dp.value = '/media/hd01/CCSPred/M210116_001_Slot1-1_1_859.d/'
        self.data_loader.rt_start.value = 20.0
        self.data_loader.rt_stop.value = 20.5

        self.prec_filter = DDAPrecursorFilter(self.data_loader)
        self.prec_filter.mz_min.value = 950.0
        self.prec_filter.mz_max.value = 1000
        self.prec_filter.intensity_min.value = 50
        self.prec_filter.sc_max.value = 1000.0

        self.precursor_points = DDAPrecursorPointCloudVis(self.prec_filter)

        self.surface = TimsSurfaceVisualizer(self.data_loader, self.prec_filter)

        self.dbscan = DBSCANVisualizer(data=self.prec_filter)

        self.hasher_vis = TimsHasherVisualizer(self.prec_filter)

        self.tab = widgets.Tab(
            children=[self.data_loader.controls,
                      self.prec_filter.controls,
                      self.precursor_points.box,
                      self.surface.widgets,
                      self.hasher_vis.controls,
                      self.dbscan.box
                      ])

        titles = ['Data', 'MS-I filter', 'MS-I points', 'Surface', 'LSH', 'DBSCAN']

        [self.tab.set_title(i, title) for i, title in enumerate(titles)]

    def display_widgets(self):
        display(self.tab)
