import os
import sys
import datetime
from collections import OrderedDict
from types import SimpleNamespace
import traceback

from PyQt5 import uic
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QApplication

from neuroconnect.connect_math import dist_from_file
from neuroconnect.main import main as control_main


class DesignerUI(object):
    def __init__(self, design_path):
        Form, Window = uic.loadUiType(design_path)
        self.app = QApplication([])
        self.window = Window()
        self.ui = Form()
        self.ui.setupUi(self.window)
        self.file_dialog = QFileDialog()

    def start(self):
        self.window.show()
        self.app.exec_()

    def getWidgets(self):
        return self.app.allWidgets()

    def getWidgetNames(self):
        return [w.objectName() for w in self.getWidgets()]


class CodeTimeUI(DesignerUI):
    def __init__(self, design_name):
        super().__init__(design_name)
        self.init_vars()
        self.linkNames()
        self.setup()
        self.parse_info()

    def init_vars(self):
        self.file_location = None
        self.connect_dists = {}
        self.connect_dists["forward"] = OrderedDict()
        self.connect_dists["recurrent"] = OrderedDict()
        self.connect_dists["localA"] = OrderedDict()
        self.connect_dists["localB"] = OrderedDict()

    def linkNames(self):
        # Model
        self.na = self.ui.lineA
        self.nb = self.ui.lineB
        self.sa = self.ui.lineSA
        self.sb = self.ui.lineSB
        self.ta = self.ui.lineTA
        self.tb = self.ui.lineTB
        self.a_info = [self.na, self.sa, self.ta]
        self.b_info = [self.nb, self.sb, self.tb]

        self.connections = {}
        self.connections["forward"] = [self.ui.MinF, self.ui.MaxF]
        self.connections["recurrent"] = [self.ui.MinR, self.ui.MaxR]
        self.connections["localA"] = [self.ui.MinA, self.ui.MaxA]
        self.connections["localB"] = [self.ui.MinB, self.ui.MaxB]
        self.connect_f = self.ui.CustF
        self.connect_r = self.ui.CustR
        self.connect_a = self.ui.CustA
        self.connect_b = self.ui.CustB
        self.connect_f.setEnabled(True)
        self.connect_r.setEnabled(True)
        self.connect_a.setEnabled(True)
        self.connect_b.setEnabled(True)

        # Parameters
        self.params = {}
        self.params["max_depth"] = self.ui.line_MD
        self.params["clt_start"] = self.ui.line_CLT
        self.params["subsample_rate"] = self.ui.line_SR
        self.params["num_iters"] = self.ui.line_GI
        self.params_box = {}
        self.params_box["approx_hypergeo"] = self.ui.box_AH
        self.params_box["do_stats"] = self.ui.box_SE
        self.params_box["do_graph"] = self.ui.box_DG
        self.params_box["do_mean"] = self.ui.box_ME

        self.run_button = self.ui.RunButton
        self.file_select_button = self.ui.SelectButton
        self.info_text = self.ui.InfoBox
        self.info_text.setReadOnly(True)

    def setup(self):
        home = os.path.expanduser("~")
        self.default_loc = os.path.join(home, ".skm_neural_connections", "default.txt")
        os.makedirs(os.path.dirname(self.default_loc), exist_ok=True)
        self.file_select_button.clicked.connect(self.selectDir)
        self.run_button.clicked.connect(self.start_main)
        try:
            self.get_default_save_location()
        except BaseException:
            self.file_location = None

        self.connect_f.clicked.connect(self.fConnect)
        self.connect_r.clicked.connect(self.rConnect)
        self.connect_a.clicked.connect(self.aConnect)
        self.connect_b.clicked.connect(self.bConnect)

    def parse_dist(self, location):
        return dist_from_file(location)

    def fConnect(self):
        self.selectFile()
        try:
            if os.path.isfile(self.last_loaded):
                self.connect_dists["forward"] = self.parse_dist(self.last_loaded)
        except BaseException:
            tb = traceback.format_exc()
            self.info_text.setText(
                "Failed to parse {} due to {}".format(self.last_loaded, tb)
            )

    def rConnect(self):
        self.selectFile()
        try:
            if os.path.isfile(self.last_loaded):
                self.connect_dists["recurrent"] = self.parse_dist(self.last_loaded)
        except BaseException:
            tb = traceback.format_exc()
            self.info_text.setText(
                "Failed to parse {} due to {}".format(self.last_loaded, tb)
            )

    def aConnect(self):
        self.selectFile()
        try:
            if os.path.isfile(self.last_loaded):
                self.connect_dists["localA"] = self.parse_dist(self.last_loaded)
        except BaseException:
            tb = traceback.format_exc()
            self.info_text.setText(
                "Failed to parse {} due to {}".format(self.last_loaded, tb)
            )

    def bConnect(self):
        self.selectFile()
        try:
            if os.path.isfile(self.last_loaded):
                self.connect_dists["localB"] = self.parse_dist(self.last_loaded)
        except BaseException:
            tb = traceback.format_exc()
            self.info_text.setText(
                "Failed to parse {} due to {}".format(self.last_loaded, tb)
            )

    def selectDir(self):
        if self.file_location is not None:
            default_dir = self.file_location
        else:
            default_dir = os.path.expanduser("~")
        file_location = self.file_dialog.getExistingDirectory(
            self.window, "Save results to directory", default_dir
        )
        if file_location == "":
            return
        self.file_location = file_location
        try:
            self.set_default_save_location()
        except BaseException:
            self.info_text.setText("Selected directory could not be parsed.")
            return

    def selectFile(self):
        if self.file_location is not None:
            default_dir = self.file_location
        else:
            default_dir = os.path.expanduser("~")
        self.last_loaded, _filter = self.file_dialog.getOpenFileName(
            self.window, "Distribution to load", default_dir
        )
        if os.path.isfile(self.last_loaded):
            self.info_text.setText(
                "Loaded distribution from {}".format(self.last_loaded)
            )

    def parse_info(self):
        """Parse out the QtWidgets."""
        self.a_parsed = []
        for val in self.a_info:
            self.a_parsed.append(int(val.text()))
        self.b_parsed = []
        for val in self.b_info:
            self.b_parsed.append(int(val.text()))
        self.connection_parsed = {}
        for key, val in self.connections.items():
            self.connection_parsed[key] = []
            if key.startswith("local"):
                self.connection_parsed[key].append(float(val[0].text()) / 100)
                self.connection_parsed[key].append(float(val[1].text()) / 100)
            else:
                self.connection_parsed[key].append(int(val[0].text()))
                self.connection_parsed[key].append(int(val[1].text()))
            self.connection_parsed[key].append(self.connect_dists[key])

        self.params_parsed = {}
        for key, val in self.params.items():
            if key != "subsample_rate":
                self.params_parsed[key] = int(val.text())
            else:
                self.params_parsed[key] = float(val.text())
        for key, val in self.params_box.items():
            self.params_parsed[key] = val.isChecked()

    def get_default_save_location(self):
        with open(self.default_loc, "r") as f:
            self.file_location = f.read().strip()
        self.info_text.setText("Will save to {}".format(self.file_location))

    def set_default_save_location(self):
        with open(self.default_loc, "w") as f:
            f.write(self.file_location)
        self.info_text.setText("Will save to {}".format(self.file_location))

    def split_into_args(self):
        ns = SimpleNamespace()
        ns.num_cpus = 1
        ns.max_depth = self.params_parsed["max_depth"]
        ns.clt_start = self.params_parsed["clt_start"]
        ns.subsample_rate = self.params_parsed["subsample_rate"]
        ns.approx_hypergeo = self.params_parsed["approx_hypergeo"]
        ns.cfg = datetime.datetime.now().strftime("%d-%m-%Y--%H-%M-%S")

        cfg = {}
        cfg["default"] = {}
        cfg["default"]["region_sizes"] = [self.a_parsed[0], self.b_parsed[0]]
        cfg["default"]["num_samples"] = [self.a_parsed[2], self.b_parsed[2]]
        cfg["default"]["connectivity_param_names"] = [
            "num_senders",
            "min_inter",
            "max_inter",
            "min_forward",
            "max_forward",
        ]
        cfg["default"]["num_senders"] = [self.a_parsed[1], self.b_parsed[1]]
        cfg["default"]["min_forward"] = [
            self.connection_parsed["forward"][0],
            self.connection_parsed["recurrent"][0],
        ]

        cfg["default"]["max_forward"] = [
            self.connection_parsed["forward"][0],
            self.connection_parsed["recurrent"][1],
        ]

        cfg["default"]["min_inter"] = [
            self.connection_parsed["localA"][0],
            self.connection_parsed["localB"][0],
        ]

        cfg["default"]["max_inter"] = [
            self.connection_parsed["localA"][1],
            self.connection_parsed["localB"][1],
        ]

        cfg["default"]["num_iters"] = self.params_parsed["num_iters"]

        if self.params_parsed["do_mean"]:
            cfg["default"]["connectivity_pattern"] = "mean_connectivity"
        else:
            cfg["default"]["connectivity_pattern"] = "recurrent_connectivity"

        cfg["Setup"] = {}
        cfg["Setup"]["do_mpf"] = self.params_parsed["do_stats"]
        cfg["Setup"]["do_graph"] = self.params_parsed["do_graph"]
        cfg["Setup"]["do_nx"] = False
        cfg["Setup"]["do_vis_graph"] = False
        cfg["Setup"]["do_only_none"] = False
        cfg["Setup"]["gen_graph_each_iter"] = False
        cfg["Setup"]["save_dir"] = self.file_location
        cfg["Setup"]["use_full_region"] = True
        cfg["Stats"] = {}
        cfg["Stats"]["region_sub_params"] = {}

        return cfg, ns

    def start_main(self):
        self.parse_info()
        self.info_text.setText("Starting the program.")
        parsed, args = self.split_into_args()
        control_main(parsed, args, True)
        self.info_text.setText("Finished running the program.")

    def __str__(self):
        s = "{}, {}\n{}\n{}".format(
            self.a_parsed, self.b_parsed, self.connection_parsed, self.params_parsed
        )
        return s


def main():
    here = os.path.dirname(os.path.realpath(__file__))
    ui_path = os.path.join(here, "assets", "connection.ui")
    ui = CodeTimeUI(ui_path)
    ui.start()
    sys.exit()


if __name__ == "__main__":
    main()
