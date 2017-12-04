import struct, shlex, sys, time

import numpy as np
from Qt import QtCore

from qspectrumanalyzer import subproc
from qspectrumanalyzer.backends import BaseInfo, BasePowerThread


class Info(BaseInfo):
    """hackrf_sweep device metadata"""
    sample_rate_min = 20000000
    sample_rate_max = 20000000
    sample_rate = 20000000
    gain_min = -1
    gain_max = 102
    gain = 40
    start_freq_min = 0
    start_freq_max = 7230
    start_freq = 0
    stop_freq_min = 0
    stop_freq_max = 7250
    stop_freq = 6000
    bin_size_min = 3
    bin_size_max = 5000
    bin_size = 1000
    interval = 0
    ppm_min = 0
    ppm_max = 0
    ppm = 0
    crop_min = 0
    crop_max = 0
    crop = 0


class PowerThread(BasePowerThread):
    """Thread which runs hackrf_sweep process"""
    def setup(self, start_freq=0, stop_freq=6000, bin_size=1000,
              interval=0.0, gain=40, ppm=0, crop=0, single_shot=False,
              device=0, sample_rate=20000000, bandwidth=0, lnb_lo=0):
        """Setup hackrf_sweep params"""
        # Small bin sizes (<40 kHz) are only suitable with an arbitrarily
        # reduced sweep interval. Bin sizes smaller than 3 kHz showed to be
        # infeasible also in these cases.
        if bin_size < 3:
            bin_size = 3
        if bin_size > 5000:
            bin_size = 5000

        # We only support whole numbers of steps with bandwidth equal to the
        # sample rate.
        step_bandwidth = sample_rate / 1000000
        total_bandwidth = stop_freq - start_freq
        step_count = 1 + (total_bandwidth - 1) // step_bandwidth
        total_bandwidth = step_count * step_bandwidth
        stop_freq = start_freq + total_bandwidth

        # distribute gain between two analog gain stages
        if gain > 102:
            gain = 102
        lna_gain = 8 * (gain // 18) if gain >= 0 else 0
        vga_gain = 2 * ((gain - lna_gain) // 2) if gain >= 0 else 0

        self.params = {
            "start_freq": start_freq,  # MHz
            "stop_freq": stop_freq,  # MHz
            "hops": 0,
            "device": 0,
            "sample_rate": 20e6,  # sps
            "bin_size": bin_size,  # kHz
            "interval": interval,  # seconds
            "gain": gain,
            "lna_gain": lna_gain,
            "vga_gain": vga_gain,
            "ppm": 0,
            "crop": 0,
            "single_shot": single_shot
        }
        self.lnb_lo = lnb_lo
        self.databuffer = {"timestamp": [], "x": [], "y": []}
        self.lastsweep = 0
        self.interval = interval

    def process_start(self):
        """Start hackrf_sweep process"""
        if not self.process and self.params:
            print("foo")
            settings = QtCore.QSettings()
            cmdline = shlex.split(settings.value("executable", "more"))
            cmdline.extend([
                '/home/david/metis_data/foo.csv'
            ])


            print('Starting backend:')
            print(' '.join(cmdline))
            print()
            self.process = subproc.Popen(cmdline, stdout=subproc.PIPE,
                                            universal_newlines=False, console=False)

    def parse_output(self, buf):
        """Parse one buf of output from hackrf_sweep"""
        (low_edge, high_edge) = (2400, 2420)
#        print(buf)
#        print(type(buf))
        #VIstring = ','.join(['%.5f' % num for num in buf])
        data = np.fromstring(buf, sep=',')
        data = data[10:]
        print(len(data))
        step = (high_edge - low_edge) / len(data)

        #if (low_edge // 1000000) <= (self.params["start_freq"] - self.lnb_lo / 1e6):
            # Reset databuffer at the start of each sweep even if we somehow
            # did not complete the previous sweep.
        self.databuffer = {"timestamp": [], "x": [], "y": []}
        x_axis = list(np.arange(low_edge + self.lnb_lo + step / 2, high_edge + self.lnb_lo, step))
        self.databuffer["x"].extend(x_axis)
        for i in range(len(data)):
            self.databuffer["y"].append(data[i])
#        if (high_edge / 1e6) >= (self.params["stop_freq"] - self.lnb_lo / 1e6):
            # We've reached the end of a pass. If it went too fast for our sweep interval, ignore it
        t_finish = time.time()
        if (t_finish < self.lastsweep + self.interval):
            return
        self.lastsweep = t_finish

        # otherwise sort and display the data.
        sorted_data = sorted(zip(self.databuffer["x"], self.databuffer["y"]))
        self.databuffer["x"], self.databuffer["y"] = [list(x) for x in zip(*sorted_data)]
        #print(self.databuffer["y"][-5:])
        self.data_storage.update(self.databuffer)

    def run(self):
        """hackrf_sweep thread main loop"""
        self.process_start()
        self.alive = True
        self.powerThreadStarted.emit()

        f = open("/home/david/metis_data/foo.csv", "r")
        
        while self.alive:
            time.sleep(0.1)
            buf = f.readline()
            self.parse_output(buf)

        self.process_stop()
        self.alive = False
        self.powerThreadStopped.emit()
