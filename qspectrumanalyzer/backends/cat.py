import struct, shlex, sys, time
import time

import scipy.signal as sig

#from keras.layers import LSTM, Dense, Activation, TimeDistributed, Flatten, BatchNormalization, LocallyConnected1D, MaxPooling1D, AveragePooling1D, Dropout, Conv1D
#from keras.models import Sequential
#from keras import backend as K
import tensorflow as tf

import numpy as np
from Qt import QtCore

from qspectrumanalyzer import subproc
from qspectrumanalyzer.backends import BaseInfo, BasePowerThread

#from sklearn import preprocessing
import pickle

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
            #cmdline.extend([
             #   '/home/david/metis_data/foo.csv'
            #])


            #print('Starting backend:')
            #print(' '.join(cmdline))
            print()
            self.process = subproc.Popen(cmdline, stdout=subproc.PIPE,
                                            universal_newlines=False, console=False)

    def parse_output(self, buf, buf_bg, buf_history, buf_bg_history):
        """Parse one buf of output from hackrf_sweep"""
        (low_edge, high_edge) = (2400000000, 2900000000)
#        print(buf)
#        print(type(buf))
        #VIstring = ','.join(['%.5f' % num for num in buf])
        
        # np_buf_history = np.zeros((200, 3248), dtype=np.float32)
        # for i in range(len(buf_history)):
        #     np_buf_history[i, :] = np.fromstring(buf_history[i], sep=',')[6:]

        # np_buf_history = sig.convolve2d(np_buf_history, np.ones((5, 5))/25, mode = 'same')

        # np_buf_bg_history = np.zeros((200, 3248), dtype=np.float32)
        # for i in range(len(buf_bg_history)):
        #     np_buf_bg_history[i, :] = np.fromstring(buf_bg_history[i], sep=',')[6:]

        # np_buf_bg_history = sig.convolve2d(np_buf_bg_history, np.ones((5, 5))/25, mode = 'same')

        # np_buf_history = np.reshape(np_buf_history, (200 * 3248,))
        # np_buf_history = np.sort(np_buf_history)
        # np_buf_bg_history = np.reshape(np_buf_bg_history, (200 * 3248,))
        # np_buf_bg_history = np.sort(np_buf_bg_history)

        # SIG_C = np.mean(np_buf_bg_history[1000:200000]) - np.mean(np_buf_history[-20000:-5000]) + 3

        # print(SIG_C, np.mean(np_buf_bg_history[1000:200000]), np.mean(np_buf_history[-20000:-5000]))



        
        data = np.fromstring(buf, sep=',')
        data = data[12:-3]


        
        data = np.convolve(data[:], np.ones(11)/float(11), mode = 'valid')
        
        #low_values_flags = data < -120  # Where values are low
        #data[low_values_flags] = -130        

        #data_bg = np.fromstring(buf_bg, sep=',')
        #data_bg = data_bg[12:-3]

        #BG_C = 0.0


        #data = (10 * np.log10(np.power(10, ((BG_C + data_bg)/10)) + np.power(10, ((SIG_C + data)/10))))

        
        
        # #print(data[:5])
        # data[:] += 122.0
        # data[:] /=2
        # #data[:] += 0.1
        # data[:] +=  0.1 * np.power(data, 2)
        # data[:] *=2
        # data[:] -= 122.0
        #print(data[:5])
        #exit()
        #print("data", data[-5:])
        # for i in range(len(data)):
        #     if data[i] < -110:
        #         data[i] = -130
#        print(len(data))
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
        if (t_finish < self.lastsweep + 0.0):
            return
        self.lastsweep = t_finish

        # otherwise sort and display the data.
        sorted_data = sorted(zip(self.databuffer["x"], self.databuffer["y"]))
        self.databuffer["x"], self.databuffer["y"] = [list(x) for x in zip(*sorted_data)]
#        print(self.databuffer["y"][-5:])
        #for i in range(10):
         #   self.databuffer["y"][-i] = 0
        self.data_storage.update(self.databuffer)

    def run(self):
        """hackrf_swep thread main loop"""
        self.process_start()
        self.alive = True
        self.powerThreadStarted.emit()

        # = open("/media/david/Wabbit/foo/Data Collect - 18 May__20180518-155159_Network100_Node101_Data_1.ndf-Network_0_Sweep_101_Node_0.csv", "r")
        #f = open("/media/david/Wabbit/Collection 20 Feb 18/Converted files/Data Collection 20 Feb__20180220-110012-20180220-110000Z.dat-Network_100_Sweep_105_Node_101.csv", buffering = 1000)
        #f = open("../Data Collection 20 Feb__20180220-110012-20180220-110000Z.dat-Network_101_Sweep_100_Node_100.csv", "r")
        #f = open("../Data Collection 20 Feb__20180220-140631-20180220-140000Z.dat-Network_100_Sweep_105_Node_101.csv", "r")
        #f = open("../MetisOfficeAmbient__20180619-165624-Network_100_Sweep_100_Node_100.csv", "r")
        #f = open("../Data Collection 20 Feb__20180220-134503-20180220-130000Z.dat-Network_100_Sweep_105_Node_101.csv", "r")
        #f = open("../FaintDroneAndControllerValidationSet__20180709-110436-Network_100_Sweep_100_Node_100_downsampled.csv", "r")
        f = open("../FaintDroneAndControllerValidationSet__20180709-110436-Network_100_Sweep_100_Node_100_downsampled.csv", "r")
        #f = open("../Data Collect - 18 May__20180620-115903-Network_100_Sweep_101_Node_101.csv", "r")
        #f = open("../Data Collection 20 Feb__20180220-140631-20180220-140000Z.dat-Network_100_Sweep_105_Node_101.csv", "r")
        # model = Sequential()

        #f_bg = open("../Untitled__20180626-135606_Network100_Node100_Data_1.ndf-Network_0_Sweep_100_Node_0_d.csv", "r")
        
        i = 0
        #storage = np.empty((1280,))
        storage = []
        buf = f.readline()
        buf = f.readline()
        #buf_bg = f_bg.readline()
        buf_bg = f.readline()
        #for k in range(int(5e5)):
        #for k in range(int(2.6e4)):
         #   buf = f.readline()

        #for k in range(int(3.7e5)):
        for k in range(int(4.7e4)):
            buf = f.readline()

        buf_history = []
        buf_bg_history = []
        for k in range(200):
            buf_history.append(f.readline())
            buf_bg_history.append(f.readline())
            
        prev_time = int(buf.split(',')[0])
        freq_tracker = 0
        while self.alive:
            buf = f.readline()
            #buf_bg = f_bg.readline()

            #buf_history.append(buf[:])
            #buf_history.pop(0)
            #buf_bg_history.append(buf_bg[:])
            #buf_bg_history.pop(0)
            
            time_now = int(buf.split(',')[0])
            if time_now > prev_time:
                prev_time = time_now
                #print("freq", freq_tracker)
                freq_tracker = 0
            else:
                freq_tracker += 1

            # if len(storage) == 0:
            #     storage = np.fromstring(buf, sep=',')[6:]
            # else:
            #     storage = np.vstack((storage, np.fromstring(buf, sep=',')[6:]))

            i += 1
#            if i % 20 == 0:
            #     #print(storage[0, :10])
            #     storage = scaler.transform(storage)
            #     #print(storage[0, :10])
            #     storage = np.expand_dims(storage, 0)
            #     storage = np.expand_dims(storage, 3)
            #     val = model.predict(storage)[0][0]
            #     print(val)
            #     if val > 0.5:
            #         print("DRONE!!!!!")
            #     storage = []
            #     #model.reset_states()

            if i % 1 == 0:
                #buf_bg = f_bg.readline()
                self.parse_output(buf, buf_bg, buf_history, buf_bg_history)
                time.sleep(0.01)

            if i % 5 == 0:
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(int(buf.split(',')[0]))))
                print(i)

        self.process_stop()
        self.alive = False
        self.powerThreadStopped.emit()
