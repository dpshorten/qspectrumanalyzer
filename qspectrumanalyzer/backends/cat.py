import struct, shlex, sys, time
import time

from keras.layers import LSTM, Dense, Activation, TimeDistributed, Flatten, BatchNormalization, LocallyConnected1D, MaxPooling1D, AveragePooling1D, Dropout, Conv1D
from keras.models import Sequential
from keras import backend as K
import tensorflow as tf

import numpy as np
from Qt import QtCore

from qspectrumanalyzer import subproc
from qspectrumanalyzer.backends import BaseInfo, BasePowerThread

from sklearn import preprocessing
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
        (low_edge, high_edge) = (2400000000, 2900000000)
#        print(buf)
#        print(type(buf))
        #VIstring = ','.join(['%.5f' % num for num in buf])
        data = np.fromstring(buf, sep=',')
        data = data[12:-3]
        data = data[0::2]
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
        if (t_finish < self.lastsweep + self.interval):
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
        """hackrf_sweep thread main loop"""
        self.process_start()
        self.alive = True
        self.powerThreadStarted.emit()

        f = open("../Data Collection 20 Feb__20180220-110012-20180220-110000Z.dat-Network_100_Aoa_102_Node_101_Vector.csv", "r")

        # model = Sequential()

        # model.add(TimeDistributed(Conv1D(32, 5, strides = 1, activation = 'relu'), input_shape = (20, 1280, 1)))
        # model.add(TimeDistributed(AveragePooling1D(4, strides = 4)))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.5))

        # model.add(TimeDistributed(Conv1D(32, 5, strides = 1, activation = 'relu')))
        # model.add(TimeDistributed(AveragePooling1D(4, strides = 4)))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.5))

        # model.add(TimeDistributed(LocallyConnected1D(40, 5, strides = 1, activation = 'relu')))
        # model.add(TimeDistributed(MaxPooling1D(3, strides = 2)))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.5))

        # model.add(TimeDistributed(LocallyConnected1D(128, 5, strides = 1, activation = 'relu')))
        # model.add(TimeDistributed(MaxPooling1D(3, strides = 2)))
        # model.add(BatchNormalization())
        # #model.add(Dropout(0.25))                                                                                                                                               

        # model.add(TimeDistributed(Flatten()))

        # model.add(LSTM(256, dropout = 0.5, return_sequences = True, stateful = False))
        # model.add(BatchNormalization())

        # #model.add(LSTM(512, dropout = 0.5, return_sequences = True, stateful = False))                                                                                         
        # #model.add(BatchNormalization())                                                                                                                                       

        # model.add(LSTM(256, return_sequences = False, dropout = 0.5, recurrent_dropout = 0.0, stateful = False))
        # model.add(BatchNormalization())

        # #model.add(Dropout(0.5))                                                                                                                                                
        # #model.add(Dense(64, activation = 'relu'))                                                                                                                              
        # #model.add(BatchNormalization())                                                                                                                                        
        # model.add(Dropout(0.5))

        # model.add(Dense(1, activation='sigmoid'))


        # fi = open("scaler.pkl", "rb")
        # scaler = pickle.load(fi)
        
        # model.load_weights('avpool-conv2-weights.99-0.98.hdf5')

        # tf_session = K.get_session()

 #       with tf_session as sess:
  #          tf.train.Saver(tf.trainable_variables()).save(sess, './tf_model')

        for i in range(300000):
            f.readline()
  
        i = 0
        #storage = np.empty((1280,))
        storage = []
        while self.alive:
            #time.sleep(0.001)
            
            buf = f.readline()
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

            if i % 20 == 0:
                self.parse_output(buf)

            if i % 1000 == 0:
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(int(buf.split(',')[0]))))
                print(i)

        self.process_stop()
        self.alive = False
        self.powerThreadStopped.emit()
