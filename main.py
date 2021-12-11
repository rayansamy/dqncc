import argparse
import utils
from model import ForGAN
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
from datetime import datetime

import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt



pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

RootDir = "./archive"
History = 60

def read_data ():
    coin_no = 0
    for name in os.listdir(RootDir):
        coin_no += 1
    
    max_length, min_length = 0, 1000000
    for name in os.listdir(RootDir):
        df = pd.read_csv(RootDir + "/" + name, parse_dates=['Date'])
        length = df.shape[0]
        if max_length < length:
            max_length = length
        if min_length > length:
            min_length = length
    
    data = np.zeros ((coin_no, max_length))
    lengths = np.zeros(coin_no, dtype = int)
    i = 0
    for name in os.listdir(RootDir):
        short_name = name[5:-4]
        df = pd.read_csv(RootDir + "/" + name, parse_dates=['Date'])
        length = df.shape[0]
        lengths[i] = length
        print (i, short_name, length)
        data[i, 0:length] = df['Close'].values  # We only keep the closing price as a sequence!
        i += 1
    
    return coin_no, lengths, data

coin_no, lengths, data = read_data ()
print ("Got", coin_no, "coins.")

def scale_data (data, lengths):
    coin_no = data.shape[0]
    shift = np.zeros (coin_no)
    factor = np.zeros (coin_no)
    for i in range (coin_no):
        max_val = data[i,:lengths[i]].max()
        min_val = data[i, :lengths[i]].min()
        shift[i] = min_val
        factor[i] = max_val - min_val
        data[i,0:lengths[i]] = (data[i,0:lengths[i]]-shift[i])/factor[i]
    return (shift, factor)
    
shift, factor = scale_data (data, lengths)

def create_sequences (data, lengths, start, end):
    x = []
    y = []
    for i in range (start, end):   # Go only over the specified coins
        for j in range(History, lengths[i]):
            x.append(data[i, j-History:j])
            y.append(data[i, j])
    return np.array(x)[:, :, np.newaxis], np.array(y)

x_train, y_train = create_sequences(data, lengths, 0, 18)
print ("Got", y_train.shape[0], "training sequenes.")
x_val, y_val = create_sequences(data, lengths, 18, 22)
print ("Got", y_val.shape[0], "validation sequenes.")
x_test, y_test = create_sequences(data, lengths, 22, 23)
print ("Got", y_test.shape[0], "test sequenes.")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # mg for Mackey Glass and itd = Internet traffic dataset (A5M)
    ap.add_argument("-ds", metavar='', dest="dataset", type=str, default="lorenz",
                    help="The name of dataset: lorenz or mg or itd")
    ap.add_argument("-t", metavar='', dest="cell_type", type=str, default="gru",
                    help="The type of cells : lstm or gru")
    ap.add_argument("-steps", metavar='', dest="n_steps", type=int, default=10000,
                    help="Number of steps for training")
    ap.add_argument("-bs", metavar='', dest="batch_size", type=int, default=1000,
                    help="Batch size")
    ap.add_argument("-lr", metavar='', dest="lr", type=float, default=0.001,
                    help="Learning rate for RMSprop optimizer")
    ap.add_argument("-n", metavar='', dest="noise_size", type=int, default=32,
                    help="The size of Noise of Vector")
    ap.add_argument("-c", metavar='', dest="condition_size", type=int, default=24,
                    help="The size of look-back window ( Condition )")
    ap.add_argument("-rg", metavar='', dest="generator_latent_size", type=int, default=8,
                    help="The number of cells in generator")
    ap.add_argument("-rd", metavar='', dest="discriminator_latent_size", type=int, default=64,
                    help="The number of cells in discriminator")
    ap.add_argument("-d_iter", metavar='', dest="d_iter", type=int, default=2,
                    help="Number of training iteration for discriminator")
    ap.add_argument("-hbin", metavar='', dest="hist_bins", type=int, default=80,
                    help="Number of histogram bins for calculating KLD")
    ap.add_argument("-hmin", metavar='', dest="hist_min", type=float, default=-11,
                    help="Min range of histogram for calculating KLD")
    ap.add_argument("-hmax", metavar='', dest="hist_max", type=float, default=11,
                    help="Max range of histogram for calculating KLD")

    opt = ap.parse_args()

    for dirname, _, filenames in os.walk('./archive/'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    
    opt.data_mean = x_train.mean()
    opt.data_std = x_train.std()
    x_train = x_train[:, :, 0]
    print("TYPPPPEEE "+str(type(x_train)))
    print(x_train.shape)
    print(y_train.shape)
    forgan = ForGAN(opt)
    forgan.train(x_train, y_train, x_val, y_val)
    forgan.test(x_test, y_test)
