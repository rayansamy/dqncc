import argparse
import utils
from model import ForGAN


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

    x_train, y_train, x_val, y_val, x_test, y_test = utils.prepare_dataset(opt.dataset, opt.condition_size)
    opt.data_mean = x_train.mean()
    opt.data_std = x_train.std()
    forgan = ForGAN(opt)
    forgan.train(x_train, y_train, x_val, y_val)
    forgan.test(x_test, y_test)
