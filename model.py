import os

import numpy as np
import torch
from torch import nn

import utils

# Fixing random seeds
torch.manual_seed(1368)
rs = np.random.RandomState(1368)
YELLOW_TEXT = '\033[93m'
ENDC = '\033[0m'
BOLD = '\033[1m'


class Generator(nn.Module):
    def __init__(self, noise_size, condition_size, generator_latent_size, cell_type, mean=0, std=1):
        super().__init__()

        self.noise_size = noise_size
        self.condition_size = condition_size
        self.generator_latent_size = generator_latent_size
        self.mean = mean
        self.std = std

        if cell_type == "lstm":
            self.cond_to_latent = nn.LSTM(input_size=1,
                                          hidden_size=generator_latent_size)
        else:
            self.cond_to_latent = nn.GRU(input_size=1,
                                         hidden_size=generator_latent_size)

        self.model = nn.Sequential(
            nn.Linear(in_features=generator_latent_size + self.noise_size,
                      out_features=generator_latent_size + self.noise_size),
            nn.ReLU(),
            nn.Linear(in_features=generator_latent_size + self.noise_size, out_features=1)

        )

    def forward(self, noise, condition):
        condition = (condition - self.mean) / self.std
        condition = condition.view(-1, self.condition_size, 1)
        condition = condition.transpose(0, 1)
        condition_latent, _ = self.cond_to_latent(condition)
        condition_latent = condition_latent[-1]
        g_input = torch.cat((condition_latent, noise), dim=1)
        output = self.model(g_input)
        output = output * self.std + self.mean

        return output

    def get_noise_size(self):
        return self.noise_size


class Discriminator(nn.Module):
    def __init__(self, condition_size, discriminator_latent_size, cell_type, mean=0, std=1):
        super().__init__()
        self.discriminator_latent_size = discriminator_latent_size
        self.condition_size = condition_size
        self.mean = mean
        self.std = std

        if cell_type == "lstm":
            self.input_to_latent = nn.LSTM(input_size=1,
                                           hidden_size=discriminator_latent_size)
        else:
            self.input_to_latent = nn.GRU(input_size=1,
                                          hidden_size=discriminator_latent_size)

        self.model = nn.Sequential(
            nn.Linear(in_features=discriminator_latent_size, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, prediction, condition):
        d_input = torch.cat((condition, prediction.view(-1, 1)), dim=1)
        d_input = (d_input - self.mean) / self.std
        d_input = d_input.view(-1, self.condition_size + 1, 1)
        d_input = d_input.transpose(0, 1)
        d_latent, _ = self.input_to_latent(d_input)
        d_latent = d_latent[-1]
        output = self.model(d_latent)
        return output

class ForGAN:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        print("*****  Hyper-parameters  *****")
        for k, v in vars(opt).items():
            print("{}:\t{}".format(k, v))
        print("************************")


        # Making required directories for logging, plots and models' checkpoints
        os.makedirs("./{}/".format(self.opt.dataset), exist_ok=True)

        # Defining GAN components
        self.generator = Generator(noise_size=opt.noise_size,
                                   condition_size=opt.condition_size,
                                   generator_latent_size=opt.generator_latent_size,
                                   cell_type=opt.cell_type,
                                   mean=opt.data_mean,
                                   std=opt.data_std)

        self.discriminator = Discriminator(condition_size=opt.condition_size,
                                           discriminator_latent_size=opt.discriminator_latent_size,
                                           cell_type=opt.cell_type,
                                           mean=opt.data_mean,
                                           std=opt.data_std)

        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        print("\nNetwork Architecture\n")
        print(self.generator)
        print(self.discriminator)
        print("\n************************\n")

    def train(self, x_train, y_train, x_val, y_val):
        x_train = torch.tensor(x_train, device=self.device, dtype=torch.float32)
        y_train = torch.tensor(y_train, device=self.device, dtype=torch.float32)
        x_val = torch.tensor(x_val, device=self.device, dtype=torch.float32)
        best_kld = np.inf
        optimizer_g = torch.optim.RMSprop(self.generator.parameters(), lr=self.opt.lr)
        optimizer_d = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.opt.lr)
        adversarial_loss = nn.BCELoss()
        adversarial_loss = adversarial_loss.to(self.device)

        for step in range(self.opt.n_steps):
            d_loss = 0
            for _ in range(self.opt.d_iter):
                # train discriminator on real data
                idx = rs.choice(x_train.shape[0], self.opt.batch_size)
                condition = x_train[idx]
                real_data = y_train[idx]
                self.discriminator.zero_grad()
                d_real_decision = self.discriminator(real_data, condition)
                d_real_loss = adversarial_loss(d_real_decision,
                                               torch.full_like(d_real_decision, 1, device=self.device))
                d_real_loss.backward()
                d_loss += d_real_loss.detach().cpu().numpy()
                # train discriminator on fake data
                noise_batch = torch.tensor(rs.normal(0, 1, (condition.size(0), self.opt.noise_size)),
                                           device=self.device, dtype=torch.float32)
                x_fake = self.generator(noise_batch, condition).detach()
                d_fake_decision = self.discriminator(x_fake, condition)
                d_fake_loss = adversarial_loss(d_fake_decision,
                                               torch.full_like(d_fake_decision, 0, device=self.device))
                d_fake_loss.backward()

                optimizer_d.step()
                d_loss += d_fake_loss.detach().cpu().numpy()

            d_loss = d_loss / (2 * self.opt.d_iter)

            self.generator.zero_grad()
            noise_batch = torch.tensor(rs.normal(0, 1, (self.opt.batch_size, self.opt.noise_size)), device=self.device,
                                       dtype=torch.float32)
            x_fake = self.generator(noise_batch, condition)
            d_g_decision = self.discriminator(x_fake, condition)
            # Mackey-Glass works best with Minmax loss in our expriements while other dataset
            # produce their best result with non-saturated loss
            if opt.dataset == "mg":
                g_loss = adversarial_loss(d_g_decision, torch.full_like(d_g_decision, 1, device=self.device))
            else:
                g_loss = -1 * adversarial_loss(d_g_decision, torch.full_like(d_g_decision, 0, device=self.device))
            g_loss.backward()
            optimizer_g.step()

            g_loss = g_loss.detach().cpu().numpy()

            # Validation
            noise_batch = torch.tensor(rs.normal(0, 1, (x_val.size(0), self.opt.noise_size)), device=self.device,
                                       dtype=torch.float32)
            preds = self.generator(noise_batch, x_val).detach().cpu().numpy().flatten()

            kld = utils.calc_kld(preds, y_val, self.opt.hist_bins, self.opt.hist_min, self.opt.hist_max)

            if kld <= best_kld and kld != np.inf:
                best_kld = kld
                print("step : {} , KLD : {}, RMSE : {}".format(step, best_kld,
                                                               np.sqrt(np.square(preds - y_val).mean())))
                torch.save({
                    'g_state_dict': self.generator.state_dict()
                }, "./{}/best.torch".format(self.opt.dataset))

            if step % 100 == 0:
                print(YELLOW_TEXT + BOLD + "step : {} , d_loss : {} , g_loss : {}".format(step, d_loss, g_loss) + ENDC)

    def test(self, x_test, y_test):
        x_test = torch.tensor(x_test, device=self.device, dtype=torch.float32)
        checkpoint = torch.load("./{}/best.torch".format(self.opt.dataset))
        self.generator.load_state_dict(checkpoint['g_state_dict'])
        y_test = y_test.flatten()
        preds = []
        rmses = []
        maes = []
        mapes = []

        for _ in range(200):
            noise_batch = torch.tensor(rs.normal(0, 1, (x_test.size(0), self.opt.noise_size)), device=self.device,
                                       dtype=torch.float32)
            pred = self.generator(noise_batch, x_test).detach().cpu().numpy().flatten()
            preds.append(pred)

            error = pred - y_test
            rmses.append(np.sqrt(np.square(error).mean()))
            maes.append(np.abs(error).mean())
            mapes.append(np.abs(error / y_test).mean() * 100)
        preds = np.vstack(preds)
        crps = np.absolute(preds[:100] - y_test).mean() - 0.5 * np.absolute(preds[:100] - preds[100:]).mean()
        preds = preds.flatten()
        kld = utils.calc_kld(preds, y_test, self.opt.hist_bins, self.opt.hist_min, self.opt.hist_max)
        print("Test resuts:\nRMSE : {}({})\nMAE : {}({})\nMAPE : {}({}) %\nCRPS : {}\nKLD : {}\n"
              .format(np.mean(rmses), np.std(rmses),
                      np.mean(maes), np.std(maes),
                      np.mean(mapes), np.std(mapes),
                      crps,
                      kld))

