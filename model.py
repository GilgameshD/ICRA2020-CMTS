import torch.nn as nn
import torch
from torch.autograd import Variable
from utils import CUDA
import numpy as np


class AdaIN(nn.Module):
    def forward(self, x, y_mean, y_std):
        x_mean = x.mean(dim=1, keepdim=True)
        x_std  = x.std(dim=1, keepdim=True)
        x_mean = x_mean.expand_as(x)
        x_std = x_std.expand_as(x)
        out = y_std * (x - x_mean) / x_std + y_mean
        #out = y_std * x + y_mean
        return out


class AdaIN_VAE(nn.Module):
    def __init__(self, z_dim):
        super(AdaIN_VAE, self).__init__()
        self.z_dim = z_dim

        self.input_size = 64
        self.num_layers = 1
        self.hidden_size_encoder = 128
        self.hidden_size_decoder = 128

        self.embedding_encoder = nn.Linear(4, self.input_size)
        self.z2hidden = nn.Linear(self.z_dim, self.hidden_size_decoder)
        self.encoder = nn.Linear(2*self.hidden_size_encoder, z_dim*2)

        self.decoder_1 = nn.Linear(self.hidden_size_encoder, 2)
        self.decoder_2 = nn.Linear(self.hidden_size_encoder, 2)
        
        self.embedding_decoder_1 = nn.Linear(2, int(self.input_size/2))
        self.embedding_decoder_2 = nn.Linear(2, int(self.input_size/2))

        self.hidden2input_1 = nn.Linear(self.hidden_size_encoder, int(self.input_size/2))
        self.hidden2input_2 = nn.Linear(self.hidden_size_encoder, int(self.input_size/2))

        # gru processing module
        self.encoder_lstm = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size_encoder, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.decoder_lstm_1 = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size_decoder, num_layers=self.num_layers, batch_first=True)
        self.decoder_lstm_2 = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size_decoder, num_layers=self.num_layers, batch_first=True)

        # image condition processing module
        self.conditional_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conditional_mean = nn.Sequential(
            nn.Linear(64*8*8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.z_dim)
        )
        
        self.conditional_logvar = nn.Sequential(
            nn.Linear(64*8*8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.z_dim)
        )

        # combine style and content
        # use image condition as the content, use latent code z as the style
        self.adain = AdaIN()

    def init_hidden(self, batch_size, hidden_size):
        hidden_h = Variable(torch.zeros(self.num_layers*2, batch_size, hidden_size))
        if torch.cuda.is_available():
            hidden_h = hidden_h.cuda()
        return hidden_h

    def condition_forward_(self, c):
        c = self.conditional_conv(c).view(-1, 8*8*64)
        c_mean = self.conditional_mean(c)
        c_std = self.conditional_logvar(c).div(2).exp()
        return c_mean, c_std

    def encoder_forward_(self, x):
        batch_size = x.size(0)
        hidden_encoder = self.init_hidden(batch_size, self.hidden_size_encoder)
        x_embedding = self.embedding_encoder(x)
        output, hidden_encoder = self.encoder_lstm(x_embedding, hidden_encoder)

        # get the last output
        last_output = self.encoder(output[:, -1, :])

        # devide the two rows into mu and logvar
        mu = last_output[:, :self.z_dim]
        logvar = last_output[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        return z, [mu, logvar]

    def decoder_forward_(self, z, start):
        batch_size = z.size(0)

        # use the start point of data
        start_point_1 = start[:, 0:1, 0:2]
        start_point_2 = start[:, 0:1, 2:4]

        hidden_decoder_1 = self.z2hidden(z.view(batch_size, 1, self.z_dim)).permute(1, 0, 2)
        hidden_decoder_2 = self.z2hidden(z.view(batch_size, 1, self.z_dim)).permute(1, 0, 2)

        start_embedding = self.embedding_decoder_1(start_point_1)
        start_embedding_ = self.embedding_decoder_2(start_point_2)
        
        hidden_1 = self.hidden2input_1(hidden_decoder_1.permute(1, 0, 2))
        hidden_2 = self.hidden2input_2(hidden_decoder_2.permute(1, 0, 2))
        start_embedding_1 = torch.cat((start_embedding, hidden_2), dim=2)
        start_embedding_2 = torch.cat((start_embedding_, hidden_1), dim=2)

        x_recon = []
        for i_pos in range(50):
            x_recon_1, hidden_decoder_1 = self.decoder_lstm_1(start_embedding_1, hidden_decoder_1)
            x_recon_2, hidden_decoder_2 = self.decoder_lstm_2(start_embedding_2, hidden_decoder_2)

            x_recon_1 = x_recon_1[:, -1, :]
            x_recon_1 = self.decoder_1(x_recon_1).view(batch_size, -1, 2)
            x_recon_2 = x_recon_2[:, -1, :]
            x_recon_2 = self.decoder_2(x_recon_2).view(batch_size, -1, 2)

            x_recon_ = torch.cat((x_recon_1, x_recon_2), dim=2)
            x_recon.append(x_recon_.view(batch_size, -1))

            # wrap the input
            start_embedding = self.embedding_decoder_1(x_recon_1)
            start_embedding_ = self.embedding_decoder_2(x_recon_2)

            hidden_1 = self.hidden2input_1(hidden_decoder_1.permute(1, 0, 2))
            hidden_2 = self.hidden2input_2(hidden_decoder_2.permute(1, 0, 2))
            start_embedding_1 = torch.cat((start_embedding, hidden_2), dim=2)
            start_embedding_2 = torch.cat((start_embedding_, hidden_1), dim=2)

        x_recon = torch.stack(x_recon, dim=0).permute(1, 0, 2)
        return x_recon

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = CUDA(Variable(std.data.new(std.size()).normal_()))
        return mu+std*eps

    def forward(self, x_s, x_t, c_s, c_t):
        z_s, z_s_bag = self.encoder_forward_(x_s)
        z_t, z_t_bag = self.encoder_forward_(x_t)

        c_s_mean, c_s_std = self.condition_forward_(c_s)
        c_t_mean, c_t_std = self.condition_forward_(c_t)

        # weight generator
        batch_size = x_s.size(0)
        weight = CUDA(Variable(torch.FloatTensor(batch_size, self.z_dim)))
        # for each batch, use only one k
        weight_np = np.random.random([1, 1])
        weight_np = np.repeat(weight_np, batch_size, axis=0) 
        weight_np = np.repeat(weight_np, self.z_dim, axis=1)
        weight.data.copy_(torch.from_numpy(weight_np))

        # combine z_f and conditions
        z_f = weight*z_s + (1-weight)*z_t
        z_f_s = self.adain(z_f, c_s_mean, c_s_std)
        z_f_t = self.adain(z_f, c_t_mean, c_t_std)
        z_c_s = self.adain(z_s, c_s_mean, c_s_std)
        z_c_t = self.adain(z_t, c_t_mean, c_t_std)

        # decoder modules
        x_s_ = self.decoder_forward_(z_c_s, x_s)
        x_t_ = self.decoder_forward_(z_c_t, x_t)
        x_f_s_ = self.decoder_forward_(z_f_s, x_s)
        x_f_t_ = self.decoder_forward_(z_f_t, x_t)

        # additional encoder
        _, z_s_bag_new = self.encoder_forward_(x_s_)
        _, z_t_bag_new = self.encoder_forward_(x_t_)
        _, z_bag_f_s = self.encoder_forward_(x_f_s_)
        _, z_bag_f_t = self.encoder_forward_(x_f_t_)

        return x_s_, x_t_, z_s_bag, z_t_bag, [z_f_s, z_bag_f_s[0], z_f_t, z_bag_f_t[0], z_c_s, z_s_bag_new[0], z_c_t, z_t_bag_new[0]]

    def test_forward(self, x_s, x_t, c_s, c_t):
        # use one sample or the average of all batch ?
        # only use the first one
        x_s_test = x_s[0:10].repeat(10, 1, 1)
        x_t_test = x_t[0:10].repeat(10, 1, 1)

        z_s, z_s_bag = self.encoder_forward_(x_s_test)
        z_t, z_t_bag = self.encoder_forward_(x_t_test)

        # the first 10 sample has the same weights and different conditions
        weight_np = np.linspace(0, 1, 10).reshape(10, 1)
        weight_np = np.repeat(weight_np, 10, axis=0)
        weight_np = np.repeat(weight_np, self.z_dim, axis=1)
        weight = CUDA(Variable(torch.FloatTensor(100, self.z_dim)))
        weight.data.copy_(torch.from_numpy(weight_np))
        z_f_test = weight*z_s + (1-weight)*z_t

        c_s_test = c_s[0:10].repeat(10, 1, 1, 1)
        c_t_test = c_t[0:10].repeat(10, 1, 1, 1)
        c_s_mean, c_s_std = self.condition_forward_(c_s_test)
        c_t_mean, c_t_std = self.condition_forward_(c_t_test)
        z_c_s = self.adain(z_f_test, c_s_mean, c_s_std)
        z_c_t = self.adain(z_f_test, c_t_mean, c_t_std)
        x_f_s = self.decoder_forward_(z_c_s, x_s_test).detach()
        x_f_t = self.decoder_forward_(z_c_t, x_t_test).detach()

        return x_f_s, x_f_t, z_s, z_t, z_f_test

    def test_forward_100(self, x_s, x_t, c_s, c_t, num=100):
        # use one sample or the average of all batch ?
        # only use the first one
        x_s_test = x_s[0:1].repeat(num, 1, 1)
        x_t_test = x_t[0:1].repeat(num, 1, 1)

        z_s, z_s_bag = self.encoder_forward_(x_s_test)
        z_t, z_t_bag = self.encoder_forward_(x_t_test)

        # the first 10 sample has the same weights and different conditions
        weight_np = np.linspace(0, 1, num).reshape(num, 1)
        weight_np = np.repeat(weight_np, self.z_dim, axis=1)
        weight = CUDA(Variable(torch.FloatTensor(num, self.z_dim)))
        weight.data.copy_(torch.from_numpy(weight_np))
        z_f_test = weight*z_s + (1-weight)*z_t

        c_s_test = c_s[0:1].repeat(num, 1, 1, 1)
        c_t_test = c_t[0:1].repeat(num, 1, 1, 1)
        c_s_mean, c_s_std = self.condition_forward_(c_s_test)
        c_t_mean, c_t_std = self.condition_forward_(c_t_test)

        z_c_s = self.adain(z_f_test, c_s_mean, c_s_std)
        z_c_t = self.adain(z_f_test, c_t_mean, c_t_std)
        x_f_s = self.decoder_forward_(z_c_s, x_s_test).detach()
        x_f_t = self.decoder_forward_(z_c_t, x_t_test).detach()

        return x_f_s, x_f_t

    def generate_forward(self, x_s, x_t, c_s, c_t):
        z_s, z_s_bag = self.encoder_forward_(x_s)
        z_t, z_t_bag = self.encoder_forward_(x_t)

        c_s_mean, c_s_std = self.condition_forward_(c_s)
        c_t_mean, c_t_std = self.condition_forward_(c_t)

        # weight generator
        batch_size = x_s.size(0)
        weight = CUDA(Variable(torch.FloatTensor(batch_size, self.z_dim)))
        weight.data.fill_(0.3)
        z_f = weight*z_s + (1-weight)*z_t

        # combine z_f and conditions
        z_f_s = self.adain(z_f, c_s_mean, c_s_std)
        z_f_t = self.adain(z_f, c_t_mean, c_t_std)

        # decoder modules
        x_f_s_ = self.decoder_forward_(z_f_s, x_s)
        x_f_t_ = self.decoder_forward_(z_f_t, x_t)

        return x_f_s_, x_f_t_

    def encode(self, x_s, x_t, x_f):
        _, z_s_bag = self.encoder_forward_(x_s)
        _, z_t_bag = self.encoder_forward_(x_t)
        _, z_f_bag = self.encoder_forward_(x_f)

        return z_s_bag[0], z_t_bag[0], z_f_bag[0]
