import os
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import mkdirs, kaiming_init, CUDA, kld_loss, mse_loss
from model import AdaIN_VAE
from dataloader import Sequence_Dataset, DPGMM_Dataset


class Trainer(object):
    def __init__(self, args):
        # Misc
        self.continue_training = args.continue_training
        self.model_id = args.id
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.print_iter = args.print_iter
        self.save_epoch = args.save_epoch
        self.num_workers = args.num_workers
        self.num_sample = args.num_sample

        self.normal_data_path = args.normal_data_path
        self.normal_map_path = args.normal_map_path
        self.collision_data_path = args.collision_data_path
        self.collision_map_path = args.collision_map_path

        self.image_size = args.image_size
        self.z_dim = args.z_dim

        self.lr = args.lr
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma

        # Checkpoint
        self.model_path = args.model_path
        mkdirs(self.model_path)

        # Output (samples)
        self.sample_path = args.sample_path
        mkdirs(self.sample_path)

        self.extended_path = args.extended_path
        mkdirs(self.extended_path)

        # model
        self.ada_vae = CUDA(AdaIN_VAE(z_dim=self.z_dim))
        self.optimizer = optim.Adam(params=self.ada_vae.parameters(), lr=self.lr)
        if self.continue_training:
            self.load_model()
            print('[*] Finish loading parameters from file')
        else:
            self.ada_vae.apply(kaiming_init)
            print('[*] Finish building model')

        # data - [B, C, H, W]
        self.x_s = CUDA(Variable(torch.FloatTensor(self.batch_size, 50, 4)))
        self.x_t = CUDA(Variable(torch.FloatTensor(self.batch_size, 50, 4)))
        self.c_s = CUDA(Variable(torch.FloatTensor(self.batch_size, 1, self.image_size, self.image_size)))
        self.c_t = CUDA(Variable(torch.FloatTensor(self.batch_size, 1, self.image_size, self.image_size)))
        
        # load dataset
        self.dataset = Sequence_Dataset(self.normal_data_path, self.collision_data_path, self.normal_map_path, self.collision_map_path, self.image_size)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        print('[*] Finish loading dataset')

    def train(self):
        loss_collector = []
        pbar_epoch = tqdm(total=self.max_epoch, desc='[Epoch]')
        max_iteration = int(len(self.dataset)/self.batch_size)
        for epoch in range(self.max_epoch):
            pbar_iteration = tqdm(total=max_iteration, desc='[Iteration]')
            iteration = 0
            for data_x_s, data_x_t, data_c_s, data_c_t in self.dataloader:
                self.batch_size = data_x_s.size(0)

                self.x_s.resize_(self.batch_size, 50, 4)
                self.x_t.resize_(self.batch_size, 50, 4)
                self.c_s.resize_(self.batch_size, 1, self.image_size, self.image_size)
                self.c_t.resize_(self.batch_size, 1, self.image_size, self.image_size)

                pbar_iteration.update(1)
                iteration += 1
                self.x_s.copy_(data_x_s)
                self.x_t.copy_(data_x_t)
                self.c_s.copy_(data_c_s)
                self.c_t.copy_(data_c_t)

                x_s_, x_t_, z_s_bag, z_t_bag, cons_bag = self.ada_vae(self.x_s, self.x_t, self.c_s, self.c_t)

                # kl divergence
                kld_s = kld_loss(z_s_bag)
                kld_t = kld_loss(z_t_bag)
                kld = kld_s + kld_t

                # reconstruction loss
                recon_s = mse_loss(self.x_s, x_s_)
                recon_t = mse_loss(self.x_t, x_t_)
                recon = recon_s + recon_t

                # fusion loss
                fusion_loss_1 = mse_loss(cons_bag[0], cons_bag[1])
                fusion_loss_2 = mse_loss(cons_bag[2], cons_bag[3])
                fusion_loss_3 = mse_loss(cons_bag[4], cons_bag[5])
                fusion_loss_4 = mse_loss(cons_bag[6], cons_bag[7])
                fusion = fusion_loss_1 + fusion_loss_2 + fusion_loss_3 + fusion_loss_4

                # total loss
                total_loss = self.alpha*recon + self.beta*kld + self.gamma*fusion

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                if iteration % self.print_iter == 0:
                    pbar_iteration.write('[%d/%d] kld: %.6f, recon: %.6f, fusion: %.6f, total_loss: %.6f' % 
                    (iteration, max_iteration, kld.detach().cpu().numpy(), recon.detach().cpu().numpy(), fusion.detach().cpu().numpy(), total_loss.detach().cpu().numpy()))
                    loss_collector.append([epoch, iteration, kld.detach().cpu().numpy(), recon.detach().cpu().numpy(), fusion.detach().cpu().numpy(), total_loss.detach().cpu().numpy()])
            
            # save model
            if epoch % self.save_epoch == 0:
                self.save_model()
                pbar_iteration.write('[*] Save one model')
                np.save(self.sample_path+'/loss.npy', np.array(loss_collector))

            pbar_iteration.close()
            pbar_epoch.update(1)

        pbar_epoch.write("[*] Training stage finishes")
        pbar_epoch.close()

    # for showing transfer results
    def sample(self):
        # load model
        self.load_model()
        print('[*] Finish loading parameters from file')

        pbar_iteration = tqdm(total=self.num_sample, desc='[Iteration]')
        count = 0
        with torch.no_grad():
            for data_x_s, data_x_t, data_c_s, data_c_t in self.dataloader:
                self.batch_size = data_x_s.size(0)
                self.x_s.resize_(self.batch_size, 50, 4)
                self.x_t.resize_(self.batch_size, 50, 4)
                self.c_s.resize_(self.batch_size, 1, self.image_size, self.image_size)
                self.c_t.resize_(self.batch_size, 1, self.image_size, self.image_size)

                pbar_iteration.update(1)
                self.x_s.copy_(data_x_s)
                self.x_t.copy_(data_x_t)
                self.c_s.copy_(data_c_s)
                self.c_t.copy_(data_c_t)

                x_f_s, x_f_t, z_s, z_t, z_f_test = self.ada_vae.test_forward(self.x_s, self.x_t, self.c_s, self.c_t)
                x_s_, x_t_, _, _, _ = self.ada_vae(self.x_s, self.x_t, self.c_s, self.c_t)

                np.save(self.sample_path+'/c_s.'+str(count)+'.npy', self.c_s[0:10].detach().cpu().numpy())
                np.save(self.sample_path+'/x_f_s.'+str(count)+'.npy', x_f_s.detach().cpu().numpy())
                np.save(self.sample_path+'/c_t.'+str(count)+'.npy', self.c_t[0:10].detach().cpu().numpy())
                np.save(self.sample_path+'/x_f_t.'+str(count)+'.npy', x_f_t.detach().cpu().numpy())
                #np.save(self.sample_path+'/x_s.'+str(count)+'.npy', self.x_s.detach().cpu().numpy())
                #np.save(self.sample_path+'/x_t.'+str(count)+'.npy', self.x_t.detach().cpu().numpy())
                #np.save(self.sample_path+'/x_s_.'+str(count)+'.npy', x_s_.detach().cpu().numpy())
                #np.save(self.sample_path+'/x_t_.'+str(count)+'.npy', x_t_.detach().cpu().numpy())
                
                count += 1
                if count > self.num_sample:
                    break

        # save latent code for showing tsne
        #save_latent_code(self.sample_path, [z_s.detach().cpu().numpy(), z_t.detach().cpu().numpy(), z_f_test.detach().cpu().numpy()])

        pbar_iteration.write('[*] Finish sampling')
        pbar_iteration.close()

    # save the generated samples with 100 different weights
    # mainly for showing tsne results
    def fusion(self):
        # load model
        self.load_model()
        print('[*] Finish loading parameters from file')

        pbar_iteration = tqdm(total=self.num_sample, desc='[Iteration]')
        count = 0
        with torch.no_grad():
            for data_x_s, data_x_t, data_c_s, data_c_t in self.dataloader:
                self.batch_size = data_x_s.size(0)
                self.x_s.resize_(self.batch_size, 50, 4)
                self.x_t.resize_(self.batch_size, 50, 4)
                self.c_s.resize_(self.batch_size, 1, self.image_size, self.image_size)
                self.c_t.resize_(self.batch_size, 1, self.image_size, self.image_size)

                pbar_iteration.update(1)
                self.x_s.copy_(data_x_s)
                self.x_t.copy_(data_x_t)
                self.c_s.copy_(data_c_s)
                self.c_t.copy_(data_c_t)

                x_f_s, x_f_t = self.ada_vae.test_forward_100(self.x_s, self.x_t, self.c_s, self.c_t, 200)

                np.save(self.sample_path+'/x_f_s.'+str(count)+'.npy', x_f_s.detach().cpu().numpy())
                np.save(self.sample_path+'/x_f_t.'+str(count)+'.npy', x_f_t.detach().cpu().numpy())

                count += 1
                if count > self.num_sample:
                    break

        pbar_iteration.write('[*] Finish sampling')
        pbar_iteration.close()

    def generate(self):
        # load model
        self.load_model()
        print('[*] Finish loading parameters from file')

        extended_dataset = np.zeros((len(self.dataset), 50, 4))
        #extended_map = np.zeros((len(self.dataset), 1, self.image_size, self.image_size))
        pbar_iteration = tqdm(total=int(len(self.dataset)/self.batch_size), desc='[Iteration]')
        count = 0
        with torch.no_grad():
            for data_x_s, data_x_t, data_c_s, data_c_t in self.dataloader:
                self.batch_size = data_x_s.size(0)
                self.x_s.resize_(self.batch_size, 50, 4)
                self.x_t.resize_(self.batch_size, 50, 4)
                self.c_s.resize_(self.batch_size, 1, self.image_size, self.image_size)
                self.c_t.resize_(self.batch_size, 1, self.image_size, self.image_size)

                pbar_iteration.update(1)
                self.x_s.copy_(data_x_s)
                self.x_t.copy_(data_x_t)
                self.c_s.copy_(data_c_s)
                self.c_t.copy_(data_c_t)

                x_f_s, x_f_t = self.ada_vae.generate_forward(self.x_s, self.x_t, self.c_s, self.c_t)
                extended_dataset[count*self.batch_size:(count+1)*self.batch_size] = x_f_s.detach().cpu().numpy()
                #extended_map[count*self.batch_size:(count+1)*self.batch_size] = self.c_s.cpu().numpy()
                count += 1

        np.save(os.path.join(self.extended_path, 'data.extended.train.npy'), extended_dataset)
        #np.save(os.path.join(self.extended_path, 'map.extended.train.npy'), extended_map)
        pbar_iteration.write('[*] Finish generating extended dataset')
        pbar_iteration.close()

    def encode(self):
        # load dataset
        self.dataset = DPGMM_Dataset(self.extended_path, self.normal_data_path, self.collision_data_path)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        print('[*] Finish loading dataset')

        # load model
        self.load_model()
        print('[*] Finish loading parameters from file')
        encoded_feature_s = np.zeros((len(self.dataset), 32))
        encoded_feature_t = np.zeros((len(self.dataset), 32))
        encoded_feature_f = np.zeros((len(self.dataset), 32))
        self.x_f = CUDA(Variable(torch.FloatTensor(self.batch_size, 50, 4)))

        pbar_iteration = tqdm(total=int(len(self.dataset)/self.batch_size), desc='[Iteration]')
        count = 0
        with torch.no_grad():
            for data_x_s, data_x_t, data_x_f in self.dataloader:
                self.batch_size = data_x_s.size(0)
                self.x_s.resize_(self.batch_size, 50, 4)
                self.x_t.resize_(self.batch_size, 50, 4)
                self.x_f.resize_(self.batch_size, 50, 4)

                pbar_iteration.update(1)
                self.x_s.copy_(data_x_s)
                self.x_t.copy_(data_x_t)
                self.x_f.copy_(data_x_f)

                z_s, z_t, z_f = self.ada_vae.encode(self.x_s, self.x_t, self.x_f)
                encoded_feature_s[count*self.batch_size:(count+1)*self.batch_size] = z_s.detach().cpu().numpy()
                encoded_feature_t[count*self.batch_size:(count+1)*self.batch_size] = z_t.detach().cpu().numpy()
                encoded_feature_f[count*self.batch_size:(count+1)*self.batch_size] = z_f.detach().cpu().numpy()
                count += 1

        np.save(os.path.join(self.extended_path, 'feature.normal.train.npy'), encoded_feature_s)
        np.save(os.path.join(self.extended_path, 'feature.collision.train.npy'), encoded_feature_t)
        np.save(os.path.join(self.extended_path, 'feature.extended.train.npy'), encoded_feature_f)
        pbar_iteration.write('[*] Finish encoding data to feature space')
        pbar_iteration.close()

    def save_model(self):
        states = {'ada_vae_states': self.ada_vae.state_dict(), 'optim_states': self.optimizer.state_dict()}

        filepath = os.path.join(self.model_path, 'model.'+str(self.model_id)+'.torch')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)

    def load_model(self):
        filepath = os.path.join(self.model_path, 'model.'+str(self.model_id)+'.torch')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)

            self.ada_vae.load_state_dict(checkpoint['ada_vae_states'])
            self.optimizer.load_state_dict(checkpoint['optim_states'])
