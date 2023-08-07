import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as tdist
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from .networks import *
from .loss import *


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0
        self.device = config.DEVICE
        self.val_iter = config.VAL_ITERS
        self.imagine_g_weights_path = os.path.join(config.PATH, 'g.pth')
        self.imagine_d_weights_path = os.path.join(config.PATH, 'd.pth')
        self.imagine_dp_weights_path = os.path.join(config.PATH, 'dp.pth')
        self.imagine_e_weights_path = os.path.join(config.PATH, 'e.pth')

    def load(self):
        if os.path.exists(self.imagine_g_weights_path):
            print('Loading %s Model ...' % self.name)

            g_data = torch.load(self.imagine_g_weights_path)
            self.g.load_state_dict(g_data['params'])
            self.iteration = g_data['iteration']

        if os.path.exists(self.imagine_d_weights_path):
            d_data = torch.load(self.imagine_d_weights_path)
            self.d.load_state_dict(d_data['params'])

        if os.path.exists(self.imagine_e_weights_path):
            e_data = torch.load(self.imagine_e_weights_path)
            self.e.load_state_dict(e_data['params'])

        if os.path.exists(self.imagine_dp_weights_path):
            dp_data = torch.load(self.imagine_dp_weights_path)
            self.d_p.load_state_dict(dp_data['params'])

    def save(self, ite=None):
        print('\nSaving %s...\n' % self.name)
        if ite is not None:
            torch.save({
                'iteration': self.iteration,
                'params': self.g.state_dict()}, self.imagine_g_weights_path + '_' + str(ite))
            torch.save({'params': self.e.state_dict()}, self.imagine_e_weights_path + '_' + str(ite))
            torch.save({'params': self.d.state_dict()}, self.imagine_d_weights_path + '_' + str(ite))
            torch.save({'params': self.d_p.state_dict()}, self.imagine_dp_weights_path + '_' + str(ite))            
        else:
            torch.save({
                'iteration': self.iteration,
                'params': self.g.state_dict()}, self.imagine_g_weights_path)
            torch.save({'params': self.e.state_dict()}, self.imagine_e_weights_path)
            torch.save({'params': self.d.state_dict()}, self.imagine_d_weights_path)
            torch.save({'params': self.d_p.state_dict()}, self.imagine_dp_weights_path)

class Network(BaseModel):
    def __init__(self, config):
        super(Network, self).__init__('Network', config)

        g = Inpainting(config)
        e = Estimating(config)
        d_p = MultiscaleDiscriminator()
        d = DenseD()

        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss()
        criterion = nn.BCELoss()
        content_loss = PerceptualLoss()

        self.add_module('g', g)
        self.add_module('d', d)
        self.add_module('d_p', d_p)
        self.add_module('adversarial_loss', adversarial_loss)
        self.add_module('l1_loss', l1_loss)
        self.add_module('e', e)
        self.add_module('criterion', criterion)
        self.add_module('content_loss', content_loss)

        self.g_optimizer = torch.optim.Adam(params=g.parameters(), lr=config.G_LR, betas=(config.BETA1, config.BETA2))
        self.e_optimizer = torch.optim.Adam(params=e.parameters(), lr=config.E_LR, betas=(config.BETA1, config.BETA2))
        self.d_optimizer = torch.optim.Adam(params=d.parameters(), lr=config.D_LR, betas=(config.BETA1, config.BETA2))
        self.d_p_optimizer = torch.optim.Adam(params=d_p.parameters(), lr=config.D_LR, betas=(config.BETA1, config.BETA2))

    def process(self, data, pdata, mask, ite):
        self.iteration += 1
        self.ite = ite
        # zero optimizers
        self.g_optimizer.zero_grad()
        self.e_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.d_p_optimizer.zero_grad()
          
        g_loss = 0
        g_adv = 0
        e_loss = 0
        d_loss = 0
        d_p_loss = 0
        c_loss = 0
        f_loss = 0

########################################Estimate##########################################
        if self.ite % self.config.N_UPDATE_E == 0:

            fea_I0 = self.g(pdata)
            mask_logit, mask_soft, mask_pred, fea_Ei = self.e(pdata, fea_I0[0].detach(), fea_I0[1].detach(), fea_I0[2].detach())  
            output = self.g(pdata, mask_soft, fea_Ei[0], fea_Ei[1], fea_Ei[2]) 

            # g bce loss
            e_bce_loss = self.criterion(mask_soft, mask) 
            e_loss += e_bce_loss

            e_loss.backward()
            self.e_optimizer.step()

            self.e_loss = e_loss

            logs = [
                ("l_e", e_loss.item()) 
                ]  

########################################Inpainting##########################################
        if self.ite % self.config.N_UPDATE_I == 0:
            fea_I0 = self.g(pdata)
            mask_logit, mask_soft, mask_pred, fea_Ei = self.e(pdata, fea_I0[0], fea_I0[1], fea_I0[2])  
            output = self.g(pdata, mask_soft.detach(), fea_Ei[0].detach(), fea_Ei[1].detach(), fea_Ei[2].detach()) 

            #### D loss####
            d_real = data
            d_fake = output.detach()
            ####Global D loss####
            d_real_g = self.d(d_real)
            d_fake_g = self.d(d_fake)
            g_fake = self.d(output)

            d_real_l = self.adversarial_loss(d_real_g, True, True)
            d_fake_l = self.adversarial_loss(d_fake_g, False, True)
            d_loss += (d_real_l + d_fake_l) / 2   
            g_adv += self.adversarial_loss(g_fake, True, False) * self.config.G1_ADV_LOSS_WEIGHT

            ####Local D loss####

            d_real_arr_p = self.d_p(d_real * mask)
            d_fake_arr_p = self.d_p(d_fake * mask)
            g_fake_arr_p = self.d_p(output * mask)
            g_p_adv = 0

            for i in range(len(d_real_arr_p)):
                d_p_real_l = self.adversarial_loss(d_real_arr_p[i], True, True)
                d_p_fake_l = self.adversarial_loss(d_fake_arr_p[i], False, True)
                d_p_loss += (d_p_real_l + d_p_fake_l) / 2

                g_p_adv += self.adversarial_loss(g_fake_arr_p[i], True, False)

            d_p_loss = d_p_loss / 2
            g_adv += g_p_adv / 2 * self.config.G1_ADV_LOSS_WEIGHT
            f_loss += g_adv / 2 

            # g l1 loss ##     
            g_l1_loss = self.l1_loss(output, data) * self.config.G2_L1_LOSS_WEIGHT
            c_loss += g_l1_loss

            # g content loss #
            g_content_loss, g_mrf_loss = self.content_loss(output, data)
            g_content_loss = g_content_loss * self.config.G1_CONTENT_LOSS_WEIGHT
            g_mrf_loss = g_mrf_loss * self.config.G2_STYLE_LOSS_WEIGHT
            c_loss += g_content_loss
            f_loss += g_mrf_loss

            g_loss = c_loss + f_loss

            d_loss.backward()
            self.d_optimizer.step()

            d_p_loss.backward()
            self.d_p_optimizer.step()

            g_loss.backward()
            self.g_optimizer.step()

            logs = [
                ("l_e", self.e_loss.item()),
                ("l_d", d_loss.item()),
                ("l_dp", d_p_loss.item()),
                ("l_g_adv", g_adv.item()),
                ("l_l1", g_l1_loss.item()),
                ("l_loss", g_loss.item())       
                ]
            
        return output, mask_pred, e_loss, d_loss, d_p_loss, g_loss, logs


    def forward(self, input):
        fea_I0 = self.g(input)
        mask_logit, mask_soft, mask_pred, fea_Ei = self.e(input, fea_I0[0], fea_I0[1], fea_I0[2])  
        output = self.g(input, mask_soft, fea_Ei[0], fea_Ei[1], fea_Ei[2])  
        return output, mask_logit, mask_soft, mask_pred
