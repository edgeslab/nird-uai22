import pdb
import logging
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import networkx as nx

import warnings
warnings.filterwarnings('ignore')

from ..cits.SIC.stattests import hrt_sobolev, compute_fdr, compute_tpr
from ..cits.SIC.SIC_imports import Ep_D, sobolev_forward, avg_sobolev_dist, eta_optim_step_
from ..cits.SIC.modules.models import init_D, init_optimizerD
from ..cits.SIC.utils import DDICT, avg_iterable
from ..cits.SIC.datasets.dataset_builder import build_dataset
from ..cits.cit import ConditionalTest


class SICTest(ConditionalTest):

    def __init__(self, seed):
        self.logger = logging.getLogger('main')

        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', default='nird', help='toy | bleitoy | liang | liang_switch | ccle | olfaction | biobank | hiv ') #sinexp
        parser.add_argument('--data-seed', type=int, default=0, help='initial random seed for data')
        parser.add_argument('--Yfunction', default='sine', help='sine | linear ')
        parser.add_argument('--task', default='PLX4720', help='(ccle): PLX4720 | (olfaction): Bakery')
        parser.add_argument('--Xdim', type=int, default=1,   help='(for toy) X dimensionality') #50
        parser.add_argument('--numSamples', type=int, default=99,   help='(for toy) num Samples in train & heldout')
        parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
        parser.add_argument('--DiscArch', default='concat_first', help='phiVpsi | concat | concat2')
        parser.add_argument('--layerSize', type=int, default=100, help='')
        parser.add_argument('--nonlin', default='ReLU', help='')
        parser.add_argument('--normalization', default='', help='None | LN layernorm | TODO more options?')
        parser.add_argument('--wdecay', type=float, default=1e-4, help='')
        parser.add_argument('--lrD', type=float, default=1e-3, help='learning rate for D = Sobolev Mut Info neural estimator')
        parser.add_argument('--beta1', type=float, default=0.5, help='Adam optimizer for D: (beta1, beta2)')
        parser.add_argument('--beta2', type=float, default=0.999, help='Adam optimizer for D: (beta1, beta2)')
        parser.add_argument('--lambdaFisher', type=float, default=0.01, help='lambda on Fisher constraint term E_Q[f^2]')
        parser.add_argument('--lambdaSobolev', type=float, default=0.01, help='lambda on Sobolev constraint term')
        parser.add_argument('--mu', default='Q', help='Q | P | P+Q.  mu: dominant measure on which to constrain expectations.')
        parser.add_argument('--eta-lr', type=float, default=0.1, help='lr for eta; in case of L1^2 this is mirror descent scale')
        parser.add_argument('--T', type=int, default=200, help='number of updates to D, training duration')
        parser.add_argument('--log-every', type=int, default=10, help='interval to log, compute metrics on heldout')
        parser.add_argument('--eta-step_type', default='mirror', help='mirror | reduced')
        parser.add_argument('--seed', type=int, default=1238, help='random seed')
        parser.add_argument('--dataseed', type=int, default=1258, help='random seed for toy datasets')
        parser.add_argument('--ftdr-cutoff', default=1, type=int, help='fdr / tpr cutoff')
        parser.add_argument('--dropout', default=0.3, type=float, help='Discriminator/Critic dropout')
        parser.add_argument('--n-critic', default=1, type=int, help='No. of critic before eta update')
        parser.add_argument('--do-hrt', action='store_true', help='perform HRT')
        parser.add_argument('--hrt-cutoff', type=int, default=20, help='maximal number of features for HRT to evaluate')
        parser.add_argument('--target-fdr', default=0.1, type=float, help='target FDR for HRT')
        parser.add_argument('--sinexp-rho', default=0.5, type=float, help='Correlation coefficient between pairs of covariates in SinExp dataset')
        parser.add_argument('--sinexp-gaussian', action='store_true', help='Sample covariates from gaussian or uniform in SinExp dataset')
        parser.add_argument('--generator-type', default='classify', help='regress | classify')
        parser.add_argument('--n-runs', type=int, default=100, help='number of repetitons over everything')
        parser.add_argument('--nocuda', action='store_true', help='disables cuda')
        self.args = parser.parse_args([])
        self.args.model = 'sic'

        if self.args.nocuda:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device:', self.device)



    def train_sobo_critic(self, opt, dataloaders, D, groundtruth_feat=None, n_epochs=1, train_last_layer_only=False, logger=None):
        """
            Args:
                opt (DDICT): parameters for training
                dataloaders (tuple): list of dataloaders
                D (nn.module): Discriminator architecture
        """
        dl_train_P, dl_train_Q, dl_test_P, dl_test_Q = dataloaders # unpack

        # Optizer
        optimizerD = init_optimizerD(opt, D, train_last_layer_only=train_last_layer_only)

        # etas are initialized uniformly
        eta_x = torch.tensor([1 / opt.Xdim] * opt.Xdim, device=next(D.parameters()).device, requires_grad=True)

        # Log architecture
        if logger: logger.info(D)

        # Train
        if logger: logger.info('Start training')

        for epoch in range(n_epochs):
            for batch_idx, (dataP, dataQ) in enumerate(zip(dl_train_P, dl_train_Q)):
                n_iter = epoch * len(dl_train_P) + batch_idx

                optimizerD.zero_grad()
                if hasattr(eta_x.grad, 'zero_'):
                    eta_x.grad.zero_()

                sobo_dist, constraint_f2, constraint_Sobo = sobolev_forward(D, eta_x, dataP, dataQ, opt.mu)

                obj_D = - sobo_dist \
                        + opt.lambdaFisher * constraint_f2 \
                        + (opt.lambdaSobolev / 2) * constraint_Sobo

                obj_D.backward()
                optimizerD.step()

                if (n_iter + 1) % opt.n_critic == 0:
                    eta_optim_step_(eta_x, opt.eta_step_type, opt.eta_lr)

            # eval / logging
            if logger and epoch % opt.log_every == 0:
                # Average test sobolev distance
                sobo_dist_te, constraint_f2_te, constraint_Sobo_te = avg_iterable(
                    zip(dl_test_P, dl_test_Q), lambda PQ: sobolev_forward(D, eta_x, PQ[0], PQ[1], opt.mu))

                obj_D_te = - sobo_dist_te \
                        + opt.lambdaFisher * constraint_f2_te \
                        + (opt.lambdaSobolev / 2) * constraint_Sobo_te

                msg = '[{:5d}]   TRAIN: obj_D={:.4f}, sobo-dist={:.4f}   TEST: obj_D={:.4f}, sobo-dist={:.4f}'\
                    .format(epoch, obj_D.item(), sobo_dist.item(), constraint_Sobo.item(), obj_D_te.item(), sobo_dist_te.item(), constraint_Sobo_te.item())

                # fdr and tpr
                if groundtruth_feat:
                    _, eta_sortix = torch.sort(eta_x, descending=True)
                    fdr = compute_fdr(eta_sortix.clone().detach().cpu(), groundtruth_feat, eta_sortix.size(0), cut_off=opt.ftdr_cutoff)
                    tpr = compute_tpr(eta_sortix.clone().detach().cpu(), groundtruth_feat, eta_sortix.size(0), cut_off=opt.ftdr_cutoff)
                    msg += '   FDR={:.3f}, TPR={:.3f}'.format(fdr, tpr)

                logger.info(msg)

        sobo_dist_tr = avg_sobolev_dist(D, dl_train_P, dl_train_Q)
        sobo_dist_te = avg_sobolev_dist(D, dl_test_P, dl_test_Q)
        return D, eta_x, sobo_dist_tr, sobo_dist_te
        


    def sobolev_dist_fn(self, dl_P):
        return Ep_D(self.D, dl_P)


    def get_data(self, g, cond=False, rmask='000'):
        #TODO: add hops parameter

        _x = []
        _y = []
        _z = [] #TODO; handle it

        for i in g.nodes():
            trt = g.nodes[i]['trt']
            out = g.nodes[i]['out']
            z = g.nodes[i]['att_0'] if 'att_0' in g.nodes[i] else None

            neighbors = list(g.neighbors(n=i))
            if rmask == '100':
                for n in neighbors:
                    _x.append(g.nodes[n]['trt'])
                    _y.append(out)
                    if z:
                        _z.append(z)
            elif rmask == '001':
                for n in neighbors:
                    _x.append(trt)
                    _y.append(out)
                    _z.append(g.nodes[n]['att_0'])
            elif rmask == '101':
                for i1 in range(len(neighbors)):
                    for i2 in range(len(neighbors)):
                        n1, n2 = neighbors[i1], neighbors[i2]
                        _x.append(g.nodes[n1]['trt'])
                        _y.append(out)
                        _z.append(g.nodes[n2]['att_0'])
            else:
                _x.append(trt)
                _y.append(out)
                if z:
                    _z.append(z)

        X = np.array(_x).reshape(-1, 1)
        Y = np.array(_y).reshape(-1, 1)

        Z = None
        if len(_z) > 0:
            Z = np.array(_z).reshape(-1, 1)

        return X, Y, Z


    def run_test(self, g, cond=False, rmask='000', samples=[], approx=False, trial=0):
        # build data
        X, Y, Z = self.get_data(g, cond, rmask)

        self.args.Xdim = 2 if cond else 1
        self.args.numSamples = X.shape[0]

        data_opt = DDICT(
            dataset=self.args.dataset,
            sinexp_gaussian=self.args.sinexp_gaussian,
            sinexp_rho=self.args.sinexp_rho,
            numSamples=self.args.numSamples,
            Xdim=self.args.Xdim,
            batchSize=self.args.batchSize,
            dataseed=self.args.data_seed,
        )

        dataloaders, fea_names, groundtruth_feat = build_dataset(data_opt, X, Y, Z)

        self.D = init_D(self.args, self.device)
        self.D, eta_x, _, _ = self.train_sobo_critic(self.args, dataloaders, self.D, groundtruth_feat=groundtruth_feat, n_epochs=self.args.T, logger=self.logger)

        p_value = hrt_sobolev( eta_x, self.sobolev_dist_fn, dataloaders, hrt_cutoff=2, 
                                    target_fdr=None, generator_type=self.args.generator_type, 
                                    n_rounds=100, logger=self.logger )
        

        return p_value