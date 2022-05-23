import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb as pdb


class WeightAnnealing:
    def __init__(self, c_lambda=0.0, warmup_lambda=1000, lambda_update_steps=1000, maxclambda=100.0, n_gpus=1, tbs=12):
        self.c_lambda = c_lambda
        self.warmup_lambda = warmup_lambda//(n_gpus*(tbs//12))
        self.maxclambda = maxclambda
        self.lambda_update_steps = lambda_update_steps//(n_gpus**(tbs//12))
        self.increment = 0.1
        self.step = 0

    def scheduler_lambda(self, val=False):
        if val:
            return self.c_lambda

        self.step += 1
        if self.step >= self.warmup_lambda and self.step % self.lambda_update_steps == 0:
            if self.c_lambda <= self.maxclambda:
                self.c_lambda += self.increment
            else:
                self.c_lambda = self.maxclambda
        return self.c_lambda

    def get_lam(self):
        return self.c_lambda


class LossDivergence:
    def __init__(self, weightlambda, divergence='kl'):
        self.weightlambda = weightlambda
        self.divergence = divergence
        if divergence == 'kl':
            self.weightlambda.increment = 2.0
        else:
            self.weightlambda.increment = 2.0

    def get_loss(self, p, q):
        if self.divergence == 'kl':
            return self.get_loss_kl(p, q)
        else:
            return self.get_loss_js(p, q)

    def get_loss_kl(self, p, q):
        loss_kl_fct = nn.KLDivLoss(reduction='batchmean')
        loss = loss_kl_fct(F.log_softmax(p, dim=1), F.softmax(q, dim=1))
        return loss

    def get_loss_js(self, p, q):
        loss_js_fct = JSD()
        finalloss = loss_js_fct(p, q)
        return finalloss


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()

    def forward(self, net_1_logits, net_2_logits):
        net_1_probs = F.softmax(net_1_logits, dim=1)
        net_2_probs = F.softmax(net_2_logits, dim=1)

        total_m = 0.5 * (net_1_probs + net_2_probs)
        loss = 0.0
        loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), total_m, reduction="batchmean")
        loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), total_m, reduction="batchmean")

        return (0.5 * loss)
