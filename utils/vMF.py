import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import ive
import numpy as np
import torch.distributed as dist
from sklearn.cluster import KMeans

#from PROCO TPAMI2024

def miller_recurrence(order, kappa):
    I_n = torch.ones(1, dtype=torch.float64).cuda()
    I_n1 = torch.zeros(1, dtype=torch.float64).cuda()

    Estimat_n = [order, order+1]
    scale0 = 0 
    scale1 = 0 
    scale = 0

    for i in range(2*order, 0, -1):
        I_n_tem, I_n1_tem = 2 * i/kappa * I_n + I_n1, I_n
        if torch.isinf(I_n_tem).any():
            I_n1 /= I_n
            scale += torch.log(I_n)
            if i >= (order + 1):
                scale0 += torch.log(I_n)
                scale1 += torch.log(I_n)
            elif i == order:
                scale0 += torch.log(I_n)

            I_n = torch.ones(1, dtype=torch.float64).cuda()
            I_n, I_n1 = 2 * i/kappa * I_n + I_n1, I_n

        else:
            I_n, I_n1 = I_n_tem, I_n1_tem

        if i == order:
            Estimat_n[0] = I_n1
        elif i == (order + 1):
            Estimat_n[1] = I_n1

    ive0 = torch.special.i0e(kappa.cuda())

    Estimat_n[0] = torch.log(ive0) + torch.log(Estimat_n[0]) - torch.log(I_n) + scale0 - scale
    Estimat_n[1] = torch.log(ive0) + torch.log(Estimat_n[1]) - torch.log(I_n) + scale1 - scale

    return Estimat_n[0], Estimat_n[1]


class LogRatioC(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, k, p, logc):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """


        nu, nu1 = miller_recurrence((p/2 - 1).int(), k.double())

        tensor = nu + k - (p/2 - 1) * torch.log(k+1e-20) - logc
        # tensor = log(ive(p/2-1, k)) + k - (p/2 - 1) * log(k) - logc
        #        = log(iv(p/2-1, k)) - (p/2 - 1) * log(k) - logc
        #        = log(C(\tilde{\kappa})/C(\kappa))
        ctx.save_for_backward(torch.exp(nu1 - nu))

        return tensor


    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        grad = ctx.saved_tensors[0]
        # grad clip
        grad[grad > 1.0] = 1.0

        grad *= grad_output

        return grad, None, None


class EstimatorCV():
    def __init__(self, feature_num, class_num, src_features=None):
        super(EstimatorCV, self).__init__()
        
        self.class_num = class_num
        self.feature_num = feature_num
        
        # 如果有源域特征，使用聚类初始化
        if src_features is not None and len(src_features) > 0:
            kmeans = KMeans(n_clusters=class_num, random_state=0)
            features_np = src_features.detach().cpu().numpy()
            kmeans.fit(features_np)
            centroids = torch.from_numpy(kmeans.cluster_centers_).float()
            self.Ave = F.normalize(centroids, dim=1)
        else:
            # 回退到随机初始化
            self.Ave = F.normalize(torch.randn(class_num, feature_num), dim=1) * 0.9
        
        self.Amount = torch.zeros(class_num)
        # 估计 kappa 基于特征分布
        if src_features is not None:
            R = torch.linalg.norm(self.Ave, dim=1)
            self.kappa = self.feature_num * R / (1 - R**2)
            self.kappa[self.kappa > 1e5] = 1e5
            self.kappa[self.kappa < 0] = 1e5
        else:
            self.kappa = torch.ones(class_num) * self.feature_num * 90 / 19
        
        # 计算归一化常数
        tem = torch.from_numpy(ive(self.feature_num/2 - 1, self.kappa.cpu().numpy().astype(np.float64))).to(self.kappa.device)
        self.logc = torch.log(tem+1e-300) + self.kappa - (self.feature_num/2 - 1) * torch.log(self.kappa+1e-300)
        
        if torch.cuda.is_available():
            self.Ave = self.Ave.cuda()
            self.Amount = self.Amount.cuda()
            self.kappa = self.kappa.cuda()
            self.logc = self.logc.cuda()

    def reset(self):
        self.Ave = F.normalize(torch.randn(self.class_num, self.feature_num), dim=1) * 0.9
        self.Amount = torch.zeros(self.class_num)
        self.kappa = torch.ones(self.class_num) * self.feature_num * 90 / 19
        tem = torch.from_numpy(ive(self.feature_num/2 - 1, self.kappa.cpu().numpy().astype(np.float64))).to(self.kappa.device)
        self.logc = torch.log(tem+1e-300) + self.kappa - (self.feature_num/2 - 1) * torch.log(self.kappa+1e-300)
        if torch.cuda.is_available():
            self.Ave = self.Ave.cuda()
            self.Amount = self.Amount.cuda()
            self.kappa = self.kappa.cuda()
            self.logc = self.logc.cuda()

    def reload_memory(self):
        if torch.cuda.is_available():
            self.Ave = self.Ave.cuda()
            self.Amount = self.Amount.cuda()
            self.kappa = self.kappa.cuda()
            self.logc = self.logc.cuda()
 
    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )

        onehot = torch.zeros(N, C)
        if torch.cuda.is_available():
            onehot = onehot.cuda()


        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA


        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)


        weight_AV = sum_weight_AV.div(
            sum_weight_AV + self.Amount.view(C, 1).expand(C, A)
        )
        weight_AV[weight_AV != weight_AV] = 0

        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()

        self.Amount += onehot.sum(0)


    def update_kappa(self, kappa_inf=False):
        R = torch.linalg.norm(self.Ave, dim=1)
        self.kappa = self.feature_num * R / (1 - R**2)

        self.kappa[self.kappa > 1e5] = 1e5
        self.kappa[self.kappa < 0] = 1e5

        nu, _ = miller_recurrence(torch.tensor(self.feature_num/2 - 1).int(), self.kappa.double())  # 计算贝塞尔函数 Iv 的对数

        self.logc = nu + self.kappa - (self.feature_num/2 - 1) * torch.log(self.kappa+1e-20)


class vFM_cluster(nn.Module):
    def __init__(self, contrast_dim, temperature=1.0, num_classes=1000,src_features=None):
        super(vFM_cluster, self).__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.feature_num = contrast_dim
        self.estimator_old = EstimatorCV(self.feature_num, num_classes,src_features)
        self.estimator = EstimatorCV(self.feature_num, num_classes,src_features)

    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        if torch.cuda.is_available():
            self.weight = self.weight.to(torch.device('cuda'))

    def reload_memory(self):
        self.estimator_old.reload_memory()
        self.estimator.reload_memory()

    def _hook_before_epoch(self, epoch, epochs):
        # exchange ave and covariances
        self.estimator_old.Ave = self.estimator.Ave
        self.estimator_old.Amount = self.estimator.Amount

        self.estimator_old.kappa = self.estimator.kappa
        self.estimator_old.logc = self.estimator.logc

        self.estimator.reset()
    
    def get_ave(self):
        Ave = self.estimator_old.Ave.detach()
        Ave_norm = F.normalize(Ave, dim=1)
        return Ave_norm

    def forward(self, features, labels=None, real_span_mask=None):
        batch_size = features.size(0)
        N = batch_size

        if labels is not None:
            self.estimator_old.update_CV(features.detach()[real_span_mask], labels[real_span_mask])
            self.estimator.update_CV(features.detach()[real_span_mask], labels[real_span_mask])
            self.estimator_old.update_kappa()

        Ave = self.estimator_old.Ave.detach()
        Ave_norm = F.normalize(Ave, dim=1)
        logc = self.estimator_old.logc.detach()     # log(C_p(\kappa_j))
        kappa = self.estimator_old.kappa.detach()   # \kappa_j

        temp = kappa.reshape(-1, 1) * Ave_norm
        temp = temp.unsqueeze(0) + features[:N].unsqueeze(1) / self.temperature
        kappa_new = torch.linalg.norm(temp, dim=2)   # \tilde{\kappa_j}

        contrast_logits = LogRatioC.apply(kappa_new, torch.tensor(self.estimator.feature_num), logc).to(dtype=torch.float32)

        return F.normalize(contrast_logits, dim=1)
