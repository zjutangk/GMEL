import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import random
import torch
from torch import nn
import torch.nn.functional as F
import faiss
import time


def multiclass_acc(preds, truths):
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = test_truth_emo > 0
    predicted_label = test_preds_emo > 0
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))

    return (tp * (n / p) + tn) / (2 * n)


def eval_senti(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()
    # 将预测值归一化到[-3,3]范围
    min_val = np.min(test_preds)
    max_val = np.max(test_preds)
    test_preds = -3 + (test_preds - min_val) * (6 / (max_val - min_val))
    # test_preds = np.clip(test_preds, -3, 3)#直接规范范围

    non_zeros = np.array(
        [i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)]
    )

    #  添加预测值分布统计
    print("\nPrediction Distribution:")
    print("-" * 50)
    # 定义区间边界，从-3到3，步长0.5
    bins = np.arange(-3, 3, 0.5)
    hist, bin_edges = np.histogram(test_preds, bins=bins)
    
    # 打印每个区间的统计信息
    for i in range(len(bins)-1):
        interval = f"[{bins[i]:.1f}, {bins[i+1]:.1f})"
        count = hist[i]
        percentage = (count / len(test_preds)) * 100
        print(f"{interval:12} : {count:4d} samples ({percentage:5.2f}%)")
    

    # 计算混淆矩阵
    print("\nConfusion Matrix:")
    print("-" * 50)
    
    # 定义区间边界
    bins = np.arange(-3, 3, 0.5)
    n_bins = len(bins) - 1
    
    # 初始化混淆矩阵
    confusion_matrix = np.zeros((n_bins, n_bins))
    
    # 计算每个样本的预测区间和真实区间
    pred_bins = np.digitize(test_preds, bins) - 1
    truth_bins = np.digitize(test_truth, bins) - 1
    
    # 填充混淆矩阵
    for i in range(len(test_preds)):
        if 0 <= pred_bins[i] < n_bins and 0 <= truth_bins[i] < n_bins:
            confusion_matrix[truth_bins[i], pred_bins[i]] += 1
    
    # 打印混淆矩阵
    print("Truth \\ Pred", end="")
    for i in range(n_bins):
        print(f" [{bins[i]:.1f},{bins[i+1]:.1f})", end="")
    print()
    
    for i in range(n_bins):
        print(f"[{bins[i]:.1f},{bins[i+1]:.1f})", end="")
        for j in range(n_bins):
            print(f" {int(confusion_matrix[i,j]):4d}", end="")
        print()
    print("-" * 50)
    # 打印基本统计量
    print("-" * 50)
    print(f"Mean: {np.mean(test_preds):.3f}")
    print(f"Std: {np.std(test_preds):.3f}")
    print(f"Min: {np.min(test_preds):.3f}")
    print(f"Max: {np.max(test_preds):.3f}")
    print("-" * 50)

    #  # 添加真实值分布统计
    # print("\nTruth Distribution:")
    # print("-" * 50)
    # # 定义区间边界，从-3到3，步长0.5
    # bins = np.arange(-5, 5.5, 0.5)
    # hist, bin_edges = np.histogram(test_truth, bins=bins)
    
    # # 打印每个区间的统计信息
    # for i in range(len(bins)-1):
    #     interval = f"[{bins[i]:.1f}, {bins[i+1]:.1f})"
    #     count = hist[i]
    #     percentage = (count / len(test_truth)) * 100
    #     print(f"{interval:12} : {count:4d} samples ({percentage:5.2f}%)")
    
    # # 打印基本统计量
    # print("-" * 50)
    # print(f"Mean: {np.mean(test_truth):.3f}")
    # print(f"Std: {np.std(test_truth):.3f}")
    # print(f"Min: {np.min(test_truth):.3f}")
    # print(f"Max: {np.max(test_truth):.3f}")
    # print("-" * 50)

    test_preds_a7 = np.clip(test_preds, a_min=-3.0, a_max=3.0)
    test_truth_a7 = np.clip(test_truth, a_min=-3.0, a_max=3.0)
    test_preds_a5 = np.clip(test_preds, a_min=-2.0, a_max=2.0)
    test_truth_a5 = np.clip(test_truth, a_min=-2.0, a_max=2.0)

    mae = np.mean(
        np.absolute(test_preds - test_truth)
    )  # Average L1 distance between preds and truths
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score(
        (test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average="weighted"
    )
    binary_truth = test_truth[non_zeros] > 0
    binary_preds = test_preds[non_zeros] > 0
    acc2 = accuracy_score(binary_truth, binary_preds)
    # 统计真实标签正负样本数目
    pos_samples = np.sum(test_truth > 0)
    neg_samples = np.sum(test_truth <= 0)
    # zero_samples = np.sum(test_truth == 0)
    print(f"Positive samples: {pos_samples}")
    print(f"Negative samples: {neg_samples}")
    # print(f"Zero samples: {zero_samples}")
    print(f"Total samples: {len(test_truth)}")

    print("MAE: ", mae)
    print("mult_acc_7: ", mult_a7)
    print("mult_acc_5: ", mult_a5)
    print("F1 score: ", f_score)
    print("Accuracy: ", acc2)
    return acc2


def count_parameters(model):
    trainable_params = 0
    total_params = 0
    trainable_params_list = list()
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_params_list.append(name)
    print(f"Total params: {total_params}, Trainable params: {trainable_params}")


def fix_para_new(model):
    for name, para in model.named_parameters():
        # if "layer_norms" not in name:
        if not ("layer_norms" in name or "out_layer_proj" in name):
            para.requires_grad = False
    return model

def fix_para(model):
    for name, para in model.named_parameters():
        print(name)
        if "layer_norms" not in name:
        # if not ("layer_norms" in name or "layers.3" in name or "layers.4" in name):
            para.requires_grad = False
    return model


def random_drop(text, audio, vision):
    seed = random.random()
    if seed < 0.33:
        text = torch.zeros_like(text)
    elif seed < 0.66:
        audio = torch.zeros_like(audio)
    else:
        vision = torch.zeros_like(vision)
    return text, audio, vision


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, verbose=False):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        N = z_j.shape[0]

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )
        if self.verbose:
            print("Similarity matrix\n", similarity_matrix, "\n")

        def l_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            if self.verbose:
                print(f"sim({i}, {j})={sim_i_j}")

            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = (
                torch.ones((2 * N,))
                .to(emb_i.device)
                .scatter_(0, torch.tensor([i]).to(emb_i.device), 0.0)
            )
            if self.verbose:
                print(f"1{{k!={i}}}", one_for_not_i)

            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )

            loss_ij = -torch.log(numerator / denominator)
            if self.verbose:
                print(f"loss({i},{j})={loss_ij}\n")

            return loss_ij.squeeze(0)

        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2 * N) * loss

def run_kmeans(x, args):
    """
    Args:
        x: data to be clustered
    """
    start_time = time.time()
    print("performing kmeans clustering")
    results = {"im2cluster": [], "centroids": [], "density": []}

    num_cluster = args.num_cluster
    d = x.shape[1]
    k = int(num_cluster)
    clus = faiss.Clustering(d, k)
    clus.verbose = False
    clus.niter = 20
    clus.nredo = 5
    clus.max_points_per_centroid = 1000
    clus.min_points_per_centroid = 10

    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = torch.cuda.current_device() #my
    index = faiss.GpuIndexFlatL2(res, d, cfg)

    clus.train(x, index)

    D, I = index.search(x, 1)
    im2cluster = [int(n[0]) for n in I]

    centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

    Dcluster = [[] for c in range(k)]
    for im, i in enumerate(im2cluster):
        Dcluster[i].append(D[im][0])

    density = np.zeros(k)
    for i, dist in enumerate(Dcluster):
        if len(dist) > 1:
            d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
            density[i] = d

    dmax = density.max()
    for i, dist in enumerate(Dcluster):
        if len(dist) <= 1:
            density[i] = dmax

    density = density.clip(np.percentile(density, 10), np.percentile(density, 90))
    # density = args.temperature * density / density.mean()
    density = 0.07 * density / density.mean()

    centroids = torch.Tensor(centroids).cuda()
    centroids = nn.functional.normalize(centroids, p=2, dim=1)

    im2cluster = torch.LongTensor(im2cluster).cuda()
    density = torch.Tensor(density).cuda()

    results["centroids"] = centroids
    results["density"] = density
    results["im2cluster"] = im2cluster

    print("Kmeans end. Eplapsed {} s".format(time.time() - start_time))

    return results

def run_gmm(x, args):
    """
    Args:
        x: numpy array of shape (n_samples, n_features) to be clustered
        args: arguments containing num_cluster
    Returns:
        results: dict containing centroids, im2cluster, density, and cluster_probs
    """
    from sklearn.mixture import GaussianMixture
    import time
    
    start_time = time.time()
    print("performing GMM clustering")
    results = {
        "im2cluster": [], 
        "centroids": [], 
        "density": [],
        "cluster_probs": [] 
    }

    gmm = GaussianMixture(
        n_components=args.num_cluster,
        covariance_type='full',
        max_iter=20,
        n_init=5,
        random_state=42
    )
    gmm.fit(x)
    
    im2cluster = gmm.predict(x)
    cluster_probs = gmm.predict_proba(x)
    centroids = gmm.means_

    centroids = torch.Tensor(centroids).cuda()
    centroids = nn.functional.normalize(centroids, p=2, dim=1)
    
    im2cluster = torch.LongTensor(im2cluster).cuda()
    cluster_probs = torch.Tensor(cluster_probs).cuda()

    results["centroids"] = centroids
    results["im2cluster"] = im2cluster
    results["cluster_probs"] = cluster_probs
    print("GMM clustering completed. Elapsed {} s".format(time.time() - start_time))
    
    return results

class ContLossforCluster(nn.Module):
    def __init__(
        self,
        temperature=0.07,
        cont_cutoff=False,
        knn_aug=False,
        num_neighbors=5,
        contrastive_clustering=1,
    ):
        super().__init__()
        self.temperature = temperature
        self.contrastive_clustering = contrastive_clustering
        self.cont_cutoff = cont_cutoff
        self.knn_aug = knn_aug
        self.num_neighbors = num_neighbors

    def forward(self, q, k, cluster_idxes=None, preds=None, start_knn_aug=False):
        batch_size = q.shape[0]
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)

        q_and_k = torch.cat([q, k], dim=0)
        l_i = torch.einsum("nc,kc->nk", [q, q_and_k]) / self.temperature

        self_mask = torch.ones_like(l_i, dtype=torch.float)
        self_mask = (
            torch.scatter(self_mask, 1, torch.arange(batch_size).view(-1, 1).cuda(), 0)
            .detach()
            .cuda()
        )

        positive_mask_i = torch.zeros_like(l_i, dtype=torch.float)
        positive_mask_i = (
            torch.scatter(
                positive_mask_i,
                1,
                batch_size + torch.arange(batch_size).view(-1, 1).cuda(),
                1,
            )
            .detach()
            .cuda()
        )

        l_i_exp = torch.exp(l_i)
        l_i_exp_sum = torch.sum((l_i_exp * self_mask), dim=1, keepdim=True)

        loss = -torch.sum(
            torch.log(l_i_exp / l_i_exp_sum) * positive_mask_i, dim=1
        ).mean()

        if cluster_idxes is not None and self.contrastive_clustering:
            cluster_idxes = cluster_idxes.view(-1, 1)
            cluster_idxes_kq = torch.cat([cluster_idxes, cluster_idxes], dim=0)
            mask = torch.eq(cluster_idxes, cluster_idxes_kq.T).float().cuda()

            if self.cont_cutoff:
                preds = preds.detach()
                pred_labels = (preds > 0) * 1
                pred_labels = pred_labels.view(-1, 1)
                pred_labels_kq = torch.cat([pred_labels, pred_labels], dim=0)
                label_mask = torch.eq(pred_labels, pred_labels_kq.T).float().cuda()

                mask = mask * label_mask

            if self.knn_aug and start_knn_aug:
                cosine_corr = q @ q_and_k.T
                if self.num_neighbors > cosine_corr.size(-1):
                    print("smaller than k",cosine_corr.size(-1))
                k_val = min(self.num_neighbors, cosine_corr.size(-1))
                _, kNN_index = torch.topk(
                    cosine_corr, k=k_val, dim=-1, largest=True
                )
                mask_kNN = torch.scatter(
                    torch.zeros(mask.shape).cuda(), 1, kNN_index, 1
                )
                mask = ((mask + mask_kNN) > 0.5) * 1

            mask = mask.float().detach().cuda()
            batch_size = q.shape[0]
            anchor_dot_contrast = torch.div(
                torch.matmul(q, q_and_k.T), self.temperature
            )
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            logits_mask = torch.scatter(
                torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).cuda(), 0
            )
            mask = mask * logits_mask

            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            loss_prot = -mean_log_prob_pos.mean()
            loss += loss_prot

        return loss

class ContLossforCluster_ALL(nn.Module):
    def __init__(
        self,
        temperature=0.07,
        knn_aug=True,
        num_neighbors=10,
    ):
        super().__init__()
        self.temperature = temperature
        self.knn_aug = knn_aug
        self.num_neighbors = num_neighbors

    def forward(self, features, cluster_idxes=None, global_features=None, global_clusters=None):
        batch_size = features.shape[0]
        features = F.normalize(features, dim=1)
        global_features = F.normalize(global_features, dim=1)

        similarity = torch.mm(features, global_features.t()) / self.temperature

        cluster_idxes = cluster_idxes.view(-1, 1)
        positive_mask = (cluster_idxes == global_clusters).float().cuda()

        # 添加kNN正样本
        if self.knn_aug:
            # 计算当前batch与所有样本的相似度
            cosine_sim = similarity.clone()  # 使用已计算的相似度矩阵
            
            # 对每个样本找到最相似的k个样本
            k_val = min(self.num_neighbors, cosine_sim.size(-1))
            _, kNN_index = torch.topk(
                cosine_sim, k=k_val, dim=-1, largest=True
            )
            
            # 创建kNN mask
            knn_mask = torch.zeros_like(positive_mask).cuda()
            knn_mask.scatter_(1, kNN_index, 1.0)
            
            # 合并聚类和kNN的正样本mask
            positive_mask = ((positive_mask + knn_mask) > 0).float()

        # 计算对比损失
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
        
        # 计算正样本的平均log概率
        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / (positive_mask.sum(1) + 1e-12)
        
        # 计算最终损失
        loss = -mean_log_prob_pos.mean()

        return loss

def entropy_regularization(predictions, temperature=0.1, num_bins=6):
        """
        将回归预测值转换为概率分布并计算熵
        Args:
            predictions: 模型预测值
            temperature: softmax温度系数
            num_bins: 将[-3,3]区间分成多少个bin
        """
        # 将预测值离散化到固定区间
        # print(predictions.shape)
        predictions = predictions.squeeze(1)
        bins = torch.linspace(-3, 3, steps=num_bins).cuda()
        binned_preds = torch.zeros(predictions.size(0), num_bins-1).cuda()
        
        # 将每个预测值映射到最近的bin
        for i in range(num_bins-1):
            left, right = bins[i], bins[i+1]
            mask = (predictions >= left) & (predictions < right)
            binned_preds[:, i] = mask.float()
        
        # 计算整个batch的分布
        batch_distribution = torch.mean(binned_preds, dim=0)
        
        # 应用softmax使其更平滑
        smoothed_dist = F.softmax(batch_distribution/temperature, dim=0)
        
        # 计算负熵作为损失（熵越大，损失越小）
        entropy = -torch.mean(smoothed_dist * torch.log(smoothed_dist + 1e-12))
        return -entropy

from torch import Tensor
def diagonal_gaussian_kl_loss(m1: Tensor, v1: Tensor,
                              m2: Tensor, v2: Tensor,
                              eps: float = 0.0,
                              dim_reduction: str = "sum") -> Tensor:
    loss = (v2.log() - v1.log() + (v1 + (m2 - m1).square()) / (v2 + eps) - 1) / 2
    if dim_reduction == "sum":
        return loss.sum()
    elif dim_reduction == "mean":
        return loss.mean()
    else:
        return loss
    # match dim_reduction:
    #     case "sum":
    #         return loss.sum()
    #     case "mean":
    #         return loss.mean()
    #     case "none":
    #         return loss

    #     case _:
    #         raise ValueError(f"Invalid dim_reduction: {dim_reduction!r}")

class WeightEMA(object):
    def __init__(self, alpha, model, ema_model):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        # self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            ema_param.data.copy_(param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)


def compute_kl_loss (p, q,pad_mask=None):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
        
        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss