import torch
from torch import nn
from models import model as mm
from utils.util import *
from utils.pseudo_label import *
import torch.optim as optim
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from utils.vMF import vFM_cluster
from utils.align_loss import get_cov,DARE_GRAM_LOSS
from utils.sinkhorn import SinkhornKnopp
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from utils.mutual_information import adapt_batch

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = torch.load(hyp_params.pretrained_model,map_location=torch.device('cuda'))
    ema_model = torch.load(hyp_params.pretrained_model,map_location=torch.device('cuda'))
    for param in ema_model.parameters():
        param.detach_()
    ema_optimizer= WeightEMA(0.95, model, ema_model)

    # for name, para in model.named_parameters():
    #     print(name)
    if hyp_params.fix_para:
        model = fix_para_new(model)
        # model = fix_para(model)
    count_parameters(model)
    if hyp_params.use_cuda:
        model = model.cuda()
    # optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    # criterion = nn.CosineSimilarity(dim=1).cuda()
    # scheduler = ReduceLROnPlateau(
    #     optimizer, mode="min", patience=hyp_params.when, factor=0.1, verbose=True
    # )
    optimizer,scheduler=get_optimizer(model,hyp_params,train_loader)
    settings = {
        "model": model,
        "ema_model": ema_model,
        "optimizer": optimizer,
        "criterion": criterion,
        "scheduler": scheduler,
        "optimizer_ema": ema_optimizer,
    }
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)

def get_optimizer(model,hyp_params,train_loader):
        warmp_proportion=0.2
        len_data=len(train_loader.dataset)
        # 计算总训练步数，使用总epoch数而不仅是warmup_epochs
        total_epochs = hyp_params.num_epochs
        num_train_optimization_steps = int(len_data / hyp_params.batch_size) * total_epochs
        num_warmup_steps = int(warmp_proportion*num_train_optimization_steps)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=hyp_params.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        return optimizer,scheduler



def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings["model"]
    ema_model = settings["ema_model"]
    optimizer = settings["optimizer"]
    criterion = settings["criterion"]
    scheduler = settings["scheduler"]
    optimizer_ema = settings["optimizer_ema"]
    tune_cri = ContrastiveLoss()
    # cluster_contri_loss = ContLossforCluster_ALL(
    #     temperature=0.5,
    #     knn_aug=False,
    #     num_neighbors=hyp_params.num_neighbors,
    # )
    cluster_contri_loss=ContLossforCluster(
            temperature=0.1,
            cont_cutoff=False,
            knn_aug=True,
            num_neighbors=hyp_params.num_neighbors,
            )
    global centroids
    centroids = None
    src_features = torch.load(f"{hyp_params.pretrained_model}_train_representations_hidden.pt")
    cluster_fnc = vFM_cluster(contrast_dim=120, temperature=0.1, num_classes=hyp_params.num_cluster,src_features=src_features)
    cluster_fnc_hidden = vFM_cluster(contrast_dim=40, temperature=0.1, num_classes=hyp_params.num_cluster,src_features=src_features)
    # cluster_fnc = vFM_cluster(contrast_dim=120, temperature=0.07, num_classes=hyp_params.num_cluster)
    src_cov = torch.load(f"{hyp_params.pretrained_model}_final_cov.pt")
    global pseudo_labels
    pseudo_labels = torch.zeros((len(train_loader.dataset), hyp_params.num_cluster)).cuda()
    global filter_mask
    filter_mask = torch.zeros(len(train_loader.dataset)).cuda()
    

    def train(model, optimizer, optimizer_ema, criterion, tune_cri,cluster_contri_loss,cluster_fnc,src_cov,epoch):
        model.train()
        loss_all=0
        loss_mi=0
        loss_rdrop=0
        loss_cluster=0
        
        cluster_fnc._hook_before_epoch(epoch, hyp_params.num_epochs)
        
        num_filtered = torch.sum(filter_mask).item()
        print(f"Number of filtered samples: {num_filtered} of {len(train_loader.dataset)}")

        for i_batch, batch in enumerate(train_loader):
            idx = batch["idx"]
            text, audio, vision, batch_Y = (
                batch["text"],
                batch["audio"],
                batch["vision"],
                batch["label"],
            )
            
            text_aug, audio_aug, vision_aug = random_drop(text, audio, vision)
            eval_attr = batch_Y.unsqueeze(-1)
            model.zero_grad()

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = (
                        text.cuda(),
                        audio.cuda(),
                        vision.cuda(),
                        eval_attr.cuda(),
                    )
                    text_aug, audio_aug, vision_aug = (
                        text_aug.cuda(),
                        audio_aug.cuda(),
                        vision_aug.cuda(),
                    )

            batch_size = text.size(0)
            # if batch_size<24:
            #     print("small_batchsize",batch_size)
            # if batch_size<10:
            #     continue
            net = nn.DataParallel(model) if batch_size > 10 else model
            if isinstance(net, nn.DataParallel):
                preds, rep0, rep1, rep2 = net.module.forward_multi_layer([text, audio, vision])
                preds_, rep_0, rep_1, rep_2 = net.module.forward_multi_layer([text, audio, vision])
                preds_aug, rep_aug_0,rep_aug_1, rep_aug_2 = net.module.forward_multi_layer([text_aug, audio_aug, vision_aug])
            else:
                preds, rep0, rep1, rep2 = net.forward_multi_layer([text, audio, vision])
                preds_, rep_0, rep_1, rep_2 = net.forward_multi_layer([text, audio, vision])
                preds_aug, rep_aug_0,rep_aug_1, rep_aug_2 = net.forward_multi_layer([text_aug, audio_aug, vision_aug])
            
            # print("rep0 shape:", rep0.shape)
            # print("rep1 shape:", rep1.shape) 
            # print("rep2 shape:", rep2.shape)
            rep = torch.cat([rep0, rep1, rep2], dim=1)
            # print("rep shape:", rep.shape)
            rep_ = torch.cat([rep_0, rep_1, rep_2], dim=1)
            rep_aug = torch.cat([rep_aug_0, rep_aug_1, rep_aug_2], dim=1)
            #get cov of batch
            batch_cov0 = get_cov(rep0)
            batch_cov1 = get_cov(rep1)
            batch_cov2 = get_cov(rep2)
            # loss_align0 = DARE_GRAM_LOSS(src_cov,batch_cov0)
            # loss_align1 = DARE_GRAM_LOSS(src_cov,batch_cov1)
            # loss_align2 = DARE_GRAM_LOSS(src_cov,batch_cov2)
            
            if epoch<0:
                # loss = tune_cri(rep, rep_aug)+loss_align
                
                loss = tune_cri(rep, rep_aug)
            else:
                #get batch pseudo labels and mask
                _, batch_pseudo_labels = torch.max(pseudo_labels[idx], dim=1)  # Get indices instead of values
                batch_filter_mask = filter_mask[idx]
                # print("batch_pseudo_labels",batch_pseudo_labels)
                # print("batch_filter_mask",batch_filter_mask)

                #vMF cluster
                sinkhorn = SinkhornKnopp(num_iters_sk=3,epsilon_sk=0.05,imb_factor=1)  #do imbalanced problem
                # Convert labels to int64 type
                batch_pseudo_labels = batch_pseudo_labels.long()  # Ensure labels are int64
                cluster_logits = cluster_fnc(rep.view(-1, 120),batch_pseudo_labels.view(-1),batch_filter_mask.bool().view(-1))
                cluster_logits_ = cluster_fnc(rep_.view(-1, 120),batch_pseudo_labels.view(-1),batch_filter_mask.bool().view(-1))
                cluster_logits_aug = cluster_fnc(rep_aug.view(-1, 120),batch_pseudo_labels.view(-1),batch_filter_mask.bool().view(-1))
                
                #rdrop_loss
                rdrop_loss = compute_kl_loss(cluster_logits,cluster_logits_)
                # +compute_kl_loss(cluster_logits,cluster_logits_aug)
                # print("cluster_logits",cluster_logits)
                # cluster_logits = cluster_fnc(rep.view(-1, 120))
                if hyp_params.use_sk:
                    cluster_logits_sk = sinkhorn(cluster_logits.clone()).view(batch_size, -1)
                    cluster_logits_sk_ = sinkhorn(cluster_logits_.clone()).view(batch_size, -1)
                    cluster_logits_sk_aug = sinkhorn(cluster_logits_aug.clone()).view(batch_size, -1)
                else:
                    cluster_logits_sk = cluster_logits.clone()
                    cluster_logits_sk_ = cluster_logits_.clone()
                    cluster_logits_sk_aug = cluster_logits_aug.clone()
                
                # cluster_logits = cluster_logits.view(batch_size, -1)
                probs,cluster_idxes = torch.max(cluster_logits_sk, dim=-1)

                #different scale cluster
                # cluster_logits0 = cluster_fnc(rep0.view(-1, 40),batch_pseudo_labels.view(-1),batch_filter_mask.bool().view(-1))
                # cluster_logits1 = cluster_fnc(rep1.view(-1, 40),batch_pseudo_labels.view(-1),batch_filter_mask.bool().view(-1))
                # cluster_logits2 = cluster_fnc(rep2.view(-1, 40),batch_pseudo_labels.view(-1),batch_filter_mask.bool().view(-1))
                #calculate mi loss
                
                cluster_logits_sk_avg = (cluster_logits_sk + cluster_logits_sk_ + cluster_logits_sk_aug) / 3
                mi_loss = adapt_batch(cluster_logits_sk_avg,hyp_params.num_cluster)
                # +adapt_batch(cluster_logits1,hyp_params.num_cluster)+adapt_batch(cluster_logits2,hyp_params.num_cluster)


                #pseudo labels update
                pseudo_labels[idx] = cluster_logits_sk_avg.detach()  # Make sure to detach to prevent gradient issues
                probs_all,cluster_idxes_all = torch.max(pseudo_labels, dim=-1)
                
                #pseudo labels update
                # Calculate ratio based on epoch, starting from 0.0 and increasing to 0.9
                ratio = min((epoch - 5) * 0.05, 0.5)
                selected_ids = []
                # Calculate thresholds for each class based on ratio
                n_classes = pseudo_labels.shape[1]
                thresholds = []
                
                #generate thresholds
                for c in range(n_classes):
                    # Get probabilities for this class
                    class_probs = probs_all[cluster_idxes_all == c]
                    if len(class_probs) > 0:
                        # Sort probabilities and find threshold at ratio percentile
                        sorted_probs = torch.sort(class_probs, descending=True)[0]
                        threshold_idx = int(len(sorted_probs) * ratio)
                        if threshold_idx > 0:
                            threshold = sorted_probs[threshold_idx-1]
                        else:
                            threshold = 1.0
                    else:
                        threshold = 1.0
                    thresholds.append(threshold)
                thresholds = torch.tensor(thresholds).to(probs_all.device)
                # print("thresholds",thresholds)
                # print("probs_all",probs_all)
                # Filter samples based on class-specific thresholds
                selected_mask = probs >= thresholds[cluster_idxes]
                # print("selected_mask",selected_mask)
                selected_mask = selected_mask.float()  # Convert boolean mask to float
                filter_mask[idx] = selected_mask  # Assign float mask to float tensor
                
                
                cluster_loss = cluster_contri_loss(
                    rep,
                    rep_aug,
                    cluster_idxes,
                    start_knn_aug=(epoch>0))
            #     loss_cluster = cluster_contri_loss(
            #     features=rep,
            #     cluster_idxes=cluster_idxes,
            #     global_features=global_features,
            #     global_clusters=global_clusters
            # )
                # loss = cluster_loss+rdrop_loss+mi_loss
                loss = cluster_loss+rdrop_loss+mi_loss
                loss_cluster+=cluster_loss
                loss_rdrop+=rdrop_loss
                loss_mi+=mi_loss
            
            loss_all+=loss.item()
            loss.backward()
            # optimizer.step()
            # scheduler.step()
            


            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            scheduler.step()
            optimizer_ema.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")
            
        print("loss_cluster",loss_cluster)
        print("loss_rdrop",loss_rdrop)
        print("loss_mi",loss_mi)
        print(f"loss of epoch{epoch} is {loss_all}")

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        truths = []

        with torch.no_grad():
            for i_batch, batch in enumerate(loader):
                text, audio, vision, batch_Y = (
                    batch["text"],
                    batch["audio"],
                    batch["vision"],
                    batch["label"],
                )
                eval_attr = batch_Y.unsqueeze(-1)

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = (
                            text.cuda(),
                            audio.cuda(),
                            vision.cuda(),
                            eval_attr.cuda(),
                        )

                batch_size = text.size(0)

                net = nn.DataParallel(model) if batch_size > 10 else model
                preds, _ = net([text, audio, vision])
                # print(preds)
                total_loss += criterion(preds, eval_attr).item()

                # Collect the results into dictionary
                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths
    
    def save_checkpoints():
        results = []
        with torch.no_grad():
            for idx, batch in enumerate(train_loader):
                text, audio, vision = batch["text"], batch["audio"], batch["vision"]

                if hyp_params.use_cuda:
                    text, audio, vision = text.cuda(), audio.cuda(), vision.cuda()

                preds, _ = model([text, audio, vision])
                results.append(preds)
        results = torch.cat(results).squeeze(-1)
        return results

    best_acc = 0
    best_acc_ema=0
    acc_test = 0
    re = save_checkpoints()
    checkpoint = [re.unsqueeze(0)]
    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        # features_all=compute_features(model)
        train(model, optimizer, optimizer_ema, criterion, tune_cri,cluster_contri_loss,cluster_fnc,src_cov,epoch)
        val_loss, r, t = evaluate(model, criterion, test=False)
        acc2 = eval_senti(r, t)

        end = time.time()
        duration = end - start
        # scheduler.step(val_loss)  # Decay learning rate by validation loss

        print("-" * 50)
        print(
            "Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f}".format(
                epoch, duration, val_loss
            )
        )
        print("-" * 50)

        if best_acc < acc2:
            print(f"Saved model at {hyp_params.name}!")
            torch.save(model, hyp_params.name)
            best_acc = acc2
            test_loss, r_test, t_test = evaluate(model, criterion, test=True)
            acc_test = eval_senti(r_test, t_test)
            print("acc_test",acc_test)

        if epoch % hyp_params.intere == 0:
            results = save_checkpoints()
            checkpoint.append(results.unsqueeze(0))
        
        # #测试 ema 性能
        # print("eval of ema model_valid")
        # test_loss_ema, r_valid_ema, t_valid_ema = evaluate(ema_model, criterion, test=False)
        # acc_valid_ema = eval_senti(r_valid_ema,t_valid_ema)
        # if best_acc_ema < acc_valid_ema:
        #     print(f"Saved model at {hyp_params.name}!")
        #     # torch.save(model, hyp_params.name)
        #     best_acc = acc_valid_ema
        #     test_loss, r_test, t_test = evaluate(ema_model, criterion, test=True)
        #     print("eval of ema model_test")
        #     acc_test = eval_senti(r_test, t_test)
        #     print("acc_test_ema",acc_test)
    checkpoint = torch.cat(checkpoint).cpu()
    print(checkpoint.shape)
    print("best_acc",best_acc)
    torch.save(checkpoint, hyp_params.pseudolabel)
