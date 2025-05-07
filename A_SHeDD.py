import torch
import torch.nn as nn
import sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.utils import shuffle
import backbone_resnet
from backbone_resnet import FC_Classifier_NoLazy_GRL
import time
from sklearn.metrics import f1_score
import torch.nn.functional as F
from torch.autograd import grad
from functions import MyDataset_Unl, MyDataset, cumulate_EMA, transform
import os
from scipy import io
from param import config
from tqdm import tqdm

WEIGHT_DECAY = config['WEIGHT_DECAY']
GP_PARAM = config['GP_PARAM']
DC_PARAM = config['DC_PARAM']
ITER_DC = config['ITER_DC']
ITER_CLF = config['ITER_CLF']
ALPHA = config['ALPHA']
TRAIN_BATCH_SIZE = config['TRAIN_BATCH_SIZE']
data_names = config['data_names']
ds = config['ds']
LEARNING_RATE = config['LEARNING_RATE']
LEARNING_RATE_DC = config['LEARNING_RATE_DC']
EPOCHS = config['EPOCHS']
MOMENTUM_EMA = config['MOMENTUM_EMA']
WARM_UP_EPOCH_EMA = config['WARM_UP_EPOCH_EMA']
TH_FIXMATCH = config['TH_FIXMATCH']

def evaluation(model, dataloader, device):
    model.eval()
    tot_pred = []
    tot_labels = []
    for x_batch, y_batch in dataloader:
        if y_batch.shape[0] == TRAIN_BATCH_SIZE:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = None
            _, _ ,_, pred = model.forward_test_target(x_batch)
            pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
            tot_pred.append( pred_npy )
            tot_labels.append( y_batch.cpu().detach().numpy())
    tot_pred = np.concatenate(tot_pred)
    tot_labels = np.concatenate(tot_labels)
    return tot_pred, tot_labels

def gradient_penalty(critic, h_s, h_t, device):
    alpha = torch.rand(h_s.size(0), 1).to(device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def train_and_eval(ds_path, out_dir, nsamples, nsplit, ds_idx, source_idx, gpu):

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    source_prefix = data_names[ds[ds_idx]][source_idx]
    target_prefix = data_names[ds[ds_idx]][source_idx - 1]

    source_data = np.load(os.path.join(ds_path, ds[ds_idx], f"{source_prefix}_data_filtered.npy"))
    target_data = np.load(os.path.join(ds_path, ds[ds_idx], f"{target_prefix}_data_filtered.npy"))
    source_label = np.load(os.path.join(ds_path, ds[ds_idx], f"{source_prefix}_label_filtered.npy"))
    target_label = np.load(os.path.join(ds_path, ds[ds_idx], f"{target_prefix}_label_filtered.npy"))

    train_target_idx = np.load( os.path.join(ds_path, ds[ds_idx], "train_idx", f"{target_prefix}_{nsplit}_{nsamples}_train_idx.npy") )
    test_target_idx = np.setdiff1d(np.arange(target_data.shape[0]), train_target_idx)

    train_target_data = target_data[train_target_idx]
    train_target_label = target_label[train_target_idx]

    test_target_data = target_data[test_target_idx]
    test_target_label = target_label[test_target_idx]

    test_target_data_unl = target_data[test_target_idx]

    n_classes = len(np.unique(source_label))

    sys.stdout.flush()

    TR_BATCH_SIZE = np.minimum(int(n_classes * nsamples), TRAIN_BATCH_SIZE)
    TR_BATCH_SIZE = int(TR_BATCH_SIZE)

    source_data, source_label = shuffle(source_data, source_label)
    train_target_data, train_target_label = shuffle(train_target_data, train_target_label)

    #DATALOADER SOURCE
    x_train_source = torch.tensor(source_data, dtype=torch.float32)
    y_train_source = torch.tensor(source_label, dtype=torch.int64)

    #dataset_source = TensorDataset(x_train_source, y_train_source)
    dataset_source = MyDataset(x_train_source, y_train_source, transform=transform)
    dataloader_source = DataLoader(dataset_source, shuffle=True, batch_size=TR_BATCH_SIZE)

    #DATALOADER TARGET TRAIN
    x_train_target = torch.tensor(train_target_data, dtype=torch.float32)
    y_train_target = torch.tensor(train_target_label, dtype=torch.int64)

    dataset_train_target = MyDataset(x_train_target, y_train_target, transform=transform)
    dataloader_train_target = DataLoader(dataset_train_target, shuffle=True, batch_size=TR_BATCH_SIZE//2)

    #DATALOADER TARGET UNLABELLED
    x_train_target_unl = torch.tensor(test_target_data_unl, dtype=torch.float32)

    dataset_train_target_unl = MyDataset_Unl(x_train_target_unl, transform)
    dataloader_train_target_unl = DataLoader(dataset_train_target_unl, shuffle=True, batch_size=TR_BATCH_SIZE//2)

    #DATALOADER TARGET TEST
    x_test_target = torch.tensor(test_target_data, dtype=torch.float32)
    y_test_target = torch.tensor(test_target_label, dtype=torch.int64)
    dataset_test_target = TensorDataset(x_test_target, y_test_target)
    dataloader_test_target = DataLoader(dataset_test_target, shuffle=False, batch_size=TRAIN_BATCH_SIZE)

    model = backbone_resnet.SHeDD(input_channel_source=source_data.shape[1], input_channel_target=target_data.shape[1], num_classes=n_classes)
    model = model.to(device)

    te_model = backbone_resnet.TeacherModel(input_channel_source=source_data.shape[1], input_channel_target=target_data.shape[1],
                            num_classes=n_classes)
    te_model = te_model.to(device)

    domain_cl = FC_Classifier_NoLazy_GRL(256, 2, ALPHA)
    domain_cl = domain_cl.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
    optimizer_cl = torch.optim.AdamW(domain_cl.parameters(), lr=LEARNING_RATE_DC)

    pbar = tqdm(range(EPOCHS))

    ema_weights = None

    for epoch in pbar:
        pbar.set_description('Epoch %d/%d' % (epoch + 1, EPOCHS))
        start = time.time()
        model.train()
        tot_loss = 0.0
        tot_ortho_loss = 0.0
        tot_fixmatch_loss = 0.0
        den = 0

        for x_batch_source, y_batch_source in dataloader_source:
            if x_batch_source.shape[0] < TR_BATCH_SIZE:
                continue  # To avoid errors on pairing source/target samples

            optimizer.zero_grad()
            x_batch_target, y_batch_target = next(iter(dataloader_train_target))
            x_batch_target_unl, x_batch_target_unl_aug = next(iter(dataloader_train_target_unl))

            x_batch_source = x_batch_source.to(device)
            y_batch_source = y_batch_source.to(device)

            x_batch_target = x_batch_target.to(device)
            y_batch_target = y_batch_target.to(device)

            x_batch_target_unl = x_batch_target_unl.to(device)
            x_batch_target_unl_aug = x_batch_target_unl_aug.to(device)

            # TRAIN DISCRIMINATOR
            set_requires_grad(model, requires_grad=False)
            set_requires_grad(domain_cl, requires_grad=True)
            with torch.no_grad():
                h_s, _, _, _ = model.forward_source(x_batch_source, 0)
                h_t, _, _, _ = model.forward_source(torch.cat((x_batch_target, x_batch_target_unl_aug), 0), 1)

            for _ in range(ITER_DC):

                h_s_grl = domain_cl(h_s)
                h_t_grl = domain_cl(h_t)

                h_grl = torch.cat([h_s_grl, h_t_grl], 0)

                y_dom_grl = torch.cat([torch.zeros(TR_BATCH_SIZE, dtype=torch.long, device=device),
                                             torch.ones(TR_BATCH_SIZE, dtype=torch.long, device=device)], dim=0)

                loss_cl = loss_fn(h_grl, y_dom_grl)

                optimizer_cl.zero_grad()
                loss_cl.backward()
                optimizer_cl.step()

            # Train classifier
            set_requires_grad(model, requires_grad=True)
            set_requires_grad(domain_cl, requires_grad=False)

            emb_source_inv, emb_source_spec, dom_source_cl, task_source_cl, emb_target_inv, emb_target_spec, dom_target_cl, task_target_cl = model(
                [x_batch_source, x_batch_target])

            pred_task = torch.cat([task_source_cl, task_target_cl], dim=0)
            pred_dom = torch.cat([dom_source_cl, dom_target_cl], dim=0)
            y_batch = torch.cat([y_batch_source, y_batch_target], dim=0)
            y_batch_dom = torch.cat([torch.zeros_like(y_batch_source), torch.ones_like(y_batch_target)], dim=0)

            loss_pred = loss_fn(pred_task, y_batch)
            loss_dom = loss_fn( pred_dom, y_batch_dom)


            ############### TEACHER MODEL ###############
            model.target.train()
            unl_target_inv, unl_target_spec, pred_unl_target_dom, pred_unl_target = model.forward_source(x_batch_target_unl, 1)
            unl_target_aug_inv, unl_target_aug_spec, pred_unl_target_strong_dom, pred_unl_target_strong = model.forward_source(x_batch_target_unl_aug, 1)

            # Prediction with teacher network
            pred_unl_target_te = te_model(x_batch_target_unl_aug)

            with torch.no_grad():
                pseudo_labels = torch.softmax(pred_unl_target_te, dim=1)
                max_probs, targets_u = torch.max(pseudo_labels, dim=1)
                mask = max_probs.ge(TH_FIXMATCH).float()

            u_pred_loss = (F.cross_entropy(pred_unl_target_strong, targets_u, reduction="none") * mask).mean()

            pred_unl_dom = torch.cat([pred_unl_target_strong_dom,pred_unl_target_dom],dim=0)
            u_loss_dom = loss_fn(pred_unl_dom, torch.ones(pred_unl_dom.shape[0]).long().to(device))

            inv_emb = torch.cat([emb_source_inv, emb_target_inv])
            spec_emb = torch.cat([emb_source_spec, emb_target_spec])
            unl_inv = torch.cat([unl_target_inv,unl_target_aug_inv],dim=0)
            unl_spec = torch.cat([unl_target_spec,unl_target_aug_spec],dim=0)

            norm_inv_emb = nn.functional.normalize(inv_emb)
            norm_spec_emb = nn.functional.normalize(spec_emb)
            norm_unl_inv = F.normalize(unl_inv)
            norm_unl_spec = F.normalize(unl_spec)

            loss_ortho = torch.mean(torch.sum( norm_inv_emb * norm_spec_emb, dim=1))
            u_loss_ortho = torch.mean( torch.sum( norm_unl_inv * norm_unl_spec, dim=1) )

            emb_t_all = torch.cat((emb_target_inv, unl_target_aug_inv), dim=0)  # all target embeddings (labelled + unlabelled)

            dom_source_cl_grl = domain_cl(emb_source_inv)
            dom_target_cl_grl = domain_cl(emb_t_all)
            pred_dom_grl = torch.cat([dom_source_cl_grl, dom_target_cl_grl], 0)
            y_batch_dom_grl = torch.cat([torch.zeros(TR_BATCH_SIZE, dtype=torch.long, device=device),
                                         torch.ones(TR_BATCH_SIZE, dtype=torch.long, device=device)], dim=0)

            loss_cl_pred = loss_fn(pred_dom_grl, y_batch_dom_grl)

            loss = loss_pred + loss_dom + loss_ortho + u_loss_dom + u_loss_ortho + loss_cl_pred + u_pred_loss

            loss.backward() # backward pass: backpropagate the prediction loss
            optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass

            # Update teacher's weights using EMA
            with torch.no_grad():
                for param1, param2 in zip(model.get_target_and_task_weights()[0], te_model.get_weights()[0]):  # target model weights
                    param2.data = MOMENTUM_EMA * param2.data + (1 - MOMENTUM_EMA) * param1.data

                for param1, param2 in zip(model.get_target_and_task_weights()[1], te_model.get_weights()[1]):  # task classifier weights
                    param2.data = MOMENTUM_EMA * param2.data + (1 - MOMENTUM_EMA) * param1.data

            tot_loss+= loss.cpu().detach().numpy()
            tot_ortho_loss+=loss_ortho.cpu().detach().numpy()
            tot_fixmatch_loss+=u_pred_loss.cpu().detach().numpy()
            den+=1.

        end = time.time()
        pred_valid, labels_valid = evaluation(model, dataloader_test_target, device)
        f1_val = f1_score(labels_valid, pred_valid, average="weighted")
        
        ####################### EMA #####################################
        f1_val_ema = 0
        if epoch >= WARM_UP_EPOCH_EMA:
            ema_weights = cumulate_EMA(model, ema_weights, MOMENTUM_EMA)
            current_state_dict = model.state_dict()
            model.load_state_dict(ema_weights)
            pred_valid, labels_valid = evaluation(model, dataloader_test_target, device)
            f1_val_ema = f1_score(labels_valid, pred_valid, average="weighted")
            f1_val_nw = f1_score(labels_valid, pred_valid, average=None)
            model.load_state_dict(current_state_dict)
        ####################### EMA #####################################
        
        pbar.set_postfix(
            {'Loss': tot_loss/den, 'F1 (ORIG)': 100*f1_val, 'F1 (EMA)': 100*f1_val_ema, 'Time': (end-start)})
        sys.stdout.flush()

    model_dir = os.path.join(out_dir, "models", source_prefix)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    output_file = os.path.join( model_dir, f"{source_prefix}_{nsplit}_{nsamples}.pth" )
    model.load_state_dict(ema_weights)
    torch.save(model.state_dict(), output_file)

    return 100 * f1_val_ema, 100 * f1_val_nw
