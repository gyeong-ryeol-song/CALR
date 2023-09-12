import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F

import datasets
import models
from instrumentation import compute_metrics
import losses
import datetime
import os
from cam_functions import *
from tqdm import tqdm
import wandb  
import random
import pandas as pd
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import q2l._init_paths
from q2l.lib.models_q2.query2label import build_q2l
from torch.optim import lr_scheduler
from sklearn.manifold import TSNE 
def run_train_q2l(P,args):
    
    
    dataset = datasets.get_data(P)
    dataloader = {}
    for phase in ['train', 'val', 'test']:
        
        dataloader[phase] = torch.utils.data.DataLoader(
            dataset[phase],
            batch_size = P['bsize'],
            shuffle = phase == 'train',
            sampler = None,
            num_workers = P['num_workers'],
            drop_last = False,
            pin_memory = True
        )

    args.num_class = P["num_classes"]
    model = build_q2l(args)
    model = model.cuda()

    args.lr_mult = args.bsize / 256
    if args.optim == 'AdamW':
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
        ]
        optimizer = getattr(torch.optim, args.optim)(
            param_dicts,
            args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
        )
      
    else:
        raise NotImplementedError
    # training loop
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(dataloader["train"]), epochs=args.num_epochs, pct_start=0.2)
    
    # model = model.to(device)
    bestmap_val = 0
    bestmap_test = 0

## logit gap threshold & ensured candidate selection parameter########

    unannotated_dist = torch.zeros(1,P["num_classes"]).to(device, non_blocking=True)
    sp_gap_dist = 0

######################################################################    

    for epoch in range(1, P['num_epochs']+1):
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()

            else:
                model.eval()
                y_pred = np.zeros((len(dataset[phase]), P['num_classes']))
                y_true = np.zeros((len(dataset[phase]), P['num_classes']))
                batch_stack = 0


            with torch.set_grad_enabled(phase == 'train'):
                
                for batch in tqdm(dataloader[phase]):
                    
                   
                    image = batch['image'].to(device, non_blocking=True)
                    label_vec_obs = batch['label_vec_obs'].to(device, non_blocking=True)
                  
                    label_vec_true = batch['label_vec_true'].clone().numpy()

                    
                    # Forward pass
                    optimizer.zero_grad()
                    
                    logits, cnn_feat_w, atts_w = model(image)
                    
                    # import pdb;pdb.set_trace()
                    if logits.dim() == 1:
                        logits = torch.unsqueeze(logits, 0)
                    preds = torch.sigmoid(logits)   
                    
                    if phase == 'train':
                        
                        imgs = batch["image_raw"]
                        image_strong = batch['image_strong'].to(device, non_blocking=True)
                        
                        logits_s, cnn_feat_s, atts_s = model(image_strong)
                        
                        
                        # feat_s = feat_s.squeeze()
                        
                        # make cam-based masked image
                        img_w = image.clone().detach()
                        if label_vec_obs.size(0) != label_vec_obs.sum().item():
                            masked_image = nus_cam_mask(img_w,atts_w,cnn_feat_w, label_vec_obs, P, device)
                        # if label_vec_obs.sum() != batch[""]
                        else:
                            cam = make_cam(atts_w, cnn_feat_w, label_vec_obs, P) # (batch, 14, 14)
                            masked_image = make_cam_mask(img_w, cam, P)
                       
                        # input mask image to model
                        mask_logits, cnn_feat_m, atts_m = model(masked_image)
                        
                        #### logit gap threshold & unsured candidate selection threshold update ###
                        gap = torch.abs(logits - mask_logits).clone().detach()
                        logits_cp = logits.clone().detach()
                        sp_gap_dist = 0.999*sp_gap_dist + 0.001*gap[label_vec_obs.type(torch.bool)].mean()
                        unannotated_dist = 0.999*unannotated_dist + 0.001*(logits_cp.sum(dim=0)/batch["label_vec_obs"].size(0))
                        ###########################################################################
                        
                        # compute batch loss
                        loss_tensor = losses.batch_loss_spbc_ran_l2(logits, mask_logits, logits_s, label_vec_obs, P, sp_gap_dist, unannotated_dist, epoch)
                        
                        
                        loss_tensor.backward()
                        optimizer.step()
                        scheduler.step()
                        


                    else:
                        preds_np = preds.cpu().numpy()
                        this_batch_size = preds_np.shape[0]
                        y_pred[batch_stack : batch_stack+this_batch_size] = preds_np
                        y_true[batch_stack : batch_stack+this_batch_size] = label_vec_true
                        batch_stack += this_batch_size
 
                   
            if phase != 'train':
                metrics = compute_metrics(y_pred, y_true)
                del y_pred
                del y_true
                map_val = metrics['map']
                


        print("Epoch {} : val mAP {:.3f} \n".format(epoch,map_val))
        
        if bestmap_val < map_val:
            bestmap_val = map_val
            bestmap_epoch = epoch
            print(f'Saving model weight for best val mAP {bestmap_val:.3f}')
            path = os.path.join(P['save_path'], '{}_{}_ct:{}_warm:{}_bb:{}.pt'.format(P['mode'],P["dataset"],P["cam_threshold"],P["warm_up"],P["backbone"]))
            torch.save((model.state_dict(), P), path)
        elif bestmap_val - map_val > 2:
            print('Early stopped.')
            break
    

 
    model_state, _ = torch.load(path)
    model.load_state_dict(model_state)
    model = model.cuda()
    phase = 'test'
    model.eval()
    y_pred = np.zeros((len(dataset[phase]), P['num_classes']))
    y_true = np.zeros((len(dataset[phase]), P['num_classes']))
    batch_stack = 0

    with torch.set_grad_enabled(phase == 'train'):
        for batch in tqdm(dataloader[phase]):
            # Move data to GPU
            image = batch['image'].to(device, non_blocking=True)

            label_vec_obs = batch['label_vec_obs'].to(device, non_blocking=True)
            label_vec_true = batch['label_vec_true'].clone().numpy()
            # Forward pass
            optimizer.zero_grad()

            logits, cnn_feat, atts  = model(image)
            
            
            if logits.dim() == 1:
                logits = torch.unsqueeze(logits, 0)
            preds = torch.sigmoid(logits)
  
            preds_np = preds.cpu().numpy()
            this_batch_size = preds_np.shape[0]
            y_pred[batch_stack : batch_stack+this_batch_size] = preds_np
            y_true[batch_stack : batch_stack+this_batch_size] = label_vec_true
            batch_stack += this_batch_size

    metrics = compute_metrics(y_pred, y_true)
    map_test = metrics['map']

    print('Training procedure completed!')
    print(f'Test mAP : {map_test:.3f} when trained until epoch {bestmap_epoch}')

