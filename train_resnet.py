import numpy as np
import torch
import torch.nn.functional as F

import datasets
import models
from instrumentation import compute_metrics
import losses
import os
from cam_functions import *
from tqdm import tqdm

from torch.optim import lr_scheduler

def run_train_resnet(P):
    
    
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

    model = models.ImageClassifier(P)
    
    feature_extractor_params = [param for param in list(model.feature_extractor.parameters()) if param.requires_grad]
    linear_classifier_params = [param for param in list(model.linear_classifier.parameters()) if param.requires_grad]
    opt_params = [
        {'params': feature_extractor_params, 'lr' : P['lr']},
        {'params': linear_classifier_params, 'lr' : P['lr_mult'] * P['lr']}
    ]
  
    if P['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(opt_params, lr=P['lr'])
    elif P['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(opt_params, lr=P['lr'], momentum=0.9, weight_decay=0.001)
    
    # training loop
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

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
                
                train_loss = 0

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
                    
                    logits, cam_w = model(image)
                    if logits.dim() == 1:
                        logits = torch.unsqueeze(logits, 0)
                    preds = torch.sigmoid(logits)   
                    
                    if phase == 'train':

                        image_strong = batch['image_strong'].to(device, non_blocking=True)
                        logits_s, _ = model(image_strong)

                        # make cam-based masked image
                        if label_vec_obs.size(0) != label_vec_obs.sum().item():
                            masked_image = make_nus_resnet_cam_mask(image, cam_w, label_vec_obs, P,device)
                        else:
                            masked_image = make_resnet_cam_mask(image, cam_w, label_vec_obs, P)
            
                        mask_logits, _ = model(masked_image)
                        gap = torch.abs(logits - mask_logits).clone().detach()
                        logits_cp = logits.clone().detach()
                        
                        # compute logit gap threshold & ensured candidate selection threshold
                        sp_gap_dist = 0.999*sp_gap_dist + 0.001*gap[label_vec_obs.type(torch.bool)].mean()
                        unannotated_dist = 0.999*unannotated_dist + 0.001*(logits_cp.sum(dim=0)/batch["label_vec_obs"].size(0))
                        
                        # compute bacth loss
                        loss_tensor = losses.batch_loss_spbc_ran_l2(logits, mask_logits, logits_s, label_vec_obs, P, sp_gap_dist, unannotated_dist, epoch)
                        
                        loss_tensor.backward()
                        optimizer.step()
                        
                        
                    else:
                        
                        preds_np = preds.cpu().numpy()
                        this_batch_size = preds_np.shape[0]
                        y_pred[batch_stack : batch_stack+this_batch_size] = preds_np
                        y_true[batch_stack : batch_stack+this_batch_size] = label_vec_true
                        batch_stack += this_batch_size
                        
                        
            if phase == 'train':


                print("\nepoch {} train loss : {}\n".format(epoch, train_loss))

                # print("unannotated dist : \n", unannotated_dist)
                print("sp_gap_dist : \n", sp_gap_dist, "\n")
                
                

                   
            if phase != 'train':
                metrics = compute_metrics(y_pred, y_true)
                del y_pred
                del y_true
                map_val = metrics['map']
                

        print(" Epoch {} : val mAP {:.3f} \n".format(epoch,map_val))
        
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

            logits, _  = model(image)
            
            
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

