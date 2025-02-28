"""
`engine.py` contains the implementation of training and evaluation loops.
"""

import os
os.environ["MKL_NUM_THREADS"] = "2" 
os.environ["NUMEXPR_NUM_THREADS"] = "2" 
os.environ["OMP_NUM_THREADS"] = "2"

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast

from tqdm import tqdm
import numpy as np

import wandb

from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@torch.no_grad()
def zscore_norm_reverse(img, mean, std):
    """ Reverse Z-score normalization """
    # Input: numpy array [H, W, C]
    # mean: [C]
    # std: [C]
    # Output: numpy array [H, W, C]
    mean_expanded = np.expand_dims(mean, axis=(0, 1))
    std_expanded = np.expand_dims(std, axis=(0, 1))

    return img * std_expanded + mean_expanded


@torch.no_grad()
def visualize(sample: dict,
              config: dict):
    # Visualize original and reconstructed images
    # TODO: should we use the same sample[0] for every epoch?
    t1 = sample['t1'][0].cpu().squeeze().numpy().transpose(1, 2, 0)
    t2 = sample['t2'][0].cpu().squeeze().numpy().transpose(1, 2, 0)
    t3 = sample['t3'][0].cpu().squeeze().numpy().transpose(1, 2, 0)
    t4 = sample['t4'][0].cpu().squeeze().numpy().transpose(1, 2, 0)
    t5 = sample['t5'][0].cpu().squeeze().numpy().transpose(1, 2, 0)

    img_seq = [t1, t2, t3, t4, t5]
    for i in range(len(img_seq)):
        img_seq[i] = zscore_norm_reverse(img_seq[i], config['datasets']['norm']['mean'], config['datasets']['norm']['std'])
        # 2024/4/18: add different visualization for RaVAEn dataset (s2 tif files)
        if config['datasets']['train_path'].split('/')[-2] == 'EuroSAT_hdf5':
            img_seq[i] = np.clip(img_seq[i], 0, 255)
            img_seq[i] = img_seq[i].astype(np.uint8)
        elif config['datasets']['train_path'].split('/')[-2] == 'ravaen_hdf5':
            img_seq[i] = np.clip(img_seq[i] / 3000., 0, 1) * 255
            img_seq[i] = img_seq[i].astype(np.uint8)

    return img_seq


def cosine_distance(x1, x2):
    """
    TODO: should we add a mlp layer to project the embedding to a lower dimension?
    for example, (B, H/P * W/P, D) -> (B, hidden_dim)?
    """
    x1 = x1.reshape(x1.shape[0], -1)
    x2 = x2.reshape(x2.shape[0], -1)
    return 1 - torch.abs(F.cosine_similarity(x1, x2, dim=-1))


def train_one_epoch(model: torch.nn.Module, 
                    criterion: torch.nn.Module, 
                    data_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler._LRScheduler,
                    scaler: torch.cuda.amp.GradScaler,
                    epoch: int,
                    config: dict):
    if config['baseline'] == 'vanilla_ae':
        losses_recon_output = AverageMeter()
        losses_embedding = AverageMeter()
        losses_total = AverageMeter()

        losses_embedding_contrastive = AverageMeter()
        losses_embedding_consistency = AverageMeter()

    elif config['baseline'] == 'vanilla_vae':
        losses_recon_output = AverageMeter()
        losses_kl_divergence = AverageMeter()
        losses_z = AverageMeter()
        losses_total = AverageMeter()

        losses_z_contrastive = AverageMeter()
        losses_z_consistency = AverageMeter()

    elif config['baseline'] == 'bi_earlyfusion' \
        or config['baseline'] == 'bi_siamconcat' \
        or config['baseline'] == 'bi_siamdiff' \
        or config['baseline'] == 'multi_earlyfusion' \
        or config['baseline'] == 'multi_siamconcat' \
        or config['baseline'] == 'multi_siamdiff':
        losses_total = AverageMeter()

    else:
        raise ValueError(f"Invalid baseline: {config['baseline']}")
    

    iterator = tqdm(data_loader)

    model.train()

    for i, sample in enumerate(iterator):
        # .cuda() is a shorthand for .to(device='cuda')
        for k, v in sample.items():
            sample[k] = v.cuda(non_blocking=True)

        # Forward pass
        if config['baseline'] == 'vanilla_ae':
            with autocast():
                recon_output, embedding = model(sample)
                losses = criterion(recon_output, embedding, sample, config['losses']['temperature'])

                loss_recon_output = losses['loss_recon_output']
                loss_embedding_contrastive = losses['loss_embedding_contrastive']
                loss_embedding_consistency = losses['loss_embedding_consistency']

                loss_embedding = config['losses']['weight_embedding_contrastive'] * loss_embedding_contrastive + \
                                config['losses']['weight_embedding_consistency'] * loss_embedding_consistency
                
                loss = config['losses']['weight_recon_output'] * loss_recon_output + \
                       config['losses']['weight_embedding'] * loss_embedding
                
            losses_recon_output.update(loss_recon_output.item(), sample['base'].size(0))
            losses_embedding.update(loss_embedding.item(), sample['base'].size(0))
            losses_total.update(loss.item(), sample['base'].size(0))

            losses_embedding_contrastive.update(loss_embedding_contrastive.item(), sample['base'].size(0))
            losses_embedding_consistency.update(loss_embedding_consistency.item(), sample['base'].size(0))

            iterator.set_description("Epoch: {}, Loss: {loss.val:.4f} ({loss.avg:.4f}), \
                                    loss_recon_output: {loss_recon_output.val:.4f} ({loss_recon_output.avg:.4f}), \
                                    loss_embedding: {loss_embedding.val:.4f} ({loss_embedding.avg:.4f}), \
                                    loss_embedding_contrastive: {loss_embedding_contrastive.val:.4f} ({loss_embedding_contrastive.avg:.4f}), \
                                    loss_embedding_consistency: {loss_embedding_consistency.val:.4f} ({loss_embedding_consistency.avg:.4f})".format \
                                    (epoch, loss=losses_total, loss_recon_output=losses_recon_output, loss_embedding=losses_embedding, \
                                    loss_embedding_contrastive=losses_embedding_contrastive, loss_embedding_consistency=losses_embedding_consistency))

            # log with wandb
            wandb.log({'step': i, 'loss': loss, 'loss_recon_output': loss_recon_output, \
                       'loss_embedding': loss_embedding, 'loss_embedding_contrastive': loss_embedding_contrastive, \
                       'loss_embedding_consistency': loss_embedding_consistency})
        
        elif config['baseline'] == 'vanilla_vae':
            with autocast():
                recon_output, z_dict, mu_dict, log_var_dict = model(sample)
                losses = criterion(recon_output, z_dict, mu_dict, log_var_dict, sample, config['losses']['temperature'])

                loss_recon_output = losses['loss_recon_output']
                loss_kl_divergence = losses['loss_kl_divergence']
                loss_z_contrastive = losses['loss_z_contrastive']
                loss_z_consistency = losses['loss_z_consistency']

                loss_z = config['losses']['weight_z_contrastive'] * loss_z_contrastive + \
                         config['losses']['weight_z_consistency'] * loss_z_consistency

                loss = config['losses']['weight_recon_output'] * loss_recon_output + \
                       config['losses']['weight_kl_divergence'] * loss_kl_divergence + \
                       config['losses']['weight_z'] * loss_z

            losses_recon_output.update(loss_recon_output.item(), sample['base'].size(0))
            losses_kl_divergence.update(loss_kl_divergence.item(), sample['base'].size(0))
            losses_z.update(loss_z.item(), sample['base'].size(0))
            losses_total.update(loss.item(), sample['base'].size(0))

            losses_z_contrastive.update(loss_z_contrastive.item(), sample['base'].size(0))
            losses_z_consistency.update(loss_z_consistency.item(), sample['base'].size(0))

            iterator.set_description("Epoch: {}, Loss: {loss.val:.4f} ({loss.avg:.4f}), \
                                    loss_recon_output: {loss_recon_output.val:.4f} ({loss_recon_output.avg:.4f}), \
                                    loss_kl_divergence: {loss_kl_divergence.val:.4f} ({loss_kl_divergence.avg:.4f}), \
                                    loss_z: {loss_z.val:.4f} ({loss_z.avg:.4f}), \
                                    loss_z_contrastive: {loss_z_contrastive.val:.4f} ({loss_z_contrastive.avg:.4f}), \
                                    loss_z_consistency: {loss_z_consistency.val:.4f} ({loss_z_consistency.avg:.4f})".format \
                                    (epoch, loss=losses_total, loss_recon_output=losses_recon_output, loss_kl_divergence=losses_kl_divergence, \
                                    loss_z=losses_z, loss_z_contrastive=losses_z_contrastive, loss_z_consistency=losses_z_consistency))

            # log with wandb
            wandb.log({'step': i, 'loss': loss, 'loss_recon_output': loss_recon_output, \
                       'loss_kl_divergence': loss_kl_divergence, 'loss_z': loss_z, \
                       'loss_z_contrastive': loss_z_contrastive, 'loss_z_consistency': loss_z_consistency})

        elif config['baseline'] == 'bi_earlyfusion' \
            or config['baseline'] == 'bi_siamconcat' \
            or config['baseline'] == 'bi_siamdiff' \
            or config['baseline'] == 'multi_earlyfusion' \
            or config['baseline'] == 'multi_siamconcat' \
            or config['baseline'] == 'multi_siamdiff':
            with autocast():
                embdding_dict, diff_dict = model(sample)
                loss = criterion(diff_dict, sample)

            losses_total.update(loss.item(), sample['base'].size(0))

            iterator.set_description("Epoch: {}, Loss: {loss.val:.4f} ({loss.avg:.4f})".format \
                                    (epoch, loss=losses_total))

            # log with wandb
            wandb.log({'step': i, 'loss': loss})

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.999)

        scaler.step(optimizer)
        scaler.update()

    if scheduler is not None:
        scheduler.step(epoch)

    # Save model checkpoint after every epoch
    if config['log']['checkpoint_dir'] is not None:
        os.makedirs(config['log']['checkpoint_dir'], exist_ok=True)

    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), \
                'optimizer_state_dict': optimizer.state_dict(), \
                'scheduler_state_dict': scheduler.state_dict()}, \
                os.path.join(config['log']['checkpoint_dir'], \
                             'checkpoint_' + config['exp_name'] + '_' + str(config['seed']) + '.pth'))


    # Log after every epoch
    if config['baseline'] == 'vanilla_ae':
        wandb.log({'epoch': epoch, 'losses': losses_total.avg, \
                   'losses_recon_output': losses_recon_output.avg, 'losses_embedding': losses_embedding.avg, \
                   'losses_embedding_contrastive': losses_embedding_contrastive.avg, 'losses_embedding_consistency': losses_embedding_consistency.avg})
    elif config['baseline'] == 'vanilla_vae':
        wandb.log({'epoch': epoch, 'losses': losses_total.avg, \
                   'losses_recon_output': losses_recon_output.avg, 'losses_kl_divergence': losses_kl_divergence.avg, \
                   'losses_z': losses_z.avg, 'losses_z_contrastive': losses_z_contrastive.avg, \
                   'losses_z_consistency': losses_z_consistency.avg})
    elif config['baseline'] == 'bi_earlyfusion' \
        or config['baseline'] == 'bi_siamconcat' \
        or config['baseline'] == 'bi_siamdiff' \
        or config['baseline'] == 'multi_earlyfusion' \
        or config['baseline'] == 'multi_siamconcat' \
        or config['baseline'] == 'multi_siamdiff':
        wandb.log({'epoch': epoch, 'losses': losses_total.avg})


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader: torch.utils.data.DataLoader,
             epoch: int,
             config: dict):
    iterator = tqdm(data_loader)
    model.eval()

    y_true = np.array([])
    y_pred = np.array([])
    y_pred_multi_temporal_mean = np.array([])

    for i, sample in enumerate(iterator):
        # .cuda() is a shorthand for .to(device='cuda')
        for k, v in sample.items():
            sample[k] = v.cuda(non_blocking=True)

        if config['baseline'] == 'vanilla_ae':
            recon_output, embedding = model(sample)
        elif config['baseline'] == 'vanilla_vae':
            recon_output, embedding, mu_dict, log_var_dict = model(sample)
        elif config['baseline'] == 'bi_earlyfusion' \
            or config['baseline'] == 'bi_siamconcat' \
            or config['baseline'] == 'bi_siamdiff' \
            or config['baseline'] == 'multi_earlyfusion' \
            or config['baseline'] == 'multi_siamconcat' \
            or config['baseline'] == 'multi_siamdiff':
            embedding, diff_dict = model(sample)

        # calculate cd between t4 and t5
        cd = cosine_distance(embedding['t4'], embedding['t5'])

        # calculate multi-temporal cd
        if config['timesteps'] == 5:
            cd_multi_temporal = torch.stack((cosine_distance(embedding['t1'], embedding['t5']), \
                                             cosine_distance(embedding['t2'], embedding['t5']), \
                                             cosine_distance(embedding['t3'], embedding['t5']), \
                                             cosine_distance(embedding['t4'], embedding['t5'])), dim=1)
        elif config['timesteps'] == 4:
            cd_multi_temporal = torch.stack((cosine_distance(embedding['t2'], embedding['t5']), \
                                             cosine_distance(embedding['t3'], embedding['t5']), \
                                             cosine_distance(embedding['t4'], embedding['t5'])), dim=1)
        elif config['timesteps'] == 3:
            cd_multi_temporal = torch.stack((cosine_distance(embedding['t3'], embedding['t5']), \
                                             cosine_distance(embedding['t4'], embedding['t5'])), dim=1)
        elif config['timesteps'] == 2:
            cd_multi_temporal = torch.stack([cosine_distance(embedding['t4'], embedding['t5'])], dim=1)
            
        # calculate the mean of multi-temporal cd
        cd_multi_temporal_mean = torch.mean(cd_multi_temporal, dim=1)

        # Save sample['change'] and cos_dist for calculating AP
        y_true = np.append(y_true, sample['change'].cpu().squeeze().numpy())
        y_pred = np.append(y_pred, cd.cpu().squeeze().numpy())
        y_pred_multi_temporal_mean = np.append(y_pred_multi_temporal_mean, cd_multi_temporal_mean.cpu().squeeze().numpy())
        
    # Calculate AP
    ap = average_precision_score(y_true, y_pred)
    ap_multi_temporal_mean = average_precision_score(y_true, y_pred_multi_temporal_mean)
    
    # log with wandb
    wandb.log({'epoch': epoch, 'AP': ap, 'AP_multi_temporal_mean': ap_multi_temporal_mean})

    # Log original and reconstructed images (only for vanilla_ae and vanilla_vae)
    if config['timesteps'] == 5 and (config['baseline'] == 'vanilla_ae' or config['baseline'] == 'vanilla_vae'):
        if epoch % config['log']['log_img_epoch_freq'] == 0:
            img_seq_original = visualize(sample, config)
            img_seq_recon = visualize(recon_output, config)
            
            # Log original and reconstructed images
            log_original = [wandb.Image(img, caption=f"t{j+1}") for j, img in enumerate(img_seq_original)]
            log_recon = [wandb.Image(img, caption=f"t{j+1}") for j, img in enumerate(img_seq_recon)]
            
            # Concatenate original and reconstructed images
            log_img = log_original + log_recon

            # Log images with wandb
            wandb.log({f"epoch_{epoch}": log_img})


@torch.no_grad()
def test(model: torch.nn.Module,
         criterion: torch.nn.Module,
         data_loader: torch.utils.data.DataLoader,
         config: dict):
    iterator = tqdm(data_loader)
    model.eval()

    """2024/4/24: add disaster type for AP calculation (ravaen_dataset)"""
    if config['datasets']['train_path'].split('/')[-2] == 'ravaen_hdf5':
        ravaen_dataset = True
    else:
        ravaen_dataset = False

    y_true = np.array([])
    y_pred = np.array([])
    y_pred_multi_temporal_mean = np.array([])

    disaster_type = np.array([])

    for i, sample in enumerate(iterator):
        # .cuda() is a shorthand for .to(device='cuda')
        for k, v in sample.items():
            sample[k] = v.cuda(non_blocking=True)

        if ravaen_dataset:
            disaster_type = np.append(disaster_type, sample['event'].cpu().squeeze().numpy())

        if config['baseline'] == 'vanilla_ae':
            recon_output, embedding = model(sample)
        elif config['baseline'] == 'vanilla_vae':
            recon_output, embedding, mu_dict, log_var_dict = model(sample)
        elif config['baseline'] == 'bi_earlyfusion' \
            or config['baseline'] == 'bi_siamconcat' \
            or config['baseline'] == 'bi_siamdiff' \
            or config['baseline'] == 'multi_earlyfusion' \
            or config['baseline'] == 'multi_siamconcat' \
            or config['baseline'] == 'multi_siamdiff':
            embedding, diff_dict = model(sample)

        # calculate cd between t4 and t5
        cd = cosine_distance(embedding['t4'], embedding['t5'])

        # calculate multi-temporal cd
        if config['timesteps'] == 5:
            cd_multi_temporal = torch.stack((cosine_distance(embedding['t1'], embedding['t5']), \
                                             cosine_distance(embedding['t2'], embedding['t5']), \
                                             cosine_distance(embedding['t3'], embedding['t5']), \
                                             cosine_distance(embedding['t4'], embedding['t5'])), dim=1)
        elif config['timesteps'] == 4:
            cd_multi_temporal = torch.stack((cosine_distance(embedding['t2'], embedding['t5']), \
                                             cosine_distance(embedding['t3'], embedding['t5']), \
                                             cosine_distance(embedding['t4'], embedding['t5'])), dim=1)
        elif config['timesteps'] == 3:
            cd_multi_temporal = torch.stack((cosine_distance(embedding['t3'], embedding['t5']), \
                                             cosine_distance(embedding['t4'], embedding['t5'])), dim=1)
        elif config['timesteps'] == 2:
            cd_multi_temporal = torch.stack([cosine_distance(embedding['t4'], embedding['t5'])], dim=1)
                
        # calculate the mean of multi-temporal cd
        cd_multi_temporal_mean = torch.mean(cd_multi_temporal, dim=1)

        # Save sample['change'] and cos_dist for calculating AP
        y_true = np.append(y_true, sample['change'].cpu().squeeze().numpy())
        y_pred = np.append(y_pred, cd.cpu().squeeze().numpy())
        y_pred_multi_temporal_mean = np.append(y_pred_multi_temporal_mean, cd_multi_temporal_mean.cpu().squeeze().numpy())

    # Calculate AP
    ap = average_precision_score(y_true, y_pred)
    ap_multi_temporal_mean = average_precision_score(y_true, y_pred_multi_temporal_mean)

    print(f"AP: {ap}, AP_multi_temporal_mean: {ap_multi_temporal_mean}")

    # Calculate Precision, Recall, F1-score
    # y_pred_binary = (y_pred > 0.5).astype(int)
    y_pred_binary = find_best_threshold_f1(y_true, y_pred)
    precision = precision_score(y_true, y_pred_binary, average='binary')
    recall = recall_score(y_true, y_pred_binary, average='binary')
    f1 = f1_score(y_true, y_pred_binary, average='binary')

    # y_pred_multi_temporal_mean_binary = (y_pred_multi_temporal_mean > 0.5).astype(int)
    y_pred_multi_temporal_mean_binary = find_best_threshold_f1(y_true, y_pred_multi_temporal_mean)
    precision_multi_temporal_mean = precision_score(y_true, y_pred_multi_temporal_mean_binary, average='binary')
    recall_multi_temporal_mean = recall_score(y_true, y_pred_multi_temporal_mean_binary, average='binary')
    f1_multi_temporal_mean = f1_score(y_true, y_pred_multi_temporal_mean_binary, average='binary')

    print(f"Precision: {precision}, Recall: {recall}, F1-score: {f1}")
    print(f"Precision_multi_temporal_mean: {precision_multi_temporal_mean}, Recall_multi_temporal_mean: {recall_multi_temporal_mean}, F1-score_multi_temporal_mean: {f1_multi_temporal_mean}")

    # Calculate metrics for each disaster type
    fire_index = np.where(disaster_type == 0)
    flood_index = np.where(disaster_type == 1)
    hurricane_index = np.where(disaster_type == 2)
    landslide_index = np.where(disaster_type == 3)

    if ravaen_dataset:
        # Calculate AP for each disaster type
        ap_fire = average_precision_score(y_true[fire_index], y_pred[fire_index])
        ap_flood = average_precision_score(y_true[flood_index], y_pred[flood_index])
        ap_hurricane = average_precision_score(y_true[hurricane_index], y_pred[hurricane_index])
        ap_landslide = average_precision_score(y_true[landslide_index], y_pred[landslide_index])

        ap_multi_temporal_mean_fire = average_precision_score(y_true[fire_index], y_pred_multi_temporal_mean[fire_index])
        ap_multi_temporal_mean_flood = average_precision_score(y_true[flood_index], y_pred_multi_temporal_mean[flood_index])
        ap_multi_temporal_mean_hurricane = average_precision_score(y_true[hurricane_index], y_pred_multi_temporal_mean[hurricane_index])
        ap_multi_temporal_mean_landslide = average_precision_score(y_true[landslide_index], y_pred_multi_temporal_mean[landslide_index])

        print(f"AP_fire: {ap_fire}, AP_flood: {ap_flood}, AP_hurricane: {ap_hurricane}, AP_landslide: {ap_landslide}")
        print(f"AP_multi_temporal_mean_fire: {ap_multi_temporal_mean_fire}, AP_multi_temporal_mean_flood: {ap_multi_temporal_mean_flood}, AP_multi_temporal_mean_hurricane: {ap_multi_temporal_mean_hurricane}, AP_multi_temporal_mean_landslide: {ap_multi_temporal_mean_landslide}")
        
        # Calculate Precision, Recall, F1-score for each disaster type
        precision_fire = precision_score(y_true[fire_index], y_pred_binary[fire_index], average='binary')
        recall_fire = recall_score(y_true[fire_index], y_pred_binary[fire_index], average='binary')
        f1_fire = f1_score(y_true[fire_index], y_pred_binary[fire_index], average='binary')

        precision_flood = precision_score(y_true[flood_index], y_pred_binary[flood_index], average='binary')
        recall_flood = recall_score(y_true[flood_index], y_pred_binary[flood_index], average='binary')
        f1_flood = f1_score(y_true[flood_index], y_pred_binary[flood_index], average='binary')

        precision_hurricane = precision_score(y_true[hurricane_index], y_pred_binary[hurricane_index], average='binary')
        recall_hurricane = recall_score(y_true[hurricane_index], y_pred_binary[hurricane_index], average='binary')
        f1_hurricane = f1_score(y_true[hurricane_index], y_pred_binary[hurricane_index], average='binary')

        precision_landslide = precision_score(y_true[landslide_index], y_pred_binary[landslide_index], average='binary')
        recall_landslide = recall_score(y_true[landslide_index], y_pred_binary[landslide_index], average='binary')
        f1_landslide = f1_score(y_true[landslide_index], y_pred_binary[landslide_index], average='binary')

        print(f"Precision_fire: {precision_fire}, Recall_fire: {recall_fire}, F1-score_fire: {f1_fire}")
        print(f"Precision_flood: {precision_flood}, Recall_flood: {recall_flood}, F1-score_flood: {f1_flood}")
        print(f"Precision_hurricane: {precision_hurricane}, Recall_hurricane: {recall_hurricane}, F1-score_hurricane: {f1_hurricane}")
        print(f"Precision_landslide: {precision_landslide}, Recall_landslide: {recall_landslide}, F1-score_landslide: {f1_landslide}")

        # Calculate Precision, Recall, F1-score for each disaster type (multi-temporal mean)
        precision_multi_temporal_mean_fire = precision_score(y_true[fire_index], y_pred_multi_temporal_mean_binary[fire_index], average='binary')
        recall_multi_temporal_mean_fire = recall_score(y_true[fire_index], y_pred_multi_temporal_mean_binary[fire_index], average='binary')
        f1_multi_temporal_mean_fire = f1_score(y_true[fire_index], y_pred_multi_temporal_mean_binary[fire_index], average='binary')

        precision_multi_temporal_mean_flood = precision_score(y_true[flood_index], y_pred_multi_temporal_mean_binary[flood_index], average='binary')
        recall_multi_temporal_mean_flood = recall_score(y_true[flood_index], y_pred_multi_temporal_mean_binary[flood_index], average='binary')
        f1_multi_temporal_mean_flood = f1_score(y_true[flood_index], y_pred_multi_temporal_mean_binary[flood_index], average='binary')

        precision_multi_temporal_mean_hurricane = precision_score(y_true[hurricane_index], y_pred_multi_temporal_mean_binary[hurricane_index], average='binary')
        recall_multi_temporal_mean_hurricane = recall_score(y_true[hurricane_index], y_pred_multi_temporal_mean_binary[hurricane_index], average='binary')
        f1_multi_temporal_mean_hurricane = f1_score(y_true[hurricane_index], y_pred_multi_temporal_mean_binary[hurricane_index], average='binary')

        precision_multi_temporal_mean_landslide = precision_score(y_true[landslide_index], y_pred_multi_temporal_mean_binary[landslide_index], average='binary')
        recall_multi_temporal_mean_landslide = recall_score(y_true[landslide_index], y_pred_multi_temporal_mean_binary[landslide_index], average='binary')
        f1_multi_temporal_mean_landslide = f1_score(y_true[landslide_index], y_pred_multi_temporal_mean_binary[landslide_index], average='binary')

        print(f"Precision_multi_temporal_mean_fire: {precision_multi_temporal_mean_fire}, Recall_multi_temporal_mean_fire: {recall_multi_temporal_mean_fire}, F1-score_multi_temporal_mean_fire: {f1_multi_temporal_mean_fire}")
        print(f"Precision_multi_temporal_mean_flood: {precision_multi_temporal_mean_flood}, Recall_multi_temporal_mean_flood: {recall_multi_temporal_mean_flood}, F1-score_multi_temporal_mean_flood: {f1_multi_temporal_mean_flood}")
        print(f"Precision_multi_temporal_mean_hurricane: {precision_multi_temporal_mean_hurricane}, Recall_multi_temporal_mean_hurricane: {recall_multi_temporal_mean_hurricane}, F1-score_multi_temporal_mean_hurricane: {f1_multi_temporal_mean_hurricane}")
        print(f"Precision_multi_temporal_mean_landslide: {precision_multi_temporal_mean_landslide}, Recall_multi_temporal_mean_landslide: {recall_multi_temporal_mean_landslide}, F1-score_multi_temporal_mean_landslide: {f1_multi_temporal_mean_landslide}")

        dict = {
            'AP': ap,
            'AP_multi_temporal_mean': ap_multi_temporal_mean,
            'AP_fire': ap_fire,
            'AP_flood': ap_flood,
            'AP_hurricane': ap_hurricane,
            'AP_landslide': ap_landslide,
            'AP_multi_temporal_mean_fire': ap_multi_temporal_mean_fire,
            'AP_multi_temporal_mean_flood': ap_multi_temporal_mean_flood,
            'AP_multi_temporal_mean_hurricane': ap_multi_temporal_mean_hurricane,
            'AP_multi_temporal_mean_landslide': ap_multi_temporal_mean_landslide,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'Precision_multi_temporal_mean': precision_multi_temporal_mean,
            'Recall_multi_temporal_mean': recall_multi_temporal_mean,
            'F1-score_multi_temporal_mean': f1_multi_temporal_mean,
            'Precision_fire': precision_fire,
            'Recall_fire': recall_fire,
            'F1-score_fire': f1_fire,
            'Precision_flood': precision_flood,
            'Recall_flood': recall_flood,
            'F1-score_flood': f1_flood,
            'Precision_hurricane': precision_hurricane,
            'Recall_hurricane': recall_hurricane,
            'F1-score_hurricane': f1_hurricane,
            'Precision_landslide': precision_landslide,
            'Recall_landslide': recall_landslide,
            'F1-score_landslide': f1_landslide,
            'Precision_multi_temporal_mean_fire': precision_multi_temporal_mean_fire,
            'Recall_multi_temporal_mean_fire': recall_multi_temporal_mean_fire,
            'F1-score_multi_temporal_mean_fire': f1_multi_temporal_mean_fire,
            'Precision_multi_temporal_mean_flood': precision_multi_temporal_mean_flood,
            'Recall_multi_temporal_mean_flood': recall_multi_temporal_mean_flood,
            'F1-score_multi_temporal_mean_flood': f1_multi_temporal_mean_flood,
            'Precision_multi_temporal_mean_hurricane': precision_multi_temporal_mean_hurricane,
            'Recall_multi_temporal_mean_hurricane': recall_multi_temporal_mean_hurricane,
            'F1-score_multi_temporal_mean_hurricane': f1_multi_temporal_mean_hurricane,
            'Precision_multi_temporal_mean_landslide': precision_multi_temporal_mean_landslide,
            'Recall_multi_temporal_mean_landslide': recall_multi_temporal_mean_landslide,
            'F1-score_multi_temporal_mean_landslide': f1_multi_temporal_mean_landslide
        }
    else:
        dict = {
            'AP': ap,
            'AP_multi_temporal_mean': ap_multi_temporal_mean,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'Precision_multi_temporal_mean': precision_multi_temporal_mean,
            'Recall_multi_temporal_mean': recall_multi_temporal_mean,
            'F1-score_multi_temporal_mean': f1_multi_temporal_mean
        }

    return dict


def find_best_threshold_f1(y_true, y_proba, step=0.01):
    """
    Finds the best threshold for maximizing F1-score
    """
    thresholds = np.arange(0, 1.01, step)
    f1_scores = []

    # Iterate over thresholds
    for threshold in thresholds:
        y_pred = (y_proba > threshold).astype(int)
        f1 = f1_score(y_true, y_pred, average='binary')
        f1_scores.append(f1)

    # Find the best threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"Best threshold: {best_threshold}, Best F1-score: {best_f1}")

    y_pred_binary = (y_proba > best_threshold).astype(int)

    return y_pred_binary