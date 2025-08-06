import argparse
import logging
import os
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import scene_flow_metrics, GeneratorWrap, EarlyStopping
from model import Convection_backbone

device = torch.device("cuda:0")

def solver(
    pc1: torch.Tensor,
    pc2: torch.Tensor,
    flow: torch.Tensor,
    options: argparse.Namespace,
    net: torch.nn.Module,
    i: int,
):
    for param in net.parameters():
        param.requires_grad = True
    
    if options.backward_flow:
        net_inv = copy.deepcopy(net)
        params = [{'params': net.parameters(), 'lr': options.lr, 'weight_decay': options.weight_decay},
                {'params': net_inv.parameters(), 'lr': options.lr, 'weight_decay': options.weight_decay}]
    else:
        params = net.parameters()
    
    if options.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=options.lr, momentum=options.momentum, weight_decay=options.weight_decay)
    elif options.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=options.lr, weight_decay=0)

    early_stopping = EarlyStopping(patience=options.early_patience, min_delta=0.0001)

    pc1 = pc1.cuda().contiguous()
    pc2 = pc2.cuda().contiguous()
    flow = flow.cuda().contiguous()

    best_loss_1 = 10.
    best_flow_1 = None
    best_epe3d_1 = 1.
    best_acc3d_strict_1 = 0.
    best_acc3d_relax_1 = 0.
    best_angle_error_1 = 1.
    best_outliers_1 = 1.
    best_epoch = 0
    
    for epoch in range(options.iters):
        optimizer.zero_grad()

        flow_pred_1 = net(pc1)
        pc1_deformed = pc1 + flow_pred_1
        loss_chamfer_1, _ = my_chamfer_fn(pc2, pc1_deformed, None, None)
        loss = loss_chamfer_1

        if options.l1regu>0:
            for param in net.parameters():
                loss += options.l1regu * torch.sum(abs(param))
        
        if options.backward_flow:
            flow_pred_1_prime = net_inv(pc1_deformed)
            pc1_prime_deformed = pc1_deformed - flow_pred_1_prime
            loss_chamfer_1_prime, _ = my_chamfer_fn(pc1_prime_deformed, pc1, None, None)
            flow_pred_2_prime = net_inv(pc2)
            pc2_prime_deformed = pc2 - flow_pred_2_prime
            loss_chamfer_2_prime, _ = my_chamfer_fn(pc2_prime_deformed, pc1, None, None)
            loss += loss_chamfer_1_prime + loss_chamfer_2_prime

        flow_pred_1_final = pc1_deformed - pc1
        
        if options.compute_metrics:
            EPE3D_1, acc3d_strict_1, acc3d_relax_1, outlier_1, angle_error_1 = scene_flow_metrics(flow_pred_1_final, flow)
        else:
            EPE3D_1, acc3d_strict_1, acc3d_relax_1, outlier_1, angle_error_1 = 0, 0, 0, 0, 0

        if loss <= best_loss_1:
            best_loss_1 = loss.item()
            best_flow_1 = flow_pred_1_final
            best_epe3d_1 = EPE3D_1
            best_acc3d_strict_1 = acc3d_strict_1
            best_acc3d_relax_1 = acc3d_relax_1
            best_angle_error_1 = angle_error_1
            best_outliers_1 = outlier_1
            best_epoch = epoch
            
        if early_stopping.step(loss):
            break
        
        loss.backward()
        optimizer.step()

    info_dict = {
        'loss': best_loss_1,
        'EPE3D_1': best_epe3d_1,
        'acc3d_strict_1': best_acc3d_strict_1,
        'acc3d_relax_1': best_acc3d_relax_1,
        'angle_error_1': best_angle_error_1,
        'outlier_1': best_outliers_1,
        'epoch': best_epoch
    }

    np.savez(f'Convection_{i}',pc1 = pc1[0].detach().cpu().numpy(),est_flow = best_flow_1[0].detach().cpu().numpy())
    return info_dict


def optimize_Convection_backbone(options, data_loader):
    save_dir_path = f"checkpoints/{options.exp_name}"
    outputs = []
    
    if options.model == 'Convection_backbone':
        net = Convection_backbone(filter_size=options.hidden_units, act_fn=options.act_fn, layer_size=options.layer_size).cuda()

    for i, data in tqdm(enumerate(data_loader), total=len(data_loader), smoothing=0.9):
        pc1, pc2, flow = data

        solver_generator = GeneratorWrap(solver(pc1, pc2, flow, options, net, i))
        for _ in solver_generator: pass
        info_dict = solver_generator.value

        info_dict['filepath'] = data_loader.dataset.datapath[i]
        outputs.append(info_dict)

    df = pd.DataFrame(outputs)
    df.loc['mean'] = df.mean()
    df.to_csv(f'{save_dir_path}/results.csv')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convection Network.")
    config.add_config(parser)
    options = parser.parse_args()

    exp_dir_path = f"checkpoints/{options.exp_name}"
    if not os.path.exists(exp_dir_path):
        os.makedirs(exp_dir_path)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[logging.FileHandler(filename=f"{exp_dir_path}/run.log"), logging.StreamHandler()])
    
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(options.seed)
    torch.cuda.manual_seed_all(options.seed)
    np.random.seed(options.seed)

    from FlowTestDataset import FlowTestDataset
    data_loader = DataLoader(FlowTestDataset())
    optimize_Convection_backbone(options, data_loader)