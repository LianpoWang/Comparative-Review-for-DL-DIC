import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
import numpy as np
import os

from tqdm import tqdm

from utils.datasets import SpeckleDataset
from networks.dictr import DICTr
from loss import flow_loss_func
from experiment import custom, rotation, tension, star5, mei, realcrack
from utils.logger import Logger
from utils import misc
from utils.dist_utils import get_dist_info, init_dist, setup_for_distributed
'''
    使用numsteps训练 不看epoch
'''

def get_args_parser():
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--checkpoint_dir', default='/home/dell/DATA/wh/dictr-main/result1/', type=str,
                        help='where to save the training log and models')
    parser.add_argument('--stage', default=['speckle'], type=str,
                        help='training stage')
    parser.add_argument('--padding_factor', default=20, type=int,
                        help='the input should be divisible by padding_factor, otherwise do padding')
    parser.add_argument('--taset', default=['speckle'], type=str, nargs='+',
                        help='validation dataset')

    # training strategy
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--num_steps', default=100000, type=int)
    parser.add_argument('--seed', default=326, type=int)
    parser.add_argument('--summary_freq', default=100, type=int)
    parser.add_argument('--val_freq', default=5000, type=int)
    parser.add_argument('--save_ckpt_freq', default=5000, type=int)
    parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)
    parser.add_argument('--gamma', default=0.9, type=float, help='loss weight for each layer')

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrain model for finetuing or resume from terminated training')
    parser.add_argument('--strict_resume', action='store_true')
    parser.add_argument('--no_resume_optimizer', action='store_true')

    # DICTr model
    parser.add_argument('--num_scales', default=2, type=int,
                        help='DICTr use 2 scale features, 1/4 for global match and 1/2 for refinement')
    parser.add_argument('--feature_channels', default=128, type=int,
                        help='DICTr use 128 channels for higher-level description of features')
    parser.add_argument('--upsample_factor', default=2, type=int,
                        help='DICTr get full resolution result by convex upsampling from 1/2 resolution')
    parser.add_argument('--num_transformer_layers', default=12, type=int,
                        help='DICTr use 12 transformer layer (6 blocks) to enhence image features')
    parser.add_argument('--num_head', default=1, type=int,
                        help='DICTr use single head attention')
    parser.add_argument('--attention_type', default='swin', type=str,
                        help='DICTr use swin transformer')
    parser.add_argument('--ffn_dim_expansion', default=4, type=int,
                        help='Dimension expansion scale in Feed-Forward Networks, follow <Attention Is All You Need>')

    # GMFlow model default setting, you can switch to a smaller window size to carry out
    # the attention mechanism when computational costs become a bottleneck
    # In DICTr, first parameter is for 1/4 scale features and second parameter is for 1/2 scale features
    # For 2D-DIC, flow propagation may be unnecessary since there is no occlusion problem
    parser.add_argument('--attn_splits_list', default=[2, 8], type=int, nargs='+',
                        help='number of splits on feature map edge to form window layout for swin transformer')#[2,8]->[3,10]
    parser.add_argument('--corr_radius_list', default=[-1, 4], type=int, nargs='+',
                        help='radius for feature matching, -1 indicates global matching')#[-1,4]
    parser.add_argument('--prop_radius_list', default=[-1, 1], type=int, nargs='+',
                        help='self-attention radius for flow propagation, -1 indicates global attention')#[-1,1]

    # inference
    parser.add_argument('--exp', action='store_true')
    parser.add_argument('--exp_type', type=str, nargs='+')

    # distributed training
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    return parser
n_val=0
@torch.no_grad()
def validate_speckle(model,val_writer,
                     attn_splits_list=False,
                     corr_radius_list=False,
                     prop_radius_list=False,
                     ):
    model.eval()
    global n_val
    val_dataset = SpeckleDataset("/home/dell/DATA/wh/DATASET/Val_annotations1",'/home/dell/DATA/wh/DATASET/Train1', 8160)
    print('Number of validation image pairs: %d' % len(val_dataset))
    epe_list = []
    results = {}

    s00_05_list = []
    s05_10_list = []
    s10plus_list = []

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        results_dict = model(image1, image2,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             )

        # useful when using parallel branches
        flow_pr = results_dict['flow_preds'][-1]
        flow = flow_pr[0].cpu()

        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        aee=epe.mean()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()
        val_writer.add_scalar("Val AEE1 per Batch",aee.item(),n_val)
        n_val+=1
        flow_gt_speed = mag

        valid_mask = (flow_gt_speed < 0.5)
        if valid_mask.max() > 0:
            s00_05_list.append(epe[valid_mask].detach().cpu().numpy())

        valid_mask = (flow_gt_speed >= 0.5) * (flow_gt_speed <= 1)
        if valid_mask.max() > 0:
            s05_10_list.append(epe[valid_mask].detach().cpu().numpy())

        valid_mask = (flow_gt_speed > 1)
        if valid_mask.max() > 0:
            s10plus_list.append(epe[valid_mask].detach().cpu().numpy())

        epe = epe.view(-1)
        val = valid_gt.view(-1) >= 0.5

        epe_list.append(epe[val].detach().cpu().numpy())
        # 释放中间变量
        del image1, image2, flow_pr, flow, epe, mag, flow_gt_speed, valid_mask, val
        torch.cuda.empty_cache()

    epe_list = np.concatenate(epe_list)

    epe = np.mean(epe_list)

    print("Validation dataset AEE: %.3f" % epe)
    results['dataset_AEE'] = epe

    s00_05 = np.mean(np.concatenate(s00_05_list))
    s05_10 = np.mean(np.concatenate(s05_10_list))
    s10plus = np.mean(np.concatenate(s10plus_list))

    print("Validation dataset AEE, s0_0.5: %.3f, s0.5_1: %.3f, s1+: %.3f" % (
        s00_05,
        s05_10,
        s10plus))

    results['dataset_AEE_s0_0.5'] = s00_05
    results['dataset_AEE_s0.5_1'] = s05_10
    results['dataset_AEE_s1+'] = s10plus

    return results

def main(args):
    global_step = 0  # 初始化全局 step

    if not args.exp:
        if args.local_rank == 0:
            print('pytorch version:', torch.__version__)
            print(args)
            # misc.save_args(args)
            misc.check_path(args.checkpoint_dir)
            # misc.save_command(args.checkpoint_dir)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = True

    if args.launcher == 'none':
        args.distributed = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    else:
        args.distributed = True

        # adjust batch size for each gpu
        assert args.batch_size % torch.cuda.device_count() == 0
        args.batch_size = args.batch_size // torch.cuda.device_count()

        dist_params = dict(backend='nccl')
        init_dist(args.launcher, **dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        args.gpu_ids = range(world_size)
        device = torch.device('cuda:{}'.format(args.local_rank))

        setup_for_distributed(args.local_rank == 0)

    # model
    model = DICTr(feature_channels=args.feature_channels,
                  num_scales=args.num_scales,
                  upsample_factor=args.upsample_factor,
                  num_head=args.num_head,
                  attention_type=args.attention_type,
                  ffn_dim_expansion=args.ffn_dim_expansion,
                  num_transformer_layers=args.num_transformer_layers,
                  ).to(device)

    if not args.exp:
        print('Model definition:')
        print(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device),
            device_ids=[args.local_rank],
            output_device=args.local_rank)
        model_without_ddp = model.module
    else:
        if torch.cuda.device_count() > 1:
            print('Use %d GPUs' % torch.cuda.device_count())
            model = torch.nn.DataParallel(model)

            model_without_ddp = model.module
        else:
            model_without_ddp = model

    num_params = sum(p.numel() for p in model.parameters())
    print('Number of params:', num_params)
    # if not args.exp:
    #    save_name = '%d_parameters' % num_params
    #    open(os.path.join(args.checkpoint_dir, save_name), 'a').close()
    train_dataset = SpeckleDataset("/home/dell/DATA/wh/DATASET/Train_annotations1.csv",
                                   '/home/dell/DATA/wh/DATASET/Train1', 32640)
    # Multi-processing
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=torch.cuda.device_count(),
            rank=args.local_rank
        )
        shuffle = False  # distributed 时 DataLoader 不需要 shuffle，sampler 会处理
    else:
        train_sampler = None
        shuffle = True  # 非分布式可以 shuffle

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler  # distributed 时传入 sampler，非分布式传 None
    )

    start_epoch = 0
    start_step = 0
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    last_epoch = start_step if args.resume and start_step > 0 else -1
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy='cos',
        last_epoch=last_epoch,
    )
    # resume checkpoints
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=args.strict_resume)
        if 'optimizer' in checkpoint and not args.no_resume_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('step', 0)
        print(f"Resumed from {args.resume}, epoch {start_epoch}, step {global_step}")

    # experiment
    if args.exp:
        if 'custom' in args.exp_type:
            custom(model_without_ddp,
                   attn_splits_list=args.attn_splits_list,
                   corr_radius_list=args.corr_radius_list,
                   prop_radius_list=args.prop_radius_list)
        if 'rotation' in args.exp_type:
            rotation(model_without_ddp,
                     attn_splits_list=args.attn_splits_list,
                     corr_radius_list=args.corr_radius_list,
                     prop_radius_list=args.prop_radius_list)
        if 'tension' in args.exp_type:
            tension(model_without_ddp,
                    attn_splits_list=args.attn_splits_list,
                    corr_radius_list=args.corr_radius_list,
                    prop_radius_list=args.prop_radius_list)
        if 'star5' in args.exp_type:
            star5(model_without_ddp,
                  attn_splits_list=args.attn_splits_list,
                  corr_radius_list=args.corr_radius_list,
                  prop_radius_list=args.prop_radius_list)
        if 'mei' in args.exp_type:
            mei(model_without_ddp,
                attn_splits_list=args.attn_splits_list,
                corr_radius_list=args.corr_radius_list,
                prop_radius_list=args.prop_radius_list)
        if 'realcrack' in args.exp_type:
            realcrack(model_without_ddp,
                      attn_splits_list=args.attn_splits_list,
                      corr_radius_list=args.corr_radius_list,
                      prop_radius_list=args.prop_radius_list)

        return




    if args.local_rank == 0:
        summary_writer = SummaryWriter(args.checkpoint_dir)
        logger = Logger(lr_scheduler, summary_writer, args.summary_freq,
                        start_step=start_step)
        trainlog_path = os.path.join(args.checkpoint_dir, 'train_log1')
        vallog_path = os.path.join(args.checkpoint_dir, 'val_log1')
        model_path = os.path.join(args.checkpoint_dir, 'model1')
        os.makedirs(trainlog_path, exist_ok=True)
        os.makedirs(vallog_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
        train_writer = SummaryWriter(trainlog_path)
        val_writer = SummaryWriter(vallog_path)
    total_steps = start_step
    print('Start training')
    step=0
    for epoch in range(start_epoch, args.num_epochs):
        total_loss = 0
        total_aee = 0
        total_steps = 0
        total_batches = len(train_loader)
        start_time = time.time()  # 记录每个 epoch 的开始时间

        # 使用 tqdm 包装 train_loader 以显示进度条
        progress_bar = tqdm(enumerate(train_loader), total=total_batches, desc=f"DICTr1_Epoch {epoch + 1}/{args.num_epochs}")
        for step, sample in progress_bar:
            img1, img2, flow_gt, valid = [x.to(device) for x in sample]
            # print(valid.shape)
            results_dict = model(img1, img2,
                                 attn_splits_list=args.attn_splits_list,
                                 corr_radius_list=args.corr_radius_list,
                                 prop_radius_list=args.prop_radius_list,
                                 )

            flow_preds = results_dict['flow_preds']

            loss, metrics = flow_loss_func(flow_preds, flow_gt, valid,
                                           gamma=args.gamma,
                                           )

            if isinstance(loss, float):
                continue

            if torch.isnan(loss):
                continue

            metrics.update({'total_loss': loss.item()})
            metrics.update({'total_aee': metrics['AEE']})
            # more efficient zero_grad
            for param in model_without_ddp.parameters():
                param.grad = None

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            torch.cuda.empty_cache()
            global_step += 1

            if args.local_rank == 0:
                train_writer.add_scalar('Loss1 per Batch', loss.item(), step)
                train_writer.add_scalar('AEE1 per Batch', metrics['AEE'], step)
                logger.push(metrics)

                elapsed_time = time.time() - start_time  # 计算已用时间
                batches_processed = step + 1
                remaining_batches = total_batches - batches_processed
                if batches_processed > 0:
                    avg_time_per_batch = elapsed_time / batches_processed
                    remaining_time = avg_time_per_batch * remaining_batches
                    # 更新进度条信息，包含剩余时间
                    progress_bar.set_postfix({'Loss': f"{loss.item():.3f}", 'AEE': f"{metrics['AEE']:.3f}",
                                              'Remaining Time': f"{remaining_time / 60:.2f} mins"})
                else:
                    progress_bar.set_postfix({'Loss': f"{loss.item():.3f}", 'AEE': f"{metrics['AEE']:.3f}"})

            total_loss += loss.item()
            total_aee += metrics['AEE']
            total_steps += 1

        if args.local_rank == 0:
            save_path = os.path.join(model_path, f'DICTr1_Epoch {epoch + 1}.pth')
            torch.save({
                'model': model_without_ddp.state_dict(),  # 模型参数
                'optimizer': optimizer.state_dict(),  # 优化器状态
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch + 1,
                'step': global_step
            }, save_path)
            train_writer.add_scalar('Loss1 per Epoch', total_loss / len(train_loader), epoch)
            train_writer.add_scalar('AEE1 per Epoch', total_aee / len(train_loader), epoch)

            print('Start validation')

            val_results = {}

            results_dict = validate_speckle(model_without_ddp,val_writer,
                                            attn_splits_list=args.attn_splits_list,
                                            corr_radius_list=args.corr_radius_list,
                                            prop_radius_list=args.prop_radius_list,
                                            )

            val_results.update(results_dict)

            logger.write_dict(val_results)

            # Save validation results
            val_file = os.path.join(args.checkpoint_dir, 'val_results1.txt')
            with open(val_file, 'a') as f:
                f.write('global_step: %06d\n' % global_step)

                metrics = ['dataset_AEE', 'dataset_AEE_s0_0.5', 'dataset_AEE_s0.5_1', 'dataset_AEE_s1+']

                eval_metrics = []
                for metric in metrics:
                    if metric in val_results.keys():
                        eval_metrics.append(metric)

                metrics_values = [val_results[metric] for metric in eval_metrics]

                num_metrics = len(eval_metrics)

                # save as Markdown format
                f.write(("| {:>20} " * num_metrics + '\n').format(*eval_metrics))
                f.write(("| {:20.3f} " * num_metrics).format(*metrics_values))

                f.write('\n\n')

        model.train()
        lr_scheduler.step()

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    main(args)
