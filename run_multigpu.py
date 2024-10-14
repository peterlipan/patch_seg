import os
import cv2
import wandb
import torch
import argparse
import numpy as np
import pandas as pd
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from segmentation_models_pytorch.utils.meter import AverageValueMeter
from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import yaml_config_hook
from dataset import PatchDataset, get_training_augmentation, get_validation_augmentation, get_preprocessing


it_classes = ['Background', 'Soft tissue', 'Tumor', 'Bone', 'Marrow', 'Normal cartilage']
ct_classes = ['Background', 'Dedifferentiated', 'G1', 'G2', 'G3']


def gray2rgb(masks, class_rgbs):
    # transform the gray-scale masks to RGB
    # mask: [C, H, W]
    # class_rgbs: [C, 3]
    rgb_image = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8)
    for i in range(masks.shape[0]):
        rgb_image[masks[i, ...] > 0] = class_rgbs[i]
    return rgb_image


def train(epoch, dataloader, model, optimizer, criteria, args, class_rgbs, logger):
    model.train()
    cur_iter = 0
    if isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(epoch)
    
    for iters, (img, label) in enumerate(dataloader):
        img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True)

        prediction = model.forward(img)
        loss = criteria(prediction, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_iter += 1

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))
        
        if args.rank == 0:
            print(f'\rEpoch {epoch}/{args.epochs} || Iter {iters}/{len(dataloader)} || Loss: {loss.item()}', end='', flush=True)
            if cur_iter % 500 == 1 and logger is not None:
                wandb_imgs = img.permute(0, 2, 3, 1).detach().cpu().numpy()[:5]
                wandb_imgs = [(item - np.min(item)) / np.ptp(item) for item in wandb_imgs]
                wandb_imgs = [cv2.resize(item, (224, 224)) for item in wandb_imgs]

                wandb_masks = label.detach().cpu().numpy()[:5]
                wandb_masks = [gray2rgb(item, class_rgbs) for item in wandb_masks]
                wandb_masks = [cv2.resize(item, (224, 224)) for item in wandb_masks]

                wandb_preds = prediction.detach().cpu().numpy()[:5]
                wandb_preds = [gray2rgb(item, class_rgbs) for item in wandb_preds]
                wandb_preds = [cv2.resize(item, (224, 224)) for item in wandb_preds]

                logger.log({'Images': [wandb.Image(item) for item in wandb_imgs],
                            'Masks': [wandb.Image(item) for item in wandb_masks],
                            'Predictions': [wandb.Image(item) for item in wandb_preds]})

            if cur_iter % 50 == 1 and logger is not None:
                logger.log({'train_loss': loss.item()})
                
    return model


def valid(dataloader, model, metrics, logger):
    model.eval()
    logs = {}
    metrics_meters = {
            metric.__name__: AverageValueMeter() for metric in metrics
        }
    with torch.no_grad():
        for img, label in dataloader:
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True)
            prediction = model.forward(img)
            for metric_fn in metrics:
                    metric_value = metric_fn(prediction, label).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
            metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
            logs.update(metrics_logs)
    if logger is not None:
        logger.log(logs)
    return logs


def log_performance(args, iou, acc, f1, precision, recall):
    df_path = './performance.csv'
    if not os.path.exists(df_path):
        df = pd.DataFrame(columns=['task', 'encoder', 'decoder', 'iou', 'acc', 'f1', 'precision', 'recall'])
    else:
        df = pd.read_csv(df_path)
    df = df._append({'task': args.task, 'encoder': args.encoder, 'decoder': args.decoder, 'iou': iou, 'acc': acc, 'f1': f1, 'precision': precision, 'recall': recall}, ignore_index=True)
    df.to_csv(df_path, index=False)


def main(gpu, args, wandb_logger):

    if gpu != 0:
        wandb_logger = None

    rank = args.nr * args.gpus + gpu
    args.rank = rank
    args.device = rank

    if args.world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(rank)

    class_color_csv = pd.read_csv('./class_color_idx.csv')
    classes = it_classes if args.task == 'IdentifyTumor' else ct_classes
    # create segmentation model with pretrained encoder
    model = getattr(smp, args.decoder)(
        encoder_name=args.encoder, 
        encoder_weights=args.weights, 
        classes=len(classes), 
        activation=args.act,
    ).cuda()

    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.weights)

    train_dataset = PatchDataset(
        args.data_root, 
        None, args.train_csv, 
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        class_color_csv=class_color_csv,
        task=args.task,
    )
    class_rgbs = train_dataset.class_rgbs
    
    if args.world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
            )
    else:
        train_sampler = None
    
    train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            drop_last=True,
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
    
    if rank == 0:
        valid_dataset = PatchDataset(args.data_root, None, args.valid_csv, augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn), class_color_csv=class_color_csv, task=args.task,)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    else:
        valid_loader = None

    criteria = utils.losses.DiceLoss()
    metrics = [
        utils.metrics.IoU(threshold=0.5).cuda(),
        utils.metrics.Fscore().cuda(),
        utils.metrics.Accuracy().cuda(),
        utils.metrics.Precision().cuda(),
        utils.metrics.Recall().cuda(),
    ]


    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=args.lr),
    ])

    if args.world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
    
    max_score = 0
    best_iou, best_acc, best_f1, best_precision, best_recall = 0, 0, 0, 0, 0
    for epoch in range(args.epochs):
        model = train(epoch, train_loader, model, optimizer, criteria, args, class_rgbs, wandb_logger)

        if rank == 0:
            logs = valid(valid_loader, model, metrics, wandb_logger)
            str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
            s = ", ".join(str_logs)
            print(f"\nEpoch {epoch}/{args.epochs} || {s}")
            if max_score < logs['iou_score']:
                max_score = logs['iou_score']
                best_iou, best_acc, best_f1, best_precision, best_recall = logs['iou_score'], logs['accuracy'], logs['fscore'], logs['precision'], logs['recall']
                model_name = f'{args.task}_{args.encoder}_{args.decoder}.pth'
                if args.world_size > 1:
                    torch.save(model.module, os.path.join(args.checkpoints, model_name))
                else:
                    torch.save(model, os.path.join(args.checkpoints, model_name))
                print('Model saved!')
        if epoch == 15:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
    if args.rank == 0:
        log_performance(args, best_iou, best_acc, best_f1, best_precision, best_recall)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    yaml_config = yaml_config_hook("./configs/identify.yaml")
    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument('--debug', action="store_true", help='debug mode(disable wandb)')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes

    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    # Master address for distributed data parallel
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    if not args.debug:
        wandb.login(key="cb1e7d54d21d9080b46d2b1ae2a13d895770aa29")
        config = vars(args)

        wandb_logger = wandb.init(
            project="Patch Segmentation",
            config=config
        )

    else:
        wandb_logger = None

    if args.world_size > 1:
        print(
            f"Training with {args.world_size} GPUS, waiting until all processes join before starting training"
        )
        mp.spawn(main, args=(args, wandb_logger, ), nprocs=args.world_size, join=True)
    else:
        main(0, args, wandb_logger,)