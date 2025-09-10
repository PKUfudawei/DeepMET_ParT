import torch, wandb, argparse, os, random
import torch.optim as optim
import numpy as np
import torch.distributed as dist

# official repository
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from glob import glob
from tqdm import tqdm
from ema_pytorch import EMA

# self-used repository
from utils.data import Dataset, cycle
from utils.helper import fix_seeds, save_model, load_model
from models.particle_transformer import ParticleTransformer


def parse_arguments():
    parser = argparse.ArgumentParser(description='Hyper parameters and setup in training.')
    parser.add_argument('-g', '--gpus', type=str, default=None, help="comma separated list of GPU ids to use, e.g. '0,2,3'")
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('--max_PF_num', type=int, default=2**10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--project', type=str, default='MET')
    parser.add_argument('--total_epochs', type=int, default=20)
    parser.add_argument('--ema_start', type=int, default=100)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--use_wandb', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pretrained_model', default=None)
    args = parser.parse_args()

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        os.environ['OMP_NUM_THREADS'] = str(args.num_workers)
    args.use_wandb = bool(args.use_wandb)
    args.name = f'b={args.batch_size}_lr={args.lr}_EMAdecay={args.ema_decay}' + ('_distill' if args.pretrained_model is not None else '')
    return args


def _check_finite(t, name):
    if not torch.isfinite(t).all():
        raise RuntimeError(f"[Non-finite detected] {name} contains NaN/Inf")


def train_batch(inputs, model, targets, optimizer, status='train', ema=None):
    _check_finite(inputs, "inputs")
    _check_finite(targets, "targets")
    if status == 'train':
        model.train()
        outputs = model(inputs)
        _check_finite(outputs, "outputs(train)")
        loss = torch.nn.functional.mse_loss(outputs, targets)
        _check_finite(loss, "loss(train)")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if ema is not None:
            ema.update()
    elif status == 'test':
        if ema is not None:
            ema_model = ema.ema_model
            ema_model.eval()
        else:
            model.eval()
        with torch.no_grad():
            if ema is not None:
                outputs = ema_model(inputs)
            else:
                outputs = model(inputs)
            _check_finite(outputs, "outputs(test)")
            loss = torch.nn.functional.mse_loss(outputs, targets)
            _check_finite(loss, "loss(test)")
    else:
        raise ValueError(f"Invalid status: {status}. Must be 'train' or 'test'.")
        
    return {f'{status}_loss': loss.item()}


if __name__ == "__main__":
    # --- 初始化分布式 ---
    args = parse_arguments()
    use_single_card = (torch.cuda.device_count() <= 1)

    if use_single_card:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
    else:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f'cuda:{local_rank}')

    # 固定随机种子
    fix_seeds(seed=42)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.benchmark = True

    # 数据集
    train_files = glob('data/train/DYJetsToMuMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/*.hdf5')
    test_files = glob('data/test/DYJetsToMuMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/*.hdf5')
    dataset_train = Dataset(files=train_files, max_PF_num=args.max_PF_num)
    dataset_test = Dataset(files=test_files, max_PF_num=args.max_PF_num)

    if use_single_card:
        train_sampler, test_sampler = None, None
    else:
        train_sampler = DistributedSampler(dataset_train, shuffle=True)
        test_sampler = DistributedSampler(dataset_test, shuffle=False)

    dataloader_train = DataLoader(
        dataset=dataset_train, batch_size=args.batch_size, sampler=train_sampler, 
        num_workers=args.num_workers, pin_memory=True, drop_last=True, shuffle=use_single_card,
    )
    dataloader_test = DataLoader(
        dataset=dataset_test, batch_size=args.batch_size, sampler=test_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True, shuffle=False,
    )

    # 模型
    model = ParticleTransformer(
        input_dim=len(dataset_train.features), output_dim=len(dataset_train.truths),
        pair_input_dim=4, use_pre_activation_pair=False, embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64], num_heads=8, num_layers=8,
        num_cls_layers=2, block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[], activation='gelu', trim=True, for_inference=False,
    ).to(device)

    if args.pretrained_model is not None:
        model = load_model(model, args.pretrained_model)

    if not use_single_card:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    ema = EMA(model if use_single_card else model.module, beta=args.ema_decay, update_after_step=args.ema_start, update_every=1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    # 只有 rank=0 初始化 wandb
    rank = 0 if use_single_card else dist.get_rank()
    if args.use_wandb and rank == 0:
        wandb.init(project=args.project, mode='offline', name=args.name)

    # 训练循环
    for e in range(args.total_epochs):
        if not use_single_card:
            train_sampler.set_epoch(e)
            test_sampler.set_epoch(e)
        if rank == 0:
            train_loader = tqdm(dataloader_train, desc=f"Epoch {e}")
            test_loader = iter(dataloader_test)
        else:
            train_loader = dataloader_train

        for index, (X, Y) in enumerate(train_loader):
            X, Y = X.to(device), Y.to(device)
            log_train = train_batch(X, model, Y, optimizer, status='train', ema=ema)

            if rank == 0:
                step = e * len(dataloader_train) + index
                if args.use_wandb:
                    wandb.log(log_train, step=step)

                # test
                if step % (len(dataloader_train)//10) == 0:
                    try:
                        X_test, Y_test = next(test_loader)
                    except StopIteration:
                        test_loader = iter(dataloader_test)
                        X_test, Y_test = next(test_loader)
                    X_test, Y_test = X_test.to(device), Y_test.to(device)
                    log_test = train_batch(X_test, model, Y_test, optimizer, status='test', ema=ema)

                    save_model(ema.ema_model, step=step, output_save_dir=f'./checkpoints/{args.project}/{args.name}')
                    if args.use_wandb:
                        wandb.log(log_test, step=step)

    dataset_train.close()
    dataset_test.close()

    if args.use_wandb and rank == 0:
        wandb.finish()

    if not use_single_card:
        dist.destroy_process_group()
