import torch, wandb, argparse, os
import torch.optim as optim
import numpy as np

from glob import glob
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.data import Dataset, cycle
from models.particle_transformer import ParticleTransformer


def parse_arguments():
    parser = argparse.ArgumentParser(description='Hyper parameters and setup in training.')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--project', type=str, default='MET')
    parser.add_argument('--test_eval_freq', type=int, default=0.5)
    parser.add_argument('--total_epochs', type=int, default=20)
    parser.add_argument('--EMA_start', type=int, default=1)
    parser.add_argument('--EMA_mu', type=float, default=0.999)
    parser.add_argument('--use_wandb', type=int, default=1)
    parser.add_argument('--pretrained_model', default=None)

    args = parser.parse_args()
    args.use_wandb = bool(args.use_wandb)
    args.name = f'b={args.batch_size}_lr={args.lr}_EMAmu={args.EMA_mu}' + ('_distill' if args.pretrained_model is not None else '')
    return args


def fix_seeds(seed=42):
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False


def save_model(model, train_iterations, output_save_dir):
    os.makedirs(os.path.join(output_save_dir, 'model'), exist_ok=True)

    # save yaml file for later runs   
    save_dir_model = os.path.join(output_save_dir, f'model/checkpoint_{train_iterations}.pt')
    save_obj = {'model': model.state_dict()}
    # save model to path
    with open(save_dir_model, 'wb') as f:
        torch.save(save_obj, f)
    print(f'\ncheckpoint saved to {output_save_dir}.')


def load_model(model, path, strict = True):
    # to avoid extra GPU memory usage in main process when using Accelerate
    with open(path, 'rb') as f:
        loaded_obj = torch.load(f, map_location='cpu')
    try:
        model.load_state_dict(loaded_obj['model'], strict = strict)
    except RuntimeError:
        print('Failed loading state dict.')
    print('\nCheckpoint loaded from {}'.format(path))
    return model


class EMA:
    def __init__(self, start=1000, mu=0.999):
        self.start = start
        self.mu = mu
        self.shadow = {}
        self.backup = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def apply(self, module, backup=True):
        for name, param in module.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                if backup:
                    self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].data)

    def restore(self, module):
        assert hasattr(self, 'backup')
        for name, param in module.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup = {}

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.apply(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def train_batch(iteration, inputs, model, targets, optimizer, args, status='train'):
    if status == 'train':
        model.train()
    elif status == 'valid':
        model.eval()
    else:
        raise ValueError(f"Invalid status: {status}. Must be 'train' or 'valid'.")

    outputs = model(inputs)
    loss = torch.nn.functional.mse_loss(outputs, targets)

    # update model
    if status == 'train':
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

    elif status == 'valid':
        save_model(model, train_iterations=iteration, output_save_dir=f'./output/{args.project}/{args.name}')
        
    # return log
    log = {
        f'{status}_loss': loss.item(),
    }
    return log


if __name__ == "__main__":
    # parse commandline arguments
    args = parse_arguments()
    
    # torch prerequisites
    fix_seeds(seed=42)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    use_double = False
    if use_double:
        torch.set_default_dtype(torch.float64)

    # inputs and targets
    features = ['PF_pt', 'PF_eta', 'PF_phi', 'PF_mass', 'PF_d0', 'PF_dz', 'PF_hcalFraction', 'PF_pdgId', 'PF_charge', 'PF_fromPV', 'PF_puppiWeightNoLep', 'PF_puppiWeight', 'PF_px', 'PF_py']
    targets = ['truth_px', 'truth_py']

    # initialize model
    model = ParticleTransformer(
        # input/output dimensions
        input_dim=len(features), output_dim=len(targets),  
        # network configurations
        pair_input_dim=4, use_pre_activation_pair=False, embed_dims=[128, 512, 128], pair_embed_dims=[64, 64, 64],
        num_heads=8, num_layers=8, num_cls_layers=2, block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0}, fc_params=[], activation='gelu',
        # misc
        trim=True, for_inference=False,
    ).to(device=device)

    if args.pretrained_model is not None:
        model = load_model(model, args.pretrained_model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {num_params}')

    # initialize PIFM
    ema = EMA(start=args.EMA_start, mu=args.EMA_mu)
    ema.register(model)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # initialize wandb
    if args.use_wandb:
        wandb.init(project=args.project, mode='offline', name=args.name)

    for e in range(args.total_epochs):
        for start_ratio in np.arange(0, 1, 0.001):
            end_ratio = start_ratio + 0.001

            dataset_train = Dataset(
                glob('data/train/*/*.parquet'),
                features=features, targets=targets, start_ratio=start_ratio, end_ratio=end_ratio,
            )
            dataset_test = Dataset(
                glob('data/test/*/*.parquet'),
                features=features, targets=targets, start_ratio=start_ratio, end_ratio=end_ratio,
            )
            dataloader_train = cycle(DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=False))
            dataloader_test = cycle(DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=True, drop_last=False))

            for i in range(len(dataset_train)//args.batch_size+1):
                # train
                inputs, targets = next(dataloader_train)
                inputs, targets = inputs.to(device=device), targets.to(device=device)
                log_train = train_batch(e+start_ratio, inputs, model, targets, optimizer, args, status='train')
                if args.use_wandb:
                    wandb.log(log_train, step=e+start_ratio)
                if e + start_ratio > ema.start:
                    ema.update(model)

                # evaluate
                if start_ratio % args.test_eval_freq == 0:
                    ema.apply(model)
                    inputs_test, targets_test = next(dataloader_test)
                    inputs_test, targets_test = inputs_test.to(device=device), targets_test.to(device=device)
                    log_test = train_batch(e+start_ratio, inputs_test, model, targets_test, optimizer, args, status='test')
                    
                    if args.use_wandb:
                        wandb.log(log_test, step=e+start_ratio)

                    ema.restore(model)

    # finalize wandb
    if args.use_wandb:
        wandb.finish()