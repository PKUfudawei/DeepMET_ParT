import torch, os
import numpy as np

def fix_seeds(seed=42):
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False


def save_model(model, step, output_save_dir):
    os.makedirs(output_save_dir, exist_ok=True)

    # save yaml file for later runs
    save_dir_model = os.path.join(output_save_dir, f'checkpoint_{step}.pt')
    save_obj = {'model': model.state_dict()}
    # save model to path
    with open(save_dir_model, 'wb') as f:
        torch.save(save_obj, f)
    #print(f'\ncheckpoint saved to {output_save_dir}.')


def load_model(model, path, strict=True):
    # to avoid extra GPU memory usage in main process when using Accelerate
    with open(path, 'rb') as f:
        loaded_obj = torch.load(f, map_location='cpu')
    try:
        model.load_state_dict(loaded_obj['model'], strict = strict)
    except RuntimeError:
        print('Failed loading state dict.')
    print('\nCheckpoint loaded from {}'.format(path))
    return model
