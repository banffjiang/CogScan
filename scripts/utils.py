import random
import os
import numpy as np
import torch

def seed_everything(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# batch_size = 64
# epochs = 7
# lr = 5e-5
# gamma = 0.7
# seed = 42