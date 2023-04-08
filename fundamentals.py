# Virtual environment @ d:\Downloads\Anaconda
# current GPU does not support CUDA (GeForce RTX 2070 super [only 7.5 compute from wiki])
# open anaconda prompt
# activate pytorch to activate

import torch
import numpy as np

print(torch.__version__)
print(torch.cuda.is_available())

x = torch.ones(5, requires_grad=True)
print(x)
