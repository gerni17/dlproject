import torch

cudaAvailable = False
if torch.cuda.is_available():
    cudaAvailable = True
Tensor = torch.cuda.FloatTensor if cudaAvailable else torch.Tensor

def toZeroThreshold(x, t=0.1):
	zeros = Tensor(x.shape).fill_(0.0)
	return torch.where(x > t, x, zeros)

def clamp(x):
	ones = Tensor(x.float().shape).fill_(1.0)
	return torch.where(x.float() <= 1.0, x.float(), ones)