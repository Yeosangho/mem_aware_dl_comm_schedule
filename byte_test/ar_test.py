from __future__ import print_function
import os 
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import torch.distributed as dist
from torchvision import models

parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--model', type=str, default='resnet18',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='number of classes')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--profiler', action='store_true', default=False,
                    help='disables profiler')
parser.add_argument('--partition', type=int, default=1000,
                    help='partition size')
parser.add_argument('--rank', type=int, default=None)
parser.add_argument('--size', type=int, default=None)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
rank = args.rank
size = args.size


os.environ['MASTER_ADDR'] = '210.107.197.167'
os.environ['MASTER_PORT'] = '30000'
dist.init_process_group('nccl', rank=rank, world_size=size)

a = torch.ones([4]).cuda()
b = torch.ones([2,4]).cuda()
handle1 = None
handle2 = None
if(rank == 0):
    dist.all_reduce(a, async_op=False)
    dist.all_reduce(b, async_op=False)
else :
    dist.all_reduce(b, async_op=False)
    dist.all_reduce(a, async_op=False)


#handle1.wait()
#handle2.wait()
#print(handle1.is_completed())

print(a)
print(b)
