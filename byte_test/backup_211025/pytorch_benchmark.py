from __future__ import print_function

import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import torch.distributed as dist
from torchvision import models
from torch_scheduler import ShardScheduler

import timeit
import numpy as np
import os
# Benchmark settings
print('111')
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
parser.add_argument('--num-iters', type=int, default=3,
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

#if args.cuda:
#    # Horovod: pin GPU to local rank.
#torch.cuda.set_device(0)

cudnn.benchmark = True

# Set up standard model.
model = getattr(models, args.model)(num_classes=args.num_classes)

if args.cuda:
    # Move model to GPU.
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01)

# Horovod: (optional) compression algorithm.

# bytescheduler wrapper
use_bytescheduler = 1
if use_bytescheduler > 0:
    if args.partition:
        os.environ["BYTESCHEDULER_PARTITION"] = str(1000 * args.partition)
    #from .torch_scheduler import ScheduledOptimizer as bsc
    #bsc.init()

# Horovod: wrap optimizer with DistributedOptimizer.

if use_bytescheduler > 0:
    optimizer = ShardScheduler(model, model.named_parameters(), size, rank, optimizer, args.num_warmup_batches + args.num_iters * args.num_batches_per_iter)



# Set up fake data
datasets = []
for _ in range(100):
    data = torch.rand(args.batch_size, 3, 224, 224)
    target = torch.LongTensor(args.batch_size).random_() % 1000
    if args.cuda:
        data, target = data.cuda(), target.cuda()
    datasets.append(data)
data_index = 0

def benchmark_step():
    global data_index

    data = datasets[data_index%len(datasets)]
    data_index += 1
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()


def log(s, nl=True):
    if rank != 0:
        return
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, size))

# Warm-up
log('Running warmup...')
timeit.timeit(benchmark_step, number=args.num_warmup_batches)

# Benchmark
log('Running benchmark...')
img_secs = []
enable_profiling = args.profiler & (rank == 0)

with torch.autograd.profiler.profile(enable_profiling) as prof:
    for x in range(args.num_iters):
        time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
        img_sec = args.batch_size * args.num_batches_per_iter / time
        log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
        img_secs.append(img_sec)
if enable_profiling:
    prof.export_chrome_trace(os.path.join('pytorch-trace', args.model+'-'+str(rank) +'.json'))
    # print(prof)
# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (size, device, size * img_sec_mean, size * img_sec_conf))

