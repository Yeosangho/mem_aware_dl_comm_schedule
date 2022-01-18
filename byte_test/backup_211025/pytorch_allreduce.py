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
import csv
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
#
#if use_bytescheduler > 0:
#    optimizer = ShardScheduler(model, model.named_parameters(), size, rank, optimizer, args.num_warmup_batches + args.num_iters * args.num_batches_per_iter)



# Set up fake data
datasets = []
for _ in range(100):
    data = torch.rand(args.batch_size, 3, 224, 224)
    target = torch.LongTensor(args.batch_size).random_() % 1000
    if args.cuda:
        data, target = data.cuda(), target.cuda()
    datasets.append(data)
data_index = 0
_grad_accs = []
queue = []

def register_hooks():
	"""Add a hook after the backward propagation of each layer to start allreduce"""
	#for param_group in self.optim.param_groups:
	i = 0
	for p in  model.parameters():
		if p.requires_grad:
			p.grad = p.data.new(p.size()).zero_()
			p_tmp = p.expand_as(p)
			grad_acc = p_tmp.grad_fn.next_functions[0][0]
			#print(f"{i} {grad_acc}", flush=False)
			#grad_acc.register_hook(self._make_hook(p.cpu(), p, i, idx))
			grad_acc.register_hook(make_hook(p, i))	
			_grad_accs.append(grad_acc)
			#grad_acc.register_hook(self._make_hook(p))
			i += 1  
			#i = i % self.thread_num      
def make_hook( p, i):
	def hook(*ignore):
		#res = nvidia_smi.nvmlDeviceGetUtilizationRates(self.smi_handle)
		#print(f'p: {p.shape}', flush=True)
		#print(f"allreduce start!!!", flush=False)
		#if(p.nelement() < 500):
		handle = dist.all_reduce(p, op=dist.ReduceOp.SUM,  async_op=True)
		#else:
		#	handle = dist.all_reduce(p, op=dist.ReduceOp.SUM, group=self.process_groups[0], async_op=True)
		#p_cpu = p.cpu()
		#handle = dist.all_reduce(p, op=dist.ReduceOp.SUM, async_op=True)
		#dist.all_reduce(p_cpu, op=dist.ReduceOp.SUM)
		#p = p_cpu.cuda()
		queue.append(handle)
	return hook  

register_hooks()

def benchmark_step():
    global data_index

    data = datasets[data_index%len(datasets)]
    data_index += 1
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    for i in range(len(queue)):
    	handle = queue.pop(0)
    	#p.data = p_cpu.data.cuda()
    	handle.wait()

    #print(len(list(model.parameters())))
    #for n, param in model.named_parameters():
    #    #with open(f'before_ar_{rank}.csv', 'a', newline='') as f:
    #    #    writer = csv.writer(f)
    #    #    writer.writerow([n, torch.sum(param.grad).item()])          
    #    dist.all_reduce(param.grad)
    #    #with open(f'after_ar_{rank}.csv', 'a', newline='') as f:
    #    #    writer = csv.writer(f)
    #    #    writer.writerow([n, torch.sum(param.grad).item()])          
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

