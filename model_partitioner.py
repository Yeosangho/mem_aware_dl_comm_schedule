import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision.models.resnet import BasicBlock, ResNet

from sklearn.linear_model import LinearRegression
import lstm as lstmpy
import time
import os
import numpy as np
import csv


class Partitioner():
    def __init__(self, model):
        #profiling bandwidth saturation point 
        self.bandwidth_profiler = NetworkProfiler(size)


        #get communication model
        self.alpha = self.bandwidth_profiler.getAlpha()
        self.beta = self.bandwidth_profiler.getBeta()

        #get partition size
        self.partition_unit = self.bandwidth_profiler.getSaturationSize()
         
        #get layer size and layer computation time 

        f_keys, f_times, f_param_sizes, b_keys, b_times, b_param_sizes = benchmark(model)
        
        self._parameter_names = {v: k for k, v in model.named_parameters()}
        self._module_names = {v: k for k, v in model.named_modules() if(len(list(v.children()))== 0 and len(list(v.parameters()))> 0)} 
        self._module_keys = [k for k, v in model.named_modules() if(len(list(v.children()))== 0 and len(list(v.parameters()))> 0)]  
        self._seq_keys = [k for k, v in model.named_parameters()]

        self._module_params_map = {}
        for name, m in model.named_modules() :
            if(len(list(m.children()))== 0 and len(list(m.parameters()))> 0) :
                self._module_params_map[name] = []
                for n, p in m.named_parameters() : 
                    self._module_params_map[name].append(n)


        ##make all possible cases
        #1. Initialize layer communication
        init_comms = {}
        for key in self._module_keys :
            layer_allreduce = Model_Partition(key, )
            init_comms[key] = layer_allreduce
        

    def partition_model_level(self):
        pass

    def partition_in_layer(self):
        pass

    def getPartitions(self):
        pass

class Model_Partition():
    def __init__(self, name, pre_op, post_op, param_size):
        self.name = name
        self.pre_op = pre_op   #start point
        self.post_op = post_op #wait point
        self.param_size = param_size

class ModelProfiler():
    def __init__(self, model):
        if isinstance(model, torch.nn.Module) is False:
            raise ValueError("Not a valid model, please provide a 'nn.Module' instance.")

        self.model = model
        self._parameter_names = {v: k for k, v
                                in model.named_parameters()}

        self._module_keys = [k for k, v in model.named_modules() if(len(list(v.children()))== 0 and len(list(v.parameters()))> 0)]                        
        self._seq_keys = [k for k, v in model.named_parameters()]
        self._backward_seq_keys = []
        self._backward_key_sizes = []
        self._forward_seq_keys = []
        self._forward_key_sizes = []
        self._grad_accs = []
        self._forward_handles = {}
        self._backward_handles = {}
        self.hook_done = False
        self._start = time.time()
        self._register_hooks()
        self._is_profiling = False

    def _register_hooks(self):
        #for name, p in self.model.named_parameters():
        #    p.register_hook(self._make_hook(name, p))  
        self._register_backward_hooks(self.model)      
        self._register_forward_hooks(self.model)


        #p.register_forward_hook(self._make_forward_hook(name, p))

    #def _make_hook(self, name, p):
    #    def hook(*ignore):
    #        if not self._is_profiling:
    #            return
    #        #p_numel = 0
    #        #for p in p.parameters():
    #        #    p_numel += p.numel()
    #        name = self._parameter_names.get(p)
    #        if len(self._backward_seq_keys) != len(self._seq_keys):
    #            self._backward_seq_keys.append(name)
    #            self._backward_key_sizes.append(p.numel())
    #        if name not in self._backward_handles:
    #            self._backward_handles[name] = []
    #        torch.cuda.synchronize()
    #        ct = self._timestamp(name)
    #        self._backward_handles[name].append(ct - self._start)
    #    return hook

    def _make_hook(self, name, module):
        def hook(*ignore):
            if not self._is_profiling:
                return
            p_numel = 0
            for p in module.parameters():
                p_numel += p.numel()
#
            if len(self._backward_seq_keys) != len(self._module_keys):
                self._backward_seq_keys.append(name)
                self._backward_key_sizes.append(p_numel)
            if name not in self._backward_handles:
                self._backward_handles[name] = []
            torch.cuda.synchronize()
            ct = self._timestamp(name)
            self._backward_handles[name].append(ct - self._start)
        return hook

    def _make_forward_hook(self, name, module):
        def forward_hook(*ignore):
            #print(f"@@@@{len(self._forward_seq_keys)} and {len(self._seq_keys)}")
            if not self._is_profiling:
                return
            p_numel = 0
            for p in module.parameters():
                p_numel += p.numel()
            #name = self._parameter_names.get(p)
            print(f"####{name}")
            if len(self._forward_seq_keys) != len(self._module_keys):
                self._forward_seq_keys.append(name)
                self._forward_key_sizes.append(p_numel)
            if name not in self._forward_handles:
                self._forward_handles[name] = []
            torch.cuda.synchronize()
            ct = self._timestamp(name)
            self._forward_handles[name].append(ct - self._start)
        return forward_hook

    def _register_backward_hooks(self, module, name=None):
        i = 0
        for name, m in module.named_modules():
            print(f"{i}in register backward hook {name}")
            if(len(list(m.children()))== 0 and len(list(m.parameters()))> 0):
                i += 1
                print(f"{i} register_backward_hook")
                m.register_backward_hook(self._make_hook(name, m)) 

    def _register_forward_hooks(self, module, name=None):
        i = 0
        for name, m in module.named_modules():
            print(f"{i}in register forward hook {name}")
            if(len(list(m.children()))== 0 and len(list(m.parameters()))> 0):
                i += 1
                print(f"{i} register_forward_hook")
                m.register_forward_hook(self._make_forward_hook(name, m))      
    def reset_start(self):
        self._start = time.time()

    def reset(self):
        self._start = time.time()
        self._handles.clear()

    def stop(self):
        self._is_profiling = False

    def start(self):
        self._is_profiling = True
        self._start = time.time()

    def get_backward_seq_keys(self):
        return self._backward_seq_keys

    def get_backward_key_sizes(self):
        return self._backward_key_sizes

    def get_forward_seq_keys(self):
        return self._forward_seq_keys

    def get_forward_key_sizes(self):
        return self._forward_key_sizes

    def get_layerwise_times(self):
        print(self._forward_seq_keys)
        num_trials = len(self._forward_handles[self._forward_seq_keys[0]])
        forward_layerwise_times_multipletest = []
        forward_totals = []
        for j in range(num_trials):
            s = 0
            total = 0.0
            layerwise_times = [] # from the last layer to the first layer
            #for i, k in enumerate(self._seq_keys[::-1]):
            for i, k in enumerate(self._forward_seq_keys):
                t = self._forward_handles[k][j]
                #print('name: ', k, ' diff: ', t-s)
                layerwise_times.append(t-s)
                total += (t-s)
                s = total
            forward_layerwise_times_multipletest.append(layerwise_times)
            forward_totals.append(total)
        array = np.array(forward_layerwise_times_multipletest)
        forward_layerwise_times = np.mean(array, axis=0)

        num_trials = len(self._backward_handles[self._backward_seq_keys[0]])
        backward_layerwise_times_multipletest = []
        backward_totals = []
        for j in range(num_trials):
            s = 0
            total = 0.0
            layerwise_times = [] # from the last layer to the first layer
            #for i, k in enumerate(self._seq_keys[::-1]):
            for i, k in enumerate(self._backward_seq_keys):
                t = self._backward_handles[k][j]
                #print('name: ', k, ' diff: ', t-s)
                layerwise_times.append(t-s)
                total += (t-s)
                s = total
            backward_layerwise_times_multipletest.append(layerwise_times)
            backward_totals.append(total)
        array = np.array(backward_layerwise_times_multipletest)
        backward_layerwise_times = np.mean(array, axis=0)

        return forward_layerwise_times, np.mean(forward_totals), backward_layerwise_times, np.mean(backward_totals)

    def _timestamp(self, name):
        return time.time()
    

class NetworkProfiler():
    def __init__(self, world_size):
        self.utilized_bandwidths = []
        self.message_sizes = []
        self.times = []
        self.saturated_bandwidths = [] 
        self.max_bandwidth_idx = []
        self.diff_allreduce_time = []
        self.all_reduce_stream = torch.cuda.Stream()
        self.saturation_point = 0.97
        self.max_bandwidth = 0
        self.unit_size = 5000
        self.world_size = world_size
        prev_allreduce_time = 0
        for i in range(300):
            
            t_size = self.unit_size*(i+1)
            #t_size = 1024*1024*64
            t_list = []

            #set up fake data
            datasets = []
            for _ in range(10):
                data = torch.rand(t_size).cuda()
                datasets.append(data)    
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for itr in range(10):
                start.record()
                with torch.cuda.stream(self.all_reduce_stream):            

                        #print(f"before {t[0]}")
                        dist.all_reduce(datasets[itr], op=dist.ReduceOp.SUM)
                        #handle.wait()
                torch.cuda.default_stream().wait_stream(self.all_reduce_stream)

                #dist.barrier()
                end.record()
                torch.cuda.synchronize()

                allreduce_time = start.elapsed_time(end) /( 1000.0)
                #self.diff_allreduce_time.append(allreduce_time - prev_allreduce_time)
                #prev_allreduce_time = allreduce_time
                self.times.append(allreduce_time)

                avg = 1.0*sum(self.times)/len(self.times)
                #exec_time += 1.0*sum(times)
                self.used_bandwidth = ((t_size*4)/avg)/(1024**3)              
                self.utilized_bandwidths.append(self.used_bandwidth)
                self.message_size = t_size*4/(1024**2)
                self.message_sizes.append(self.message_size)
        t_utilized_bandwidths  = torch.tensor(self.utilized_bandwidths).cuda() / self.world_size
        t_times = torch.tensor(self.times).cuda() / self.world_size
        t_message_sizes = torch.tensor(self.message_sizes).cuda() / self.world_size       
        dist.all_reduce(t_utilized_bandwidths, async_op=False)
        dist.all_reduce(t_times, async_op=False)
        dist.all_reduce(t_message_sizes, async_op=False)      
        self.utilized_bandwidths = t_utilized_bandwidths.tolist()
        self.times = t_times.tolist()
        self.message_sizes = t_message_sizes.tolist()

        self.max_bandwidth = max(self.utilized_bandwidths)    
        self.saturated_bandwidths = [(bandwidth / self.max_bandwidth) for bandwidth in self.utilized_bandwidths] 
        self.max_bandwidth_idx = [int(i) for i in range(len(self.utilized_bandwidths)) if self.utilized_bandwidths[i] >= self.max_bandwidth*self.saturation_point ]
        #print(self.diff_allreduce_time[int(self.max_bandwidth_idx[0]+1)])
        Y = np.array(self.times[self.max_bandwidth_idx[0]:])
        #beta = np.array(self.beta)
        X = np.array(self.message_sizes[self.max_bandwidth_idx[0]:]).reshape((-1,1)) 
        model = LinearRegression()
        model.fit(X,Y) 
        self.alpha = model.intercept_
        self.beta = model.coef_[0]


    def getSaturationBandwidth(self):
        return self.max_bandwidth

    def getSaturationPoint(self):
        return self.max_bandwidth_idx[0]
    def getSaturationSize(self):
        return self.message_sizes[self.max_bandwidth_idx[0]]
    def getAlpha(self):
        return np.average(self.alpha)

    def getBeta(self):
        return np.average(self.beta)    

def benchmark(model):
    # Benchmark to achieve the backward time per layer
    p = ModelProfiler(model)
    # Warmup
    warmup = 5 # warmup should be 0 on some GPUs (e.g., P102-100)
    iteration = 50
    batch_size = 32
    image_size = 256
    label_class = 1000
    criterion = nn.CrossEntropyLoss().cuda()
    for i in range(iteration+warmup):
        inputs = torch.rand(batch_size, 3, image_size, image_size).cuda()
        labels = torch.rand(batch_size).cuda() * label_class
        labels = labels.type(torch.cuda.LongTensor)
        #inputs, labels = inputs.cuda(non_blocking=True), labels_cpu.cuda(non_blocking=True)

        # forward + backward + optimize
        outputs = model(inputs)
        #print(outputs.shape)
        #print(labels.shape)
        loss = criterion(outputs, labels)
        torch.cuda.synchronize()

        if i >= warmup:
            p.start()
        loss.backward()
        torch.cuda.synchronize()
    forward_layerwise_times, forward_sum_total, backward_layerwise_times, backward_sum_total = p.get_layerwise_times()
    forward_seq_keys = p.get_forward_seq_keys()
    backward_seq_keys = p.get_backward_seq_keys()
    p.stop()
    return forward_seq_keys[::-1], forward_layerwise_times[::-1], p.get_forward_key_sizes()[::-1], backward_seq_keys, backward_layerwise_times, p.get_backward_key_sizes()    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--rank', type=int, default=None)
    parser.add_argument('--size', type=int, default=None)

    args = parser.parse_args()
    rank = args.rank
    size = args.size

    os.environ['MASTER_ADDR'] = '210.107.197.167'
    os.environ['MASTER_PORT'] = '30000'
    dist.init_process_group('nccl', rank=rank, world_size=size)
    cudnn.benchmark = True

    model = ResNet(BasicBlock, [2, 2, 2, 2]) #it means "resnet18 model"
    model.cuda()
    i = 0
    #for n, p in model.named_parameters():
    #    print(n)
    #    i += 1
    #print(i)
    #i = 0
    #for idx, m in (model.named_modules()):
    #    if(len(list(m.children()))== 0 and len(list(m.parameters()))> 0):
    #        print(idx)
    #        i += len(list(m.parameters()))
    #print(i)
    #f_keys, f_times, f_param_sizes, b_keys, b_times, b_param_sizes = benchmark(model)
    #print(f"#### forward  {len(f_keys)}####")
    #for i in range(len(f_keys)):
    #    print(f"{f_keys[i]}, {f_param_sizes[i]}, {f_times[i]}")
#
    #print(f"#### backward  {len(b_keys)}####")
    #for i in range(len(b_keys)):
    #    print(f"{b_keys[i]}, {b_param_sizes[i]}, {b_times[i]}")        
    #bandwidth_profiler = NetworkProfiler(size)
    ##print(f'{self._desc} before allreduce {name}  :  {torch.sum(tensor)}')
    #with open(f'bandwidth_profiler.csv', 'a', newline='') as f:
    #    writer = csv.writer(f)
    #    for i in range(len(bandwidth_profiler.message_sizes)):
    #        writer.writerow([bandwidth_profiler.utilized_bandwidths[i], bandwidth_profiler.message_sizes[i], bandwidth_profiler.saturated_bandwidths[i]*100, bandwidth_profiler.times[i]])
    #        
    #print(bandwidth_profiler.alpha)
    #print(bandwidth_profiler.beta)

    p = Partitioner(model)
    p.getPartitions()

    