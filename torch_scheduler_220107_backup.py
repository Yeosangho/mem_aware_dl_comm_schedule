from __future__ import absolute_import
import os
import threading
import logging
import time
try:
    import queue
except ImportError:
    import Queue as queue
import torch
from bytecore_custom import core
from torch_task import TorchTask, BYTESCHEDULER_LIB
from torch.nn.parameter import Parameter

import time
import math
import torch.distributed as dist
import csv
from fairscale.utils.parallel import (
    chunk_and_pad,
    enable_pytorch_sync_bn,
    get_process_group_cached,
    validate_process_group,
)
#logging.basicConfig(level=logging.DEBUG)
class ShardScheduler(torch.optim.Optimizer):
    """An optimizer that wraps a hvd._DistributedOptimizer, intercepting allreduce operations and wrap as tasks."""
    def __init__(self, model, named_parameters, size, rank, opt, partition_threshold, done_counts, partition_counts, locks, conditions, forward_conditions, num_steps=10**6, comm_stream=None):
        """Construct a new ScheduledOptimizer, which uses horovod optimizer under the hood for averaging gradients
         across all the Horovod ranks.

        Args:
            model: The training model. ByteScheduler uses the model object to register hooks.
            hvd_opt: Optimizer to use for averaging gradients and applying updates.
            num_steps: The maximum number of training steps. ByteScheduler needs to know when to stop cross-iteration
            scheduling.

        Usage example:
        ```
        import bytescheduler.pytorch.horovod as bsc
        bsc.init()
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters, compression)
        optimizer = bsc.ScheduledOptimizer(model, optimizer, num_steps)
        ```
        """
        print("!!!!!!!!!!!!!!!!!!!!")
        #handle = BYTESCHEDULER_LIB.bytescheduler_create_event(0)
        #super(self.__class__, self).__init__(model.parameters())
        self._model = model
        self._size= size
        self._rank = rank
        self._opt = opt
        self._logger = logging.getLogger("ByteScheduler")
        self._logger.debug("hvd size {}, rank {}".format(size, rank))
        self._desc = "rank {}".format(rank)
        self._grad_accs = []
        self._requires_update = set()
        self._handles = {}
        #self._handlequeue = queue.Queue()
        self._handlequeue = []
        # Track training steps
        self._step = 0
        self._final_step = num_steps

        
        self.partition_threshold = partition_threshold
        self.done_counts = done_counts
        self.partition_counts = partition_counts

        self._locks = locks
        self._conditions = conditions
        self._forward_conditions = forward_conditions
        self.comm_stream = comm_stream
        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = [(f'allreduce.noname.{i}.{j}', v)
                                for i, param_group in enumerate(self.param_groups)
                                for j, v in enumerate(param_group['params'])]
        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')

        dups = ShardScheduler.find_duplicates([k for k, _ in named_parameters])
        if len(dups) > 0:
            raise ValueError('Parameter names in named_parameters must be unique. '
                             'Found duplicates: %s' % ', '.join(dups))

        all_param_ids = {id(v)
                         for param_group in self.param_groups
                         for v in param_group['params']}
        named_param_ids = {id(v) for k, v in named_parameters}
        unnamed_param_ids = all_param_ids - named_param_ids
        if len(unnamed_param_ids):
            raise ValueError('named_parameters was specified, but one or more model '
                             'parameters were not named. Python object ids: '
                             '%s' % ', '.join(str(id) for id in unnamed_param_ids))
        backward_passes_per_step=1
        self._parameter_names = {v: k for k, v in sorted(named_parameters)}
        self.backward_passes_per_step = backward_passes_per_step
        self._allreduce_delay = {v: self.backward_passes_per_step
                                 for _, v in sorted(named_parameters)}


        # Use lock to block the forward propagation of each parameter.

        

        # The closer to input layer, the higher the priority is.
        self._priority_indexes = {}
        priority = 0
        for p in model.parameters():
            self._priority_indexes[p] = priority
            priority += 1



        # Poll whether the tensor is ready for allreduce or whether the allreduce is finished.
        self.event_queue = queue.Queue()
        self.all_reduce_stream = torch.cuda.Stream()
        self._poller = threading.Thread(target=self._poll, args=())
        self._poller.start()

        # Let rank 0 decide the communication order.
        self._immediate = False
        #if self._rank != 0:
        #    self._immediate = True

        #core.start(self._parameter_names, rank=self._rank, arch="allreduce")

    @staticmethod
    def find_duplicates(lst):
        seen = set()
        dups = set()
        for el in lst:
            if el in seen:
                dups.add(el)
            seen.add(el)
        return dups

    def __getattr__(self, item):
        return getattr(self._opt, item)

    def __del__(self):
        """Clean up"""
        self.event_queue.put((None, None, None, None, None))
        self._poller.join()
        #core.shutdown(wait_for_all=False)

    def step(self, closure=None):
        """Override the default step function."""
        self._logger.debug("{} calls step() {}".format(self._desc, self._step))
        #for i in range(self._handlequeue.qsize()) :
    	#    handle = self._handlequeue.get()
    	#    handle.wait()
        #for i in self._handlequeue :
    	#    handle = self._handlequeue.pop(0)
    	#    handle.wait()        
        # Step 0 is called for parameter initialization after parameter broadcast
        if self._size > 1 and self._step > 0:
            # if it is the final training step, wait for the completion of all tensors
            if self._step == self._final_step:
                self._logger.debug("final step {}, waiting for allreduce completion.".format(self._final_step))
                while not self.event_queue.empty():
                    time.sleep(0.001)
            loss = None
            if closure is not None:
                loss = closure()
            self._step += 1
            return loss
        else:
            # SGD.step() will be triggered when user calls hvd.broadcast_optimizer_sate()
            #super(self._opt.__class__, self._opt).step()
            self._opt.step()
            self._step += 1

        #for i in self._handlequeue :
    	#    handle = self._handlequeue.pop(0)
    	#    #p.data = p_cpu.data.cuda()
    	#    handle.wait()
        #self._opt.step()
        #self._step += 1

    def zero_grad(self):
        """Override the default zero_grad function

        Clears the gradients of all optimized :class:`torch.Tensor` s.
        """
        self._logger.debug("{} calls zero_grad() of step {}".format(self._desc, self._step))
        if self._size > 1 and self._step > 0:
            return
        else:
            self._opt.zero_grad()

    def allreduce_grad_async(self, tensor, name):
        """Call horovod API to allreduce gradient asynchronously

        Arguments:
            tensor: The tensor to be allreduced.
            name: The name of the tensor.

        Returns:
            an allreduce handle and context
        """
        #ctx = tensor.type()
        ctx = name
        #print(f'{self._desc} before allreduce {name}  :  {torch.sum(tensor)}')
        #with open(f'before_{self._desc}.csv', 'a', newline='') as f:
        #    writer = csv.writer(f)
        #    writer.writerow([name, torch.sum(tensor).item()])
        handle = dist.all_reduce(tensor,async_op=True)
        self._handlequeue.put(handle)
        return handle, ctx




    #comms with tensor partition
    def _poll(self):
        """Poll the completion of the tensor's backward or allreduce from a FIFO event_queue"""
        with torch.cuda.stream(self.comm_stream):
            while True:
                #for p,g,h,ctx,cb in list(self.event_queue.queue):
                #   print(f'{ctx} {self._parameter_names[p]}')
                for param_group in self.param_groups:
                    backward_params = param_group['params'][::]
                    for p in backward_params:  
                        #print(p)
                        #while self.partition_counts[p] > self.done_counts[p] :                 
                        if self._locks[p].locked():
                            #print(f"_poll {self._parameter_names[p]}")
                            #handle = dist.all_reduce(p.grad, async_op=True)
                            #self._handlequeue.append(handle)
                            dist.all_reduce(p.grad)
                        else : 
                            with self._conditions[p] :
                                self._conditions[p].wait()
                                #handle = dist.all_reduce(p.grad, async_op=True)
                                #self._handlequeue.append(handle)
                                dist.all_reduce(p.grad)  
                        
                        p.grad = p.grad / 2
                        print(f"output p.grad[0] {p.grad.shape} {torch.sum(p.grad)}")

                        #self._finalize_parameters(p)                            
                        self._adam(p)
                        #self._sgd(p)
                        self._zero_one_grad(p)
                        #p.grad = None                                                    
                        self._locks[p].release()
                        with self._forward_conditions[p] :
                            self._forward_conditions[p].notify_all()
                print(f"after backward  {torch.cuda.memory_allocated() / 1024 /1024}")             

    def _poll_wfbp(self):
        """Poll the completion of the tensor's backward or allreduce from a FIFO event_queue"""
        with torch.cuda.stream(self.comm_stream):
            while True:
                #for p,g,h,ctx,cb in list(self.event_queue.queue):
                #   print(f'{ctx} {self._parameter_names[p]}')
                for param_group in self.param_groups:
                    backward_params = param_group['params'][::]
                    for p in backward_params:  
                        #print(p)                  
                        if self._locks[p].locked():
                            #print(f"_poll {self._parameter_names[p]}")
                            #handle = dist.all_reduce(p.grad, async_op=True)
                            #self._handlequeue.append(handle)
                            dist.all_reduce(p.grad)
                        else : 
                            with self._conditions[p] :
                                self._conditions[p].wait()
                                #handle = dist.all_reduce(p.grad, async_op=True)
                                #self._handlequeue.append(handle)
                                dist.all_reduce(p.grad)  
                                
                        p.grad = p.grad / 2
                        print(f"output p.grad[0] {p.grad.shape} {torch.sum(p.grad)}")

                        #self._finalize_parameters(p)                            
                        self._adam(p)
                        #self._sgd(p)
                        self._zero_one_grad(p)
                        #p.grad = None                                                    
                        self._locks[p].release()
                        with self._forward_conditions[p] :
                            self._forward_conditions[p].notify_all()
                print(f"after backward  {torch.cuda.memory_allocated() / 1024 /1024}")     

    def _poll_RS(self):
        """Poll the completion of the tensor's backward or allreduce from a FIFO event_queue"""
        with torch.cuda.stream(self.comm_stream):
            while True:
                #for p,g,h,ctx,cb in list(self.event_queue.queue):
                #   print(f'{ctx} {self._parameter_names[p]}')
                for param_group in self.param_groups:
                    #backward_params = param_group['params'][::-1]
                    backward_params = param_group['params'][::]
                    for p in backward_params:  
                        #print(f"enter poll {self._parameter_names[p]}")                 
                        if self._locks[p].locked():
                            None
                            #grad = p.grad.data
                            #grad_chunks = chunk_and_pad(grad, 2)
                            #p.grad.data = torch.zeros_like(grad_chunks[0]).type(p.grad.dtype).to(p.device)
                            #print(f"reduce scatter lock  {self._parameter_names[p]}")
                            #dist.reduce_scatter(p.grad, grad_chunks, async_op=False)  
                            
                        else : 
                            with self._conditions[p] :
                                #print("wait in poll!!")
                                self._conditions[p].wait() 
                                #grad = p.grad.data
                                #grad_chunks = chunk_and_pad(grad, 2)
                                #p.grad.data = torch.zeros_like(grad_chunks[0]).type(p.grad.dtype).to(p.device)
                                #print(f"reduce scatter condition {self._parameter_names[p]}")
                                #dist.reduce_scatter(p.grad, grad_chunks, async_op=False)  
                        grad = p.grad.data
                        grad_chunks = chunk_and_pad(grad, 2)
                        p.grad.data = torch.zeros_like(grad_chunks[0]).type(p.grad.dtype).to(p.device)

                    ###
                        dist.reduce_scatter(p.grad, grad_chunks, async_op=False)  
                        print(f"output p.grad[0] {p.grad.shape} {torch.sum(p.grad)}")

                        #p.grad = None        
                        self._post_reduction_hook(p, p.grad.data)
                        self._finalize_parameters(p)
                        self._adam(p)
                        #self._sgd(p)
#
                        self._zero_one_grad(p)
                        #p.grad = None
                        grad = None
                        grad_chunks = None     
                        #torch.cuda.synchronize()
                        #p_data = p.data.to(p._full_param_padded.device)
                        #p_size = p._full_param_padded.size()
                        #p_data.new_zeros(p_size)
                        #p._full_param_padded.storage().resize_(p_size.numel())
                        #output_tensor = p._full_param_padded
                        #print(f"all_gather output tensor {output_tensor.shape}")
                        #print(f"all gather input tensor {p_data.shape}")
                        #dist._all_gather_base(output_tensor, p_data)
                        #p._full_param_padded.storage().resize_(0)
                        #p_data = None
                        #p_data = None
                        #print(f"after update {p.data.shape} {torch.sum(p.data)}")                            
                        #output_tensor.record_stream(torch.cuda.current_stream())
                        self._locks[p].release()
                        with self._forward_conditions[p] :
                            self._forward_conditions[p].notify_all()
                        #print(f"release lock {self._parameter_names[p]}")
                    print(f"after backward  {torch.cuda.memory_allocated() / 1024 /1024}") 

    def _post_reduction_hook(self, param: Parameter, reduced_grad: torch.Tensor) -> None:
        """Hook to call on each param after the reduce-scatter."""

        if param._is_sharded:
            # Accumulate into the gradient shard.
            if getattr(param, "_saved_grad_shard", None) is None:
                param._saved_grad_shard = reduced_grad.data
            else:
                assert (
                    param._saved_grad_shard.shape == reduced_grad.shape
                ), f"{param._saved_grad_shard.shape} vs {reduced_grad.shape}"
                param._saved_grad_shard.data += reduced_grad.data
            reduced_grad = param._saved_grad_shard.data



    def _finalize_parameters(self, p):
        if not p.requires_grad:
            return
        if hasattr(p, "_shard_bwd_hook"):
            assert len(p._shard_bwd_hook) == 2, len(p._shard_bwd_hook)
            p._shard_bwd_hook[1].remove()
            delattr(p, "_shard_bwd_hook")

        # Leave the gradient accumulation state as-is if not synchronizing this pass. This ensures p.grad
        # remains the unsharded gradient accumulated from prior no-sync passes, and p._saved_grad_shard
        # remains the sharded gradient from the last synchronized pass. This also allows interleaved no-sync and
        # sync passes, if desired.
        #if not self._require_backward_grad_sync:
        #    return

        # Parameter and gradient devices must match.
        if hasattr(p, "_cpu_grad"):
            assert p.device == torch.device("cpu")
            p.grad = p._cpu_grad
        elif hasattr(p, "_saved_grad_shard"):
            assert p.device == p._saved_grad_shard.device

            p.grad = p._saved_grad_shard
    
        if hasattr(p, "_saved_grad_shard"):
            delattr(p, "_saved_grad_shard")

    def _zero_one_grad(self, p):
        """Clears the gradient of one variable as PyTorch accumulates gradients by default.

        Arguments:
            p: the parameter.
        """
        if p.grad is not None:
            # Not sure whether to do detach_ or not
            p.grad.detach_()
            p.grad.zero_()

    """Below are the implementations of optimizers, e.g., SGD, Adam."""

    def _sgd(self, p):
        """Performs a single optimization step using SGD optimizer on a parameter.

        Arguments:
            p: The parameter to be updated.
        """
        # TODO: support other optimizers later, or figure out a walk around way
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for gp in group['params']:
                if self._parameter_names[p] != self._parameter_names[gp] or gp.shape != p.shape:
                    continue
                self._logger.debug("{} is updating {}".format(self._desc, self._parameter_names[p]))
                if p.grad is None:
                    continue
                d_p = p.grad.data
                #print(self._parameter_names[p])
                #print(p.data.shape)
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-group['lr'], d_p)
                #d_p = None
                #print(f"p.shape & data {p.shape} {p.data[0]}")
                break

    def _adam(self, p):
        """Performs a single optimization step using Adam optimizer on a parameter.

        Arguments:
            p: The parameter to be updated.
        """
        for group in self.param_groups:
            for gp in group['params']:
                if self._parameter_names[p] != self._parameter_names[gp] or gp.shape != p.shape:
                    continue
                self._logger.debug("{} is updating {}".format(self._desc, self._parameter_names[p]))
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0

                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)

                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)

                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)
                break


def init():
    """Replace _register_hook() function in hvd._DistributedOptimizer with empty function."""

    def hijack(obj, func_name):
        orig_func = getattr(obj, func_name)
        print("hijack function {}".format(orig_func))

        def wrapped_func(*args, **kwargs):
            print("function {} is hijacked to do nothing.".format(orig_func))
            return
        setattr(obj, func_name, wrapped_func)

    #hijack(hvd._DistributedOptimizer, '_register_hooks')
