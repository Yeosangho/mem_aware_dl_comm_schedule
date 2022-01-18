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
import time
import math
import torch.distributed as dist
from sklearn.linear_model import LinearRegression

import csv

logging.basicConfig(level=logging.DEBUG)
class ShardScheduler(torch.optim.Optimizer):
    """An optimizer that wraps a hvd._DistributedOptimizer, intercepting allreduce operations and wrap as tasks."""
    def __init__(self, model, named_parameters, size, rank, opt, num_steps=10**6):
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
        self.bucket = torch.zeros(100)
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
        self._locks = {}
        self._conditions = {}
        self._forward_conditions = {}
        for param_group in self.param_groups:
            for p in param_group['params']:
                self._locks[p] = threading.Lock()
                self._conditions[p] = threading.Condition(threading.Lock())
                self._forward_conditions[p] = threading.Condition(threading.Lock())
        

        # The closer to input layer, the higher the priority is.
        self._priority_indexes = {}
        priority = 0
        for p in model.parameters():
            self._priority_indexes[p] = priority
            priority += 1

        assert len(self._grad_accs) == 0
        if self._size > 1:
            self._register_forward_hooks()
            self._register_hooks()

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

        #bucket overload test


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

    def _poll(self):
        """Poll the completion of the tensor's backward or allreduce from a FIFO event_queue"""
        with torch.cuda.stream(self.all_reduce_stream):
            while True:
                #for p,g,h,ctx,cb in list(self.event_queue.queue):
                #   print(f'{ctx} {self._parameter_names[p]}')
                for param_group in self.param_groups:
                    backward_params = param_group['params'][::-1]
                    for p in backward_params:
                        #self.bucket.storage().resize_(p.size().numel())
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
                        #self.bucket.storage().resize_(0)                                   
                        self._locks[p].release()
                        with self._forward_conditions[p] :
                            self._forward_conditions[p].notify_all()

                    

    def _register_forward_hooks(self):
        """Add hook before forward propagation of each layer to block forward computation until the allreduce and
        parameter update is finished. The blocking is implemented using a lock."""
        # Recursively find all submodules
        submodules = []
        q = queue.LifoQueue()
        for mod in self._model.children():
            q.put(mod)
        while not q.empty():
            mod = q.get()
            if len(list(mod.children())) == 0:
                submodules.append(mod)
            else:
                for m in mod.children():
                    q.put(m)

        def pre_forward_hook(mod, input):
            #for lock in self._locks :
            #    print(lock)
            #print("!!!!!!!")
            for p in mod.parameters():
                #print(p)
                if not self._locks[p].locked():
                    #if len(self._handlequeue) > 0 :
                    #    #print(f"pre forward {self._parameter_names[p]}")
                    #    handle = self._handlequeue.pop(0)
                    #    handle.wait()
                    #self._sgd(p)
                    None 
                else :
                    with self._forward_conditions[p] :
                        self._forward_conditions[p].wait()
                        #if len(self._handlequeue) > 0 :
                        #    #print(f"pre forward {self._parameter_names[p]}")
                        #    handle = self._handlequeue.pop(0)
                        #    handle.wait()                        
                    #self._sgd(p)
                    None 

                if p not in self._locks:
                     continue
  
                    #self._locks[p].release()                    
                ##if isinstance(self._opt, torch.optim.SGD):
                #    self._sgd(p)
                #elif isinstance(self._opt, torch.optim.Adam):
                #    self._adam(p)
                #else:
                #    self._logger.error("unknown optimizer!")
                #self._zero_one_grad(p)                
                #with self._locks[p]:
                #    self._logger.debug("{} {} is ready.".format(self._desc, self._parameter_names[p]))
                
                
    

            #print("{} starts forward {}.".format(self._desc, mod))

        def after_forward_hook(mod, input, result):
            self._logger.debug("{} finished forward {}.".format(self._desc, mod))

        # Register pre-hook and hook for each module
        for mod in reversed(submodules):
            self._logger.debug("{} registers forward hook on module {}".format(self._desc, mod))
            mod.register_forward_pre_hook(pre_forward_hook)
            mod.register_forward_hook(after_forward_hook)

    def _register_hooks(self):
        """Add a hook after the backward propagation of each layer to start allreduce"""
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _make_hook(self, p):
        """Define hook for backward propagation. The hook wraps the allreduce OP as a ByteTask and
        posts it to Core.

        Arguments:
            p: the parameter.
        """
        self._logger.debug("{} calls make_hook for {}".format(self._desc, self._parameter_names[p]))

        def hook(*ignore):
            self._locks[p].acquire()
            with self._conditions[p]:
                self._conditions[p].notify_all()
        return hook

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
