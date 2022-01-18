#!/usr/bin/python

from __future__ import absolute_import
import sys

try:
    import queue
except ImportError:
    import Queue as queue
import threading
import logging
import collections
from bytetask import ByteTask
from bytescheduler.common.tuner import Tuner
from bytescheduler.common.profiler import Profiler
import torch.distributed as dist

import os
import time

class ByteCore_Custom(object):
    """The core of ByteScheduler. Once Core gets a ByteTask (which represents a communication operation, e.g., push,
    allreduce), it partitions the ByteTask and decides when to send each partition according to priority."""

    def __init__(self, logger=None):
        """
        Args:
            logger: ByteScheduler logger object
        """
        if logger is None:
            self._logger = logging.getLogger("ByteScheduler")
        else:
            self._logger = logger

        # A priority queue of ByteTask, tasks are sorted according to its priority.
        self._queue = queue.Queue()

        # Scheduler thread
        self._scheduler = threading.Thread(target=self._loop, args=())
        self._scheduler.daemon = True
        self._is_started = False

        # DATA represents normal tasks and EXIT signals the scheduler thread to be terminated.
        self._commands = {'DATA': 0, 'EXIT': 1}

        # Control credit
        self._condition = threading.Condition(threading.Lock())

        # Pending tasks that are not ready
        self._pending = set()
        self._pending_lock = threading.Lock()

        # Only used to avoid task being garbage collected before completion.
        self._running = set()
        self._finished = collections.OrderedDict()

        # The rank of a worker
        self._rank = None

        # The communication architecture used, e.g., ps or allreduce.
        self._arch = None

        # Partition unit, i.e., the number of parameters
        self._partition = int(os.environ.get('BYTESCHEDULER_PARTITION', 1000000))

        # Credit, i.e., the max number of unacknowledged parameters
        self._credit = float(os.environ.get('BYTESCHEDULER_CREDIT', 4000000))
        self._credit_limit = self._credit

        # We expect that the first key is same across iterations and we use it to count how many training steps have
        # been run.
        self._first_key = None
        self._step = 0

        # Tuning
        self._credit_tuning = int(os.environ.get('BYTESCHEDULER_CREDIT_TUNING', 1))
        self._partition_tuning = int(os.environ.get('BYTESCHEDULER_PARTITION_TUNING', 0))
        self._tuner = None

        # hyper parameters of auto-tuning.
        self._current_point = {
            "credit": self._credit,
        }
        self._next_point = None

        # profiling
        self._timeline = os.environ.get('BYTESCHEDULER_TIMELINE', '')
        self._profiler = None
        self._parameter_names = []
        self._idx = 0
        self._handlequeue = queue.Queue()
    def start(self, model, param_names, rank, arch):
        """Start core.
        Args:
            rank: the rank of the worker
            arch: the communication architecture, "ps" or "allreduce"
        """
        if self._is_started:
            self._logger.warning("Core is already started.")
            return
        self._rank = rank
        self._parameter_names = param_names
        self._idx = len(self._parameter_names) - 1
        #self._idx = 0
        self._value_list = list(self._parameter_names.values())
        # Setup profiler
        if self._rank == 0 and self._timeline:
            self._logger.info("rank {}: profiler is enabled.".format(self._rank))
            self._profiler = Profiler(self._timeline)
        else:
            self._profiler = Profiler('')

        assert arch == "ps" or arch == "allreduce", arch + " not supported!"
        self._arch = arch

        # Support tuning partition for allreduce
        if self._partition_tuning:
            assert arch == "allreduce", "Do not support partition tuning for ps."
            self._current_point["partition"] = self._partition

        if (rank == 0 and self._credit_tuning) or self._partition_tuning:
            self._tuner = Tuner(rank=self._rank, arch=arch, credit_tuning=self._credit_tuning,
                                partition_tuning=self._partition_tuning, logger=self._logger)

        self._scheduler.start()
        self._is_started = True

        self._logger.info(
            "start Core {}: credit {}, partition {}, credit tuning {}, partition tuning {}.".format(
                self._rank, self._credit, self._partition, self._credit_tuning, self._partition_tuning))

    def shutdown(self, wait_for_all=False):
        """Shut Core down.

        Args:
            wait_for_all: Flag indicating whether to wait completion of undone tasks.
        """
        if not self._is_started:
            self._logger.warning("Core is already shutdown.")
            return
        #if wait_for_all:
        #    self._queue.put((sys.maxint, self._commands['EXIT'], None))
        #else:
        #    self._queue.put((-sys.maxint, self._commands['EXIT'], None))
        with self._condition:
            self._credit = sys.maxint
            self._condition.notify_all()
        self._scheduler.join()
        self._is_started = False
        if self._tuner:
            self._tuner.exit()
        self._profiler.stop()
        self._logger.info("shutdown Core {}.".format(self._rank))

    def post(self, lock, grad, name):
        """Post a communication task to Core for scheduling.
        Args:
            task: a ByteTask object
        Returns:
            A boolean value indicating whether the task is successfully posted
        """

        self._queue.put((lock, grad, name))
            # The callback runs when a non-immediate task is ready.
            #def _start_callback(task, self):
            #    with self._pending_lock:
            #        self._pending.remove(task)
            #    self._profiler.put(task.name, task.op + 'QUEUE', 'B')
            #    #print(f'before put {task.name}')
            #    with self._condition:
            #        self._queue.put((task.priority, self._commands['DATA'], task))
            #        if task.name == self._value_list[self._idx] :
            #            self._condition.notify_all()
            #            #print(f'start call back notify_all {task.name}')
            #    self._logger.debug(
            #        "{} has been posted into Core with priority {}".format(task.desc, task.priority))

            # Prepare the task, i.e., add dependency Proxies.
                #with self._pending_lock:
                #    self._pending.add(t)
                #t.prepare(callback=_start_callback, callback_context=self)
                #with self._condition:
                    
                    #self._condition.notify_all() 
                    #if t.name == self._value_list[self._idx] :
                    #    self._condition.notify_all()                    
        return True

    def _loop(self):
        """The main scheduling logic is a while loop that pops a task from queue each time and do it if credit is enough.
        The credit decreases when a task is running and increases when a task is finished.
        """

        # The callback runs when a non-immediate task is finished.

        while True:
            #    self._condition.wait() 
            #while self._queue.empty() :
            #    time.sleep(0.001)
            #lock, grad, name = self._queue.get() 
            #while True:            
            #    try:
            #        priority, cmd, task = self._queue.get(False)
            #    except:
            #        continue
            #    if task.name != self._value_list[self._idx] :
            #        self._queue.put((priority, cmd, task))
            #        # wait for (potential) new credit
            #        time.sleep(0.001)
            #    else:
            #        #print(f"pop queue {task.name}")
            #        break

            #self._profiler.put(task.name, task.op + 'QUEUE', 'E')
            #self._running.add(task)
            self._idx -= 1
            #self._idx += 1
            self._idx = self._idx % len(self._value_list)
            #print(f'all_reduce {task.name}')         
            handle = dist.all_reduce(grad, async_op=False)
            self._handlequeue.put(handle)
            #lock.release()
            #self._profiler.put(task.name, task.op + 'COMMUNICATION', 'B')
            #self._running.remove(task)
            #self._finished[task.name] = task
            #self._profiler.put(task.name, task.op + 'COMMUNICATION', 'E')
       

    def _tune(self):
        if self._tuner.stopped and self._next_point is None:
            self._tuner.exit()
            return
        # Only rank 0 runs auto-tuning algorithm
        if self._rank == 0:
            self._tuner.record(self._current_point, self._step)
        if self._next_point is None:
            self._next_point = self._tuner.next_point()
        if self._next_point is not None and self._step == self._next_point["step"]:
            with self._condition:
                if "credit" in self._next_point:
                    self._credit_limit = self._next_point["credit"]
                    self._credit = self._next_point["credit"]
                    self._logger.info("core {}: autotuning sets credit to {}K at training step {}.".format(
                            self._rank, int(self._credit / 1000), self._step))
                if "partition" in self._next_point:
                    self._partition_unit = self._next_point["partition"]
                    self._logger.info("core {}: autotuning sets partition to {}K at training step {}.".format(
                            self._rank, int(self._partition / 1000), self._step))
                self._current_point = self._next_point
                self._next_point = None

# Init a core once the module is imported
core = ByteCore_Custom()
