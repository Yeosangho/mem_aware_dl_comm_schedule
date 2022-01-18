import torch 
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.models as models
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import os
import time
from multiprocessing import Process, log_to_stderr
import csv
from gossip_module.utils import flatten_tensors, flatten_tensors_grad, unflatten_tensors, unflatten_tensors_grad
from fsdp_custom import FullyShardedDataParallel as FSDP
#from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

from dp_custom import DataParallel_Custom as DP

from auto_wrap_custom import enable_wrap, auto_wrap, wrap
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torch_scheduler import ShardScheduler
from randomScheduleGenerator import randomSchedule
import threading
import argparse
import timeit
import numpy as np
from torchsummary import summary
import copy
def module_check(module):
	#if (len(list(module.children())) == 0 ):
		#print(module)
		#print(module.data.size())
	for name, child in module.named_children():
		module_check(child)


class Trainer:
	def __init__(self, world_size, rank, shard, precision):


		torch.backends.cudnn.benchmark = True
		#world_size = int(os.environ["WORLD_SIZE"])
		self.world_size = world_size
		print(f'world_size : {world_size}')
		ngpus_per_node = torch.cuda.device_count()
		self.shard = shard
		#rank = int(os.environ['SLURM_PROCID'])
		self.rank = rank
		
		print(f'rank : {rank}')


		self.device = torch.device("cuda:"  + str(rank%ngpus_per_node))
		torch.cuda.set_device(rank%ngpus_per_node)
		print("cuda:"  + str(rank%ngpus_per_node))
		self.process_groups = []
		world_list = [x for x in range(world_size) ]

		#self.process_groups = []
		#world_list = [x for x in range(world_size) ]
		#for i in range(self.thread_num):
		#    ng = dist.new_group(world_list, backend='gloo')
		#    self.process_groups.append(ng) 

		self.batch_size = 16
		self.image_size = 42
		self.classification_num = 1000
		#self.model = models.resnet101()
		print(f"before init model  {torch.cuda.memory_allocated() / 1024 /1024}") 
		self.model = ResNet(Bottleneck, [3, 8, 36, 3]) #it means "resnet18 model"
		self.model.cuda()
		print(f"after init model  {torch.cuda.memory_allocated() / 1024 /1024}") 

		self._rs_locks = {}
		self._ag_locks = {}
		self._ag_fsdp_locks = {}

		self._rs_conditions = {}
		self._ag_conditions = {}
		self._ag_fsdp_conditions = {} 

		self._forward_conditions = {}
		self._backward_conditions = {}

		#check lazy init
		self._lazy_init_locks = {}
		self._lazy_init_conditions = {}

		self._partition_counts = {}
		self._scheduled_comms = {}
		self._done_counts = {}

		self.model_parameter_names = {}
		module_check(self.model)

		self.datasets = []
		self.target = None
		self.data_index = 0
		print(f"before init dataset  {torch.cuda.memory_allocated() / 1024 /1024}") 
		
		for _ in range(100):
		    data = torch.rand(self.batch_size, 3, 80, 80)
		    self.target = torch.LongTensor(self.batch_size).random_() % 1000
		    data, self.target = data.cuda(), self.target.cuda()
		    self.datasets.append(data)
		print(f"after init dataset  {torch.cuda.memory_allocated() / 1024 /1024}") 

		summary(self.model, ( 3, 80, 80))
		self.profiled_memory_utilization = []
		wrap_cls = DP
		if(self.shard == 0):
			wrap_cls = DP
		elif(self.shard == 1):
			wrap_cls = FSDP
		mixed_precision_bool = False
		if(precision == 0):
			mixed_precision_bool = False
		else :
			mixed_precision_bool = True
		self.comm_stream = torch.cuda.Stream()
		self.fsdp_params = dict(wrapper_cls=FSDP, mixed_precision=False, flatten_parameters=True, 
								done_counts=self._done_counts, partition_counts=self._partition_counts, 

								rs_locks=self._rs_locks, ag_locks=self._ag_locks, ag_fsdp_locks=self._ag_fsdp_locks, 

								rs_conditions=self._rs_conditions, ag_conditions=self._ag_conditions, ag_fsdp_conditions=self._ag_fsdp_conditions,
								forward_conditions=self._forward_conditions, backward_conditions=self._backward_conditions, 

								lazy_init_locks=self._lazy_init_locks, lazy_init_conditions=self._lazy_init_conditions, 

								memory_record=self.profiled_memory_utilization, comm_stream=self.comm_stream,
								
								model_parameter_names=self.model_parameter_names
								)
		#self.fsdp_params = dict(wrapper_cls=FSDP, mixed_precision=False, flatten_parameters=False,  memory_record=self.profiled_memory_utilization)
		self.sharded_module = None
		self.optimizer = None
		self.criterion = None
		self.partition_threshold = 20000
		
		with enable_wrap(**self.fsdp_params):
			self.sharded_module = auto_wrap(self.model)
			self._scheduled_comms = randomSchedule(self.sharded_module)
			for n, p in self.sharded_module.named_parameters():
				#print(p.numel())
				self._partition_counts[p] = (p.numel() // self.partition_threshold) + 1
				self._done_counts[p] = 0

				self._rs_locks[p] = threading.Lock()
				self._ag_locks[p] = threading.Lock()
				self._ag_locks[p].acquire()
				self._ag_fsdp_locks[p] = threading.Lock()

				self._rs_conditions[p] = threading.Condition(threading.Lock())
				self._ag_conditions[p] = threading.Condition(threading.Lock())
				self._ag_fsdp_conditions[p] = threading.Condition(threading.Lock())

				self._forward_conditions[p] = threading.Condition(threading.Lock())
				self._backward_conditions[p] = threading.Condition(threading.Lock())

				self._lazy_init_locks[p] = threading.Lock()
				self._lazy_init_conditions[p] = threading.Condition(threading.Lock())

				self.model_parameter_names[p] = n
		#model_summary = {}
		#for n, p in self.sharded_module.named_parameters():
		#	layer_info = {}
		#	layer_info["numel"] = p.numel()
		#	layer_info["partitions"] = (p.numel() // 20000) + 1
		#	layer_info["name"] = n
		#	model_summary[p] = copy.deepcopy(layer_info)    
		#with open("foo.txt", "w") as f:
		#	for comm in scheduled_comms :
		#		f.write("--------------------\n")
		#		f.write(comm["type"]+ "\n")
		#		f.write(comm["comm_type"] + "\n")    
		#		for param in comm["params"] :
		#			param_summary = model_summary[param]
		#			f.write(param_summary["name"] + "\n")#       

		#self.sharded_module = FSDP(self.model, memory_record=self.profiled_memory_utilization)
		print(f"before init optimizer  {torch.cuda.memory_allocated() / 1024 /1024}") 
		#self.optimizer = torch.optim.SGD(self.sharded_module.parameters() , lr=0.001, momentum=0.9, nesterov=True)
		self.optimizer = torch.optim.Adam(self.sharded_module.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
		self.optimizer = ShardScheduler(self.sharded_module, self.sharded_module.named_parameters(), self.world_size, self.rank, self.optimizer,
		                                self.partition_threshold, self._done_counts, self._partition_counts,

										self._rs_locks, self._ag_locks, self._ag_fsdp_locks,

										self._rs_conditions, self._ag_conditions, self._ag_fsdp_conditions,
										
										self._forward_conditions, self._backward_conditions,

										self._lazy_init_locks, self._lazy_init_conditions,

										10**6, self.comm_stream, self._scheduled_comms)
		print(f"after init optimizer  {torch.cuda.memory_allocated() / 1024 /1024}") 
		
		self.criterion = nn.CrossEntropyLoss()

		#if(wftp == True):
		#	self._register_hooks()
		self.scaler = GradScaler()

	def benchmark_step(self):
		with enable_wrap(**self.fsdp_params):
			data = self.datasets[self.data_index%len(self.datasets)]
			self.data_index += 1
			print(f"before forward  {torch.cuda.memory_allocated() / 1024 /1024}") 
			output = self.sharded_module(data)
			print(f"after forward  {torch.cuda.memory_allocated() / 1024 /1024}") 
	#
			loss = self.criterion(output,self.target)
			print(f"before backward  {torch.cuda.memory_allocated() / 1024 /1024}") 
	#
			loss.backward()
			
			#print(len(self.profiled_memory_utilization))
#
			#self.optimizer.step()
			#self.optimizer.zero_grad()
			#torch.cuda.empty_cache()

			#self.optimizer.zero_grad()
			#data = self.datasets[self.data_index%len(self.datasets)]
			##with autocast():
			#output = self.model(data)
			#loss = self.criterion(output,self.target)
			#print(torch.cuda.memory_allocated() / 1024 /1024) 	
			#loss.backward()
			#self.optimizer.step()
			##self.scaler.scale(loss).backward()
			##self.scaler.step(self.optimizer)
#
			##self.scaler.update()
			#print(torch.cuda.memory_allocated() / 1024 /1024) 
			#torch.cuda.empty_cache()
		#self.profiled_memory_utilization = self.profiled_memory_utilization[:0]	
	def train(self):
		f_times = []
		b_times = []
		itr_times = []
		proc = None
		for itr in range(2):
			print(itr)

			batch = torch.rand(self.batch_size, 3, self.image_size, self.image_size).cuda()
			t = torch.randint(0, self.classification_num-1, (self.batch_size,))
			#a = torch.zeros((self.batch_size, self.classification_num))
			#a[:, t] = 1
			target = t.cuda().type(torch.cuda.LongTensor)

			with enable_wrap(**self.fsdp_params):
				output = self.sharded_module(batch)
				loss = self.criterion(output,target)
				loss.backward()
	
	
				#self.optimizer.step()
				#self.optimizer.zero_grad()

	
if __name__ == '__main__':
	os.environ['MASTER_ADDR'] = '210.107.197.218'
	os.environ['MASTER_PORT'] = '30000'
	parser = argparse.ArgumentParser()
	parser.add_argument('--rank', dest='rank', default=0, type=int)
	parser.add_argument('--shard', dest='shard', default=0, type=int)
	parser.add_argument('--mixed_precision', dest='mixed_precision', default=0, type=int)

	args = parser.parse_args()

	world_size = 2
	rank = args.rank
	shard = args.shard
	mixed_precision = args.mixed_precision 

	#world_size = int(os.environ["WORLD_SIZE"])
	#rank = int(os.environ['SLURM_PROCID'])		
	dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
	#case 1 
	trainer = Trainer(world_size, rank, shard, mixed_precision)
	
	img_secs = [] 
	print(torch.cuda.memory_allocated() / 1024 /1024) 
	for x in range(1):
		time = timeit.timeit(trainer.benchmark_step, number=5)
		img_sec = 32 * 10 / time
		print('Iter #%d: %.1f img/sec per'  % (x, img_sec))
		img_secs.append(img_sec)
	
	shard_tag = "DP" if shard == 0 else "FSDP"
	mixed_precision_tag = "full_preicision" if mixed_precision == 0 else "mixed_precision"
	
	with open(f'{shard_tag}_{mixed_precision_tag}_memory_utilization.csv', 'w', newline='') as f:
		writer = csv.writer(f)
		for i in range(len(trainer.profiled_memory_utilization)):
			#print(f"{i} : {trainer.profiled_memory_utilization[i]}")
			writer.writerow([trainer.profiled_memory_utilization[i]])

	img_sec_mean = np.mean(img_secs)
	img_sec_conf = 1.96 * np.std(img_secs)
	print('Img/sec : %.1f +-%.1f' % ( img_sec_mean, img_sec_conf))
	print('Total img/sec on %d (s): %.1f +-%.1f' % (world_size, world_size * img_sec_mean, world_size * img_sec_conf))
