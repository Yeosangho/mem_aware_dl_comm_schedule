import torch 
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.models as models
import torch.nn as nn
import os
import time
from multiprocessing import Process
import csv
from gossip_module.utils import flatten_tensors, flatten_tensors_grad, unflatten_tensors, unflatten_tensors_grad
from fsdp_custom import FullyShardedDataParallel as FSDP
from auto_wrap_custom import enable_wrap, auto_wrap, wrap
from torchvision.models.resnet import BasicBlock, ResNet
import argparse
global a 
a = 0 
def module_check(module):
	global a 

	if (len(list(module.children())) == 0 ):
		a += 1
		print(module)
		for param in module.parameters() :
			print(param.data.size())
		#print(module.data.size())
	for name, child in module.named_children():
		module_check(child)

	return 	
if __name__ == '__main__':
	model = ResNet(BasicBlock, [2, 2, 2, 2])
	model.cuda()
	count = 0
	for name, p in model.named_parameters() :
		if 'weight' in name :
			print(name)
			count += 1
	print(count)
	module_check(model)
	print(a)
