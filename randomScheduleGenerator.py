import torch 
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.models as models
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import os
import time
import copy
from multiprocessing import Process, log_to_stderr
import csv
from gossip_module.utils import flatten_tensors, flatten_tensors_grad, unflatten_tensors, unflatten_tensors_grad
from fsdp_custom import FullyShardedDataParallel as FSDP
#from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

from dp_custom import DataParallel_Custom as DP

from auto_wrap_custom import enable_wrap, auto_wrap, wrap
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torch_scheduler import ShardScheduler
import threading
import argparse
import timeit
import numpy as np
import random
from torchsummary import summary

def compareTensors(a_list, b_list, model_summary):
    if len(a_list) != len(b_list) :
        return False
    for a in a_list :
        equal_flag = False
        for b in b_list :
            a_summary = model_summary[a]
            b_summary = model_summary[b]
            equal_flag = equal_flag or (a_summary["name"] == b_summary["name"])
        if not equal_flag:
            return False
    return True 


def checkScheduledList(target, scheduled_list, model_summary):
    for comm in scheduled_list :
        if compareTensors(comm["params"], target["params"], model_summary):
            if target["comm_type"] == comm["comm_type"] :
                print("====================")                 
                print(comm["type"])
                print(comm["comm_type"])
                for param in comm["params"] :
                    param_summary = model_summary[param]
                    print(param_summary["name"])#
                    print(param.shape)
                print("--------------------")
                print(target["type"])
                print(target["comm_type"])
                for param in target["params"] :
                    param_summary = model_summary[param]
                    print(param_summary["name"])# 
                    print(param.shape)

                print("====================")                 
                return True
    return False 

def check_scheduled_comm_type(param_chains, p, check_scheduled, comm_type):
    if(comm_type == "AG_FSDP"):
        param_ends = False
    
        current_tensor = p
        current = param_chains[current_tensor]["current"]

        while not param_ends :
            hash_key = hash(current_tensor) + hash("AG")
            
            if(check_scheduled.get(hash_key, None) == None):
                return False
            current = param_chains[current_tensor]["previous"]
            current_tensor = param_chains[current_tensor]["previous_tensor"]
            if(current == "start"):
                param_ends = True
        return True
    elif(comm_type == "RS") :
        param_ends = False
        current_tensor = p
        current = param_chains[current_tensor]["current"]

        while not param_ends :
            hash_key = hash(param_chains[current_tensor]["current_tensor"]) + hash("AG_FSDP")
            if(check_scheduled.get(hash_key, None) == None):
                return False
            current = param_chains[current_tensor]["next"]
            current_tensor = param_chains[current_tensor]["next_tensor"]
            if(current == "end"):
                param_ends = True
        return True

def validComm(target, scheduled_list, check_scheduled, model_summary, comm, model, param_chains):

    if checkScheduledList(target, scheduled_list, model_summary) :
        #print("duplicated")
        return False
    else :
        if target["comm_type"] == "AG" :
            return True
        elif target["comm_type"] == "AG_FSDP" : 
            for p in comm["params"]:
                hash_key = hash(p) + hash("AG")
                if not check_scheduled_comm_type(param_chains, p, check_scheduled, "AG_FSDP"):
                    return False
            return True
        elif target["comm_type"] == "RS" :
            for p in comm["params"]:
                hash_key = hash(p) + hash("AG_FSDP")
                if not check_scheduled_comm_type(param_chains, p, check_scheduled, "RS"):
                    return False
            return True           


def randomSchedule(model):
    #summary model parameters
    model_summary = {}
    tensor_threshold = 20000
    for n, p in model.named_parameters():
        layer_info = {}
        layer_info["numel"] = p.numel()
        layer_info["partitions"] = (p.numel() // tensor_threshold) + 1
        layer_info["name"] = n
        model_summary[p] = copy.deepcopy(layer_info)

    #FSDP ==> ReduceScatter AllGather AllGather_FSDP, 
    keywords = ["AG", "AG_FSDP", "RS"]
    scheduled_info_dict = {}
    random.seed(50)
    for keyword in keywords :
        for n,p in model.named_parameters():
            key = hash(p) + hash(keyword)
            scheduled_info = copy.deepcopy(model_summary[p])
            #unscheduled_info["param"] = p
            #unscheduled_info["dones"] = 0
            #unscheduled_info["fusion"] = False

            #select scheduling 
            #scheduling_methods = ["TF", "TP", "layer"]
            scheduling_methods = ["TF", "layer"]
            scheduling_method = random.choice(scheduling_methods)
            scheduled_info["scheduled"] = 0

            if(scheduling_method == "TF"):
                scheduled_info["TF"] = True
                scheduled_info["TP"] = False

            elif(scheduling_method == "TP"):
                scheduled_info["TF"] = False
                scheduled_info["TP"] = True
                partition_range = []
                while scheduled_info["dones"] < scheduled_info["partitions"]:
                   rand_max = scheduled_info["partitions"]+1 - scheduled_info["dones"]
                   rand_min = 1
                   partition_num = random.randrange(rand_min, rand_max)
                   partition_range.append(partition_num)
                   scheduled_info["dones"] += partition_num
                scheduled_info["partitions"] = copy.deepcopy(partition_range)
            else :
                scheduled_info["TF"] = False
                scheduled_info["TP"] = False

            scheduled_info_dict[key] = copy.deepcopy(scheduled_info)


    #build communication lists
    communications = []
    #AG, RS, AG_FSDP 


    #communication info 
    #scheduling type : fusion, partitioning, layer

    #communicated parameters : parameter list
    #if scheduling type is partitioning ==> start index and end index

    communication_info = {}
    for keyword in keywords :
        communication_info = dict()
        fusioned_parameter = []
        fusioned_parameter_size = 0
        scheduling_info = {}
        for n,p in model.named_parameters():

            key = hash(p) + hash(keyword)
            scheduling_info = scheduled_info_dict[key]
            if(scheduling_info["TF"] == True):
                fusioned_parameter.append(p)
                fusioned_parameter_size += p.numel()
            #elif(scheduling_info["TP"] == True):
            #    if(fusioned_parameter_size > tensor_threshold )
            #        for() : #add tensor partition cases
            #    else :
            #        fusioned_parameter.append(p)
            #        fusioned_parameter_size += p.numel()                
            elif((scheduling_info["TF"] == False ) and (scheduling_info["TP"] == False)):
            
                if(fusioned_parameter_size > 0):
                    #add tensor fusion cases
                    communication_info["type"] = "TF"
                    #fusioned_parameter.append(p)
                    #fusioned_parameter_size += p.numel()
                else :
                    communication_info["type"] = "layer"

                fusioned_parameter.append(p)
                fusioned_parameter_size += p.numel()   

                communication_info["comm_type"] = keyword
                communication_info["params"] = copy.copy(fusioned_parameter)
                #print(model_summary[communication_info["params"][0]]["name"])
                communication_info["numel"] = fusioned_parameter_size 
                #print(fusioned_parameter_size)
                fusioned_parameter = []
                fusioned_parameter_size = 0    
                communications.append(copy.copy(communication_info))
                #print(model_summary[communications[0]["params"][0]]["name"])
                #communications.append(communication_info)
                #add direct communication cases

        if(scheduling_info["TF"] == True):
            if(len(fusioned_parameter) == 1):
                communication_info["type"] = "layer"       
            else : 
                communication_info["type"] = "TF"
            communication_info["comm_type"] = keyword
            communication_info["params"] = copy.copy(fusioned_parameter)
            #print(model_summary[communication_info["params"][0]]["name"])
            communication_info["numel"] = fusioned_parameter_size 
            #print(fusioned_parameter_size)
            fusioned_parameter = []
            fusioned_parameter_size = 0    
            communications.append(copy.copy(communication_info))     

    #with open("foo.txt", "w") as f:
    #    for comm in communications :    
    #        f.write("--------------------\n")
    #        f.write(comm["type"]+"\n")
    #        f.write(comm["comm_type"]+ "\n")
    #        for param in comm["params"] :
    #            param_summary = model_summary[param]
    #            f.write(param_summary["name"]+"\n")#


    #scheduling communication lists
    scheduled_comms = []
    check_scheduled = {}

    index_list = [x for x in range(0, len(communications))]
    #print(communications)
    #with open("foo.txt", "w") as f:
    param_list = list(model.parameters())
    param_chains = {}
    previous = -1
    count = 0
    previous_access = "start"
    previous_previous_access = 0
    for p in model.parameters() :
        if(count == 1):
            chain = {}
            chain["previous"] = previous_previous_access
            chain["current"] = model_summary[previous_access]["name"] 
            chain["next"] =  model_summary[p]["name"] 

            chain["previous_tensor"] = previous_previous_access
            chain["current_tensor"] = previous_access 
            chain["next_tensor"] =  p    

            param_chains[previous_access] = copy.copy(chain)            
        elif(count > 1):
            chain = {}
            chain["previous"] = model_summary[previous_previous_access]["name"]
            chain["current"] = model_summary[previous_access]["name"] 
            chain["next"] =  model_summary[p]["name"] 

            chain["previous_tensor"] = previous_previous_access
            chain["current_tensor"] = previous_access 
            chain["next_tensor"] =  p 

            param_chains[previous_access] = copy.copy(chain)
        previous_previous_access = previous_access
        previous_access = p         
        count += 1
    chain = {}
    chain["previous"] = model_summary[previous_previous_access]["name"]
    chain["current"] = model_summary[previous_access]["name"] 
    chain["next"] = "end" 

    chain["previous_tensor"] = previous_previous_access
    chain["current_tensor"] = previous_access
    chain["next_tensor"] = "end" 

    param_chains[p] = copy.copy(chain)

    while len(scheduled_comms) < len(communications):
        comm_idx = random.choice(index_list)
        comm = communications[comm_idx]
        print(len(scheduled_comms))
        if(validComm(comm, scheduled_comms, check_scheduled, model_summary, comm, model, param_chains)):
            #f.write(str(comm_idx)+"\n")
            index_list.remove(comm_idx)
            scheduled_comms.append(copy.copy(comm))
            for p in comm["params"] :
                hash_key = hash(p) + hash(comm["comm_type"])
                #print(comm["comm_type"])
                #print(hash("AG"))
                check_scheduled[hash_key] = 1
    #print(len(scheduled_comms))
    #print(len(communications))


    
    #summary schedule result
    #for p in model.parameters():
    #    #print(p)
    #    print(model_summary[p]["name"])


    #with open("foo.txt", "w") as f:
    #    for comm in scheduled_comms :
    #        f.write("--------------------\n")
    #        f.write(comm["type"]+ "\n")
    #        f.write(comm["comm_type"] + "\n")    
    #        for param in comm["params"] :
    #            param_summary = model_summary[param]
    #            f.write(param_summary["name"] + "\n")#

    return scheduled_comms

if __name__ == '__main__':
    model = ResNet(Bottleneck, [3, 8, 36, 3]) #it means "resnet18 model"
    scheduled_comms = randomSchedule(model)
    model_summary = {}
    tensor_threshold = 20000
    for n, p in model.named_parameters():
        layer_info = {}
        layer_info["numel"] = p.numel()
        layer_info["partitions"] = (p.numel() // tensor_threshold) + 1
        layer_info["name"] = n
        model_summary[p] = copy.deepcopy(layer_info)    
    with open("foo.txt", "w") as f:
        for comm in scheduled_comms :
            f.write("--------------------\n")
            f.write(comm["type"]+ "\n")
            f.write(comm["comm_type"] + "\n")    
            for param in comm["params"] :
                param_summary = model_summary[param]
                f.write(param_summary["name"] + "\n")#   
