import cvxpy
import copy
import numpy as np

class Block():
    def __init__(self, start_time, comm_time, end_time, layer, partition=False, start_flag=False, end_flag=False, name=""):
        self.maximum_start_time = start_time
        self.comm_time = comm_time
        self.minimum_end_time = end_time
        self.communicated_layer = layer
        self.start_time = 0
        self.pseudo_start_time = self.start_time
        self.end_time = 0
        self.bandwidth_utilization_per_layer = [0,0,0,0,0]
        if(partition):
            self.start_flag_layer = [0,0,0,0,0]
            self.end_flag_layer = [0, 0, 0, 0, 0]
            if(start_flag):
                self.start_flag_layer = [1 if elem>0 else 0 for elem in layer]
            elif(end_flag):
                self.end_flag_layer = [1 if elem>0 else 0 for elem in layer]
        else:
            self.start_flag_layer = layer
            self.end_flag_layer = layer
        self.name = name

    def __eq__(self, other):
        """Overrides the default implementation"""
        comparisons = self.communicated_layer == other.communicated_layer
        return comparisons


    def set_schedule_time(self, start_time):
        self.start_time = start_time
        if(self.start_time > 0):
            self.pseudo_start_time = 10 - self.start_time 
        else:
            self.pseudo_start_time = 10
    
        self.end_time = self.start_time + self.comm_time
        if(self.minimum_end_time < self.end_time):
            self.end_time = self.minimum_end_time
        self.bandwidth_utilization_per_layer = [0,0,0,0,0]
        start_idx = 0 
        end_idx = 0
        for comp_time in computation_time_per_layers:
            if(comp_time < self.start_time) :
                start_idx += 1
            if(comp_time < self.end_time):
                end_idx += 1
        #print(f"end_time : {end_idx}, start_time : {start_idx}")                 
        if(start_idx == end_idx):
            self.bandwidth_utilization_per_layer[start_idx] = bandwidth * (self.end_time - self.start_time)
        if(start_idx < end_idx):
            self.bandwidth_utilization_per_layer[start_idx] = bandwidth * (computation_time_per_layers[start_idx] - self.start_time)
            self.bandwidth_utilization_per_layer[end_idx] = bandwidth * (computation_time_per_layers[end_idx] - self.end_time)
            for i in range(start_idx+1, end_idx):
               self.bandwidth_utilization_per_layer[i] = bandwidth * (computation_time_per_layers[i] - computation_time_per_layers[i-1]) 
               #print(bandwidth * (computation_time_per_layers[i] - computation_time_per_layers[i-1]) )
        #print(self.bandwidth_utilization_per_layer)
    def update_maximum_start_time(self, start_time):
        self.maximum_start_time = max(start_time, self.maximum_start_time)
    def update_minimum_end_time(self, end_time):
        self.minimum_end_time = min(end_time, self.minimum_end_time)
    def fusion(self, start_time, end_time, comm_time, layer):
        update_maximum_start_time(start_time)
        update_minimum_end_time(end_time)
        self.comm_time += comm_time - latency
        self.communicated_layer = layer
    def fusion_by_block(self, block):
        self.update_maximum_start_time(block.maximum_start_time)
        self.update_minimum_end_time(block.minimum_end_time)
        self.comm_time += block.comm_time - latency
        print(f"self.communicated_layer {self.communicated_layer}")
        print(f"block.communicated_layer {block.communicated_layer}")
        self.communicated_layer = np.add(self.communicated_layer, block.communicated_layer)
        self.start_flag_layer = self.communicated_layer
        self.end_flag_layer = self.communicated_layer

def partition_block(idx, blocks_without_flag, blocks_with_flag, start_flag=False, end_flag=False):
    partitioned_blocks = int(size_per_layers[idx] / partitioned_size)
    for i in range(int(partitioned_blocks-1)):
        layer = [0,0,0,0,0]
        layer[idx] = (i+1)/partitioned_blocks
        comm_time = layer[idx] * size * bandwidth + latency        
        if(start_flag):
            #Block(comp_time, comm_time, 1.7, copy.deepcopy(layer))
            b_with_flag = Block(computation_time_per_layers[i], comm_time, 1.7, copy.deepcopy(layer), partition=True, start_flag=True)
            blocks_with_flag.append(b_with_flag)
        elif(end_flag) :
            b_with_flag = Block(computation_time_per_layers[i], comm_time, 1.7, copy.deepcopy(layer), partition=True, end_flag=True)
            blocks_with_flag.append(b_with_flag)

        b_without_flag = Block(computation_time_per_layers[i], comm_time, 1.7, copy.deepcopy(layer), partition=True)
        blocks_without_flag.append(b_without_flag)

memory_limit = [10, 5, 2, 5, 10]
#SDP SDP DP FSDP FSDP 
computation_time_per_layers = [0.3, 0.5, 0.8, 1.2, 1.7]
shard_per_layers = ['SDP', 'SDP', 'DP', 'FSDP', 'FSDP']
comm_num_per_layer = [2,2,1,3,3]
size_per_layers = [3,4,2,3,2]
latency = 0.01
bandwidth = 0.5

comm_all_gather = [1,1,0,1,1]
all_gather_start_flag = [1,1,0,1,1]

comm_reduce_scatter = [1,1,0,1,1]
reduce_scatter_end_flag = [1,1,0,1,1]

comm_all_gather_fsdp = [0,0,0,1,1]
comm_all_reduce = [0,0,1,0,0]

partitioned_threshold = 3
partitioned_size = 1.1


blocks_all_gather = []
blocks_reduce_scatter = []
blocks_all_reduce = []
blocks_all_gather_fsdp = []

#blocks partitioned
blocks_partitioned_all_gather = []
blocks_partitioned_reduce_scatter = []
blocks_partitioned_all_reduce = []
blocks_partitioned_all_gather_fsdp = []

#blocks partitioned with end flag / start_flag
blocks_partitioned_reduce_scatter_with_end_flag  = []
blocks_partitioned_all_gather_with_start_flag = []


block_all_gather = Block(0, 0, [0,0,0,0,0], [0,0,0,0,0])
block_reduce_scatter = Block(0, 0, [0,0,0,0,0], [0,0,0,0,0])
block_all_reduce = Block(0, 0, [0,0,0,0,0], [0,0,0,0,0])
block_all_gather_fsdp = Block(0, 0, [0,0,0,0,0], [0,0,0,0,0])

#classify by communicationi type
idx = 0
for shard_strategy, size, comp_time, comm_num_per_layer in zip(shard_per_layers, size_per_layers, computation_time_per_layers, comm_num_per_layer) :
    
    layer = [0,0,0,0,0]
    layer[idx] = 1    
    if(shard_strategy == 'SDP' or shard_strategy == "FSDP"):
        comm_time = size * bandwidth + latency
        block_all_gather = Block(comp_time, comm_time, 1.7, copy.deepcopy(layer))
        block_reduce_scatter = Block(comp_time, comm_time, 1.7, copy.deepcopy(layer))
        blocks_all_gather.append(block_all_gather)
        blocks_reduce_scatter.append(block_reduce_scatter)
    elif(shard_strategy == 'DP'):
        comm_time = 2 * size * bandwidth + latency
        block_all_reduce = Block(comp_time, comm_time, 1.7, copy.deepcopy(layer))
        blocks_all_reduce.append(block_all_reduce)
    if(shard_strategy == 'FSDP') :
        comm_time = size * bandwidth + latency
        block_all_gather_fsdp = Block(0, comm_time, computation_time_per_layers[idx-1], copy.deepcopy(layer))
        blocks_all_gather_fsdp.append(block_all_gather_fsdp)
    idx += 1

    
#tensor fusion
idx = 0
cp_blocks_all_gather = copy.deepcopy(blocks_all_gather)
for root_block in cp_blocks_all_gather : 
    inner_idx = 0 
    cp_root_block = None
    for block in cp_blocks_all_gather[idx:]:
        print(f"block.maximum_start_time {block.maximum_start_time}")
        if(size_per_layers[idx] >= partitioned_threshold):
            partition_block(idx, blocks_partitioned_all_gather, blocks_partitioned_all_gather_with_start_flag, start_flag=True)        
        if(inner_idx == 0):
            cp_root_block = copy.deepcopy(root_block)
        else :
            cp_root_block.fusion_by_block(copy.deepcopy(block))
        blocks_all_gather.append(copy.deepcopy(cp_root_block))

        inner_idx += 1
    idx += 1   

idx = 0
cp_blocks_all_reduce = copy.deepcopy(blocks_all_reduce)
for root_block in cp_blocks_all_reduce :
    inner_idx = 0 
    cp_root_block = None
    for block in cp_blocks_all_reduce[idx:]:
        
        if(inner_idx == 0):
            cp_root_block = copy.deepcopy(root_block)
        else :
            cp_root_block.fusion_by_block(copy.deepcopy(block))
        blocks_all_reduce.append(copy.deepcopy(cp_root_block))
        inner_idx += 1
    idx += 1   

idx = 0
cp_blocks_all_gather_fsdp = copy.deepcopy(blocks_all_gather_fsdp)
for root_block in cp_blocks_all_gather_fsdp :
    inner_idx = 0 
    cp_root_block = None
    for block in cp_blocks_all_gather_fsdp[idx:]:
        
        if(inner_idx == 0):
            cp_root_block = copy.deepcopy(root_block)
        else :
            cp_root_block.fusion_by_block(copy.deepcopy(block))
        blocks_all_gather_fsdp.append(copy.deepcopy(cp_root_block))
        inner_idx += 1
    idx += 1   

idx = 0
cp_blocks_reduce_scatter = copy.deepcopy(blocks_reduce_scatter)
for root_block in cp_blocks_reduce_scatter :
    inner_idx = 0 
    cp_root_block = None
    for block in cp_blocks_reduce_scatter[idx:]:
        #tensor parititioning
        if(size_per_layers[idx] >= partitioned_threshold):
            partition_block(idx, blocks_partitioned_reduce_scatter, blocks_partitioned_reduce_scatter_with_end_flag, end_flag=True)
        if(inner_idx == 0):
            cp_root_block = copy.deepcopy(root_block)
        else :
            cp_root_block.fusion_by_block(copy.deepcopy(block))
        blocks_reduce_scatter.append(copy.deepcopy(cp_root_block))
        inner_idx += 1
    idx += 1        




#coefficient matrix 
#for i in range(len(communication_time_per_layers)):
#    layer_flags = [0,0,0,0,0]
#    comm_time = communication_time_per_layers[i]
#    start_time = computation_time_per_layers[i]
#    memory = memory_per_layers[i]
#    layer_flags[i] = 1
#    p = partition(start_time, comm_time, memory, copy.deepcopy(layer_flags))
#    partitions.append(p)
#    for j in range(len(communication_time_per_layers)-i-1):
#        comm_time += communication_time_per_layers[i+j+1] - latency
#        memory += memory_per_layers[i+j+1]
#        layer_flags[j+i+1] = 1 
#        for k in range(len(communication_time_per_layers)-j-i-1):
#            start_time = computation_time_per_layers[i+k+j+1]
#            p = partition(start_time, comm_time, memory, copy.deepcopy(layer_flags))
#            partitions.append(p)
#
len_blocks_all_gather = len(blocks_all_gather)
coeff_mat_all_gather = np.empty((5,len_blocks_all_gather), dtype=object)
i =0
for block in blocks_all_gather :
    j = 0
    for comp_time in computation_time_per_layers :
        print(f"maximum start time {block.maximum_start_time} , comp_time {comp_time}")
        if(block.maximum_start_time <= comp_time):
            block.set_schedule_time(comp_time)
        coeff_mat_all_gather[j][i] = copy.deepcopy(block)
        j += 1
    i += 1

coeff_mat_all_reduce = np.empty((5,len(blocks_all_reduce)), dtype=object)
i =0
for block in blocks_all_reduce :
    j = 0
    for comp_time in computation_time_per_layers :
        if(block.maximum_start_time <= comp_time):
            block.set_schedule_time(comp_time)
        coeff_mat_all_reduce[j][i] = copy.deepcopy(block)
        j += 1
    i += 1

coeff_mat_reduce_scatter = np.empty((5,len(blocks_reduce_scatter)), dtype=object)
i =0
for block in blocks_reduce_scatter :
    j = 0
    for comp_time in computation_time_per_layers :
        if(block.maximum_start_time <= comp_time):
            block.set_schedule_time(comp_time)
        coeff_mat_reduce_scatter[j][i] = copy.deepcopy(block)
        j += 1
    i += 1

coeff_mat_all_gather_fsdp = np.empty((5,len(blocks_all_gather_fsdp)), dtype=object)
i =0
for block in blocks_all_gather_fsdp :
    j = 0
    for comp_time in computation_time_per_layers :
        if(block.maximum_start_time <= comp_time):
            block.set_schedule_time(comp_time)
        #print(block.bandwidth_utilization_per_layer)
        coeff_mat_all_gather_fsdp[j][i] = copy.deepcopy(block)
        j += 1
    i += 1

coeff_mat_partitioned_all_gather = np.empty((5, len(blocks_partitioned_all_gather)), dtype=object)
i = 0
for block in blocks_partitioned_all_gather :
    j = 0
    for comp_time in computation_time_per_layers :
        if(block.maximum_start_time <= comp_time):
            block.set_schedule_time(comp_time)
        #print(block.bandwidth_utilization_per_layer)
        coeff_mat_partitioned_all_gather[j][i] = copy.deepcopy(block)
        j += 1
    i += 1

coeff_mat_partitioned_all_gather_with_start_flag = np.empty((5, len(blocks_partitioned_all_gather_with_start_flag)), dtype=object)
i = 0
for block in blocks_partitioned_all_gather_with_start_flag :
    j = 0
    for comp_time in computation_time_per_layers :
        if(block.maximum_start_time <= comp_time):
            block.set_schedule_time(comp_time)
        #print(block.bandwidth_utilization_per_layer)
        coeff_mat_partitioned_all_gather_with_start_flag[j][i] = copy.deepcopy(block)
        j += 1
    i += 1

coeff_mat_partitioned_reduce_scatter = np.empty((5, len(blocks_partitioned_reduce_scatter)), dtype=object)
i = 0
for block in blocks_partitioned_reduce_scatter :
    j = 0
    for comp_time in computation_time_per_layers :
        if(block.maximum_start_time <= comp_time):
            block.set_schedule_time(comp_time)
        #print(block.bandwidth_utilization_per_layer)
        coeff_mat_partitioned_reduce_scatter[j][i] = copy.deepcopy(block)
        j += 1
    i += 1

coeff_mat_partitioned_reduce_scatter_with_end_flag = np.empty((5, len(blocks_partitioned_reduce_scatter_with_end_flag)), dtype=object)
i = 0
for block in blocks_partitioned_reduce_scatter_with_end_flag :
    j = 0
    for comp_time in computation_time_per_layers :
        if(block.maximum_start_time <= comp_time):
            block.set_schedule_time(comp_time)
        #print(block.bandwidth_utilization_per_layer)
        coeff_mat_partitioned_reduce_scatter_with_end_flag[j][i] = copy.deepcopy(block)
        j += 1
    i += 1


a = []
for i in range(5):
    for j in range(len(blocks_all_gather)):
         a.append(coeff_mat_all_gather[i][j].bandwidth_utilization_per_layer)
x = [] 
for i in range(5):
    for j in range(len(blocks_all_gather)):
        x.append(coeff_mat_all_gather[i][j].communicated_layer)
        print(coeff_mat_all_gather[i][j].communicated_layer)




all_gather_start_time = [] 
for i in range(5):
    for j in range(len(blocks_all_gather)):
        all_gather_start_time.append(coeff_mat_all_gather[i][j].pseudo_start_time)
        print(coeff_mat_all_gather[i][j].start_time)

reduce_scatter_end_time = [] 
for i in range(5):
    for j in range(len(blocks_reduce_scatter)):
        reduce_scatter_end_time.append(coeff_mat_reduce_scatter[i][j].end_time)
        print(coeff_mat_reduce_scatter[i][j].end_time)


b = []
for i in range(5):
    for j in range(len(blocks_reduce_scatter)):
         b.append(coeff_mat_reduce_scatter[i][j].bandwidth_utilization_per_layer)
y = [] 
for i in range(5):
    for j in range(len(blocks_reduce_scatter)):
        y.append(coeff_mat_reduce_scatter[i][j].communicated_layer)
        print(coeff_mat_reduce_scatter[i][j].communicated_layer)

########################## set matrix for coefficient 4 #################################

part_ag_cl = [] 
part_ag_start_time = []
part_ag_pseudo_start_time = []
part_ag_bw = []
part_ag_w_sf_bw = []
for i in range(5):
    for j in range(len(blocks_partitioned_all_gather)):
        part_ag_cl.append(coeff_mat_partitioned_all_gather[i][j].communicated_layer)
        part_ag_bw.append(coeff_mat_partitioned_reduce_scatter[i][j].bandwidth_utilization_per_layer)
        part_ag_start_time.append(coeff_mat_partitioned_all_gather[i][j].start_time)
        part_ag_pseudo_start_time.append(coeff_mat_partitioned_all_gather[i][j].pseudo_start_time)
        #print(coeff_mat_partitioned_all_gather[i][j].communicated_layer)

part_ag_cl_w_sf = []
part_ag_w_sf_start_time = []
part_ag_w_sf_pseudo_start_time = []
part_ag_start_flag = []
for i in range(5):
    for j in range(len(blocks_partitioned_all_gather_with_start_flag)):
        part_ag_cl_w_sf.append(coeff_mat_partitioned_all_gather_with_start_flag[i][j].communicated_layer)
        part_ag_w_sf_bw.append(coeff_mat_partitioned_reduce_scatter[i][j].bandwidth_utilization_per_layer)
        part_ag_w_sf_start_time.append(coeff_mat_partitioned_all_gather_with_start_flag[i][j].start_time)
        part_ag_start_flag.append(coeff_mat_partitioned_all_gather_with_start_flag[i][j].start_flag_layer)
        part_ag_w_sf_pseudo_start_time.append(coeff_mat_partitioned_all_gather_with_start_flag[i][j].pseudo_start_time)
        #print(coeff_mat_partitioned_all_gather[i][j].communicated_layer)

part_rs_cl = []
part_rs_bw = []
part_rs_w_ef_bw = []
part_rs_end_time = []
part_rs_w_ef_cl = []
part_rs_w_ef_end_time = []
part_rs_w_ef_end_flag = []

for i in range(5):
    for j in range(len(blocks_partitioned_reduce_scatter)):
        part_rs_cl.append(coeff_mat_partitioned_reduce_scatter[i][j].communicated_layer)
        part_rs_bw.append(coeff_mat_partitioned_reduce_scatter[i][j].bandwidth_utilization_per_layer)
        part_rs_end_time.append(coeff_mat_partitioned_reduce_scatter[i][j].end_time)

for i in range(5):
    for j in range(len(blocks_partitioned_reduce_scatter_with_end_flag)):
        part_rs_w_ef_cl.append(coeff_mat_partitioned_reduce_scatter_with_end_flag[i][j].communicated_layer)
        part_rs_w_ef_bw.append(coeff_mat_partitioned_reduce_scatter[i][j].bandwidth_utilization_per_layer)
        part_rs_w_ef_end_time.append(coeff_mat_partitioned_reduce_scatter_with_end_flag[i][j].end_time)
        part_rs_w_ef_end_flag.append(coeff_mat_partitioned_reduce_scatter_with_end_flag[i][j].end_flag_layer)


selection_all_reduce = cvxpy.Variable((1, 5*len(blocks_all_reduce)), boolean=True)
selection_all_gather = cvxpy.Variable((1,5*len(blocks_all_gather)), boolean=True)
selection_all_gather_fsdp = cvxpy.Variable((1,5*len(blocks_all_gather_fsdp)), boolean=True)
selection_reduce_scatter = cvxpy.Variable((1,5*len(blocks_reduce_scatter)), boolean=True)

#selection variable for partitioning
selection_partitioned_all_gather               = cvxpy.Variable((1, 5*len(blocks_partitioned_all_gather)), boolean=True)
selection_partitioned_all_gather_start_flag    = cvxpy.Variable((1, 5*len(blocks_partitioned_all_gather_with_start_flag)), boolean=True)    
selection_partitioned_reduce_scatter           = cvxpy.Variable((1, 5*len(blocks_partitioned_reduce_scatter)), boolean=True)
selection_partitioned_reduce_scatter_end_flag  = cvxpy.Variable((1, 5*len(blocks_partitioned_reduce_scatter_with_end_flag)), boolean=True)



#selected_block = a * selection_all_gather.T

#selected_block_RS  = b * selection_reduce_scatter.T

selection_bw_part_ag       = part_ag_bw * selection_partitioned_all_gather.T
selection_bw_part_ag_w_sf  = part_ag_w_sf_bw * selection_partitioned_all_gather_start_flag.T 
selection_bw_part_rs       = part_rs_bw * selection_partitioned_reduce_scatter.T
selection_bw_part_rs_w_ef  = part_rs_w_ef_bw * selection_partitioned_reduce_scatter_end_flag.T
#selected_all_gather_start_time =  selection_all_gather.T @ all_gather_start_time 

#selected_communicated_layer = x @ selection_all_gather.T
#constraint1 =  cvxpy.sum(x * selection_all_gather.T , axis=1) == comm_all_gather 
#constraint2 =  cvxpy.sum(y * selection_reduce_scatter.T , axis=1) == comm_reduce_scatter 
constraint1 =  cvxpy.sum(part_ag_cl * selection_partitioned_all_gather.T +  part_ag_cl_w_sf * selection_partitioned_all_gather_start_flag.T  , axis=1) == [0,1.0,0,0,0]
constraint2 =  cvxpy.sum(part_rs_cl * selection_partitioned_reduce_scatter.T + part_rs_w_ef_cl * selection_partitioned_reduce_scatter_end_flag.T , axis=1) == [0,1.0,0,0,0]
constraint3 =  cvxpy.sum(part_ag_start_flag * selection_partitioned_all_gather_start_flag.T, axis=1) == [0,1,0,0,0]
constraint4 =  cvxpy.sum(part_rs_w_ef_end_flag * selection_partitioned_reduce_scatter_end_flag.T, axis=1) ==[0,1,0,0,0] 
#end_time of reduce scatter of layer l < start_time of all gather of layer l
#constraint3 
#first write naive algorithm 
selected_start_times_AG = []
selected_communications_AG = []

rs_ag_constraints = [constraint1, constraint2, constraint3, constraint4]

layer_checker = [0,0,0,0,0]
layer1_ag = []
layer1_rs = []

layer1_selection_ag = []
layer1_selection_rs = []


max_comm_time = 10
for i in range(5):
    layer_checker[i] = 1
    for j in range(5*len(blocks_all_gather)):
        ag_communicated_layer = coeff_mat_all_gather[int(j/len(blocks_all_gather))][j%len(blocks_all_gather)].communicated_layer
        ag_communicated_layer_np = np.array(ag_communicated_layer)
        layer_checker_np = np.array(layer_checker)
        if(np.sum(ag_communicated_layer_np * layer_checker_np) > 0 ):
            selected_ag_start_time = selection_all_gather[0][j]  * all_gather_start_time[j] 
            layer1_selection_ag.append(selection_all_gather[0][j])
            layer1_ag.append(selected_ag_start_time)
    for j in range(5*len(blocks_reduce_scatter)):
        rs_communicated_layer = coeff_mat_reduce_scatter[int(j/len(blocks_reduce_scatter))][j%len(blocks_reduce_scatter)].communicated_layer 
        rs_communicated_layer_np = np.array(rs_communicated_layer)
        layer_checker_np = np.array(layer_checker)
        if(np.sum(rs_communicated_layer_np * layer_checker_np) > 0 ):
            selected_rs_end_time = selection_reduce_scatter[0][j] * reduce_scatter_end_time[j] 
            layer1_rs.append(selected_rs_end_time)
            layer1_selection_rs.append(selection_reduce_scatter[0][j])
    break      

layer2_selection_ag = []
layer2_selection_rs = []

layer2_ag = []
layer2_rs = []

layer_checker = [0,0,0,0,0]
layer_checker[1] = 1
for j in range(5, 5*len(blocks_all_gather)):
    ag_communicated_layer = coeff_mat_all_gather[int(j/len(blocks_all_gather))][j%len(blocks_all_gather)].communicated_layer
    ag_communicated_layer_np = np.array(ag_communicated_layer)
    layer_checker_np = np.array(layer_checker)
    if(np.sum(ag_communicated_layer_np * layer_checker_np) > 0 ):
        selected_ag_start_time =  selection_all_gather[0][j]  * all_gather_start_time[j] 
        layer2_selection_ag.append(selection_all_gather[0][j])
        layer2_ag.append(selected_ag_start_time)

for j in range(5, 5*len(blocks_reduce_scatter)):
    rs_communicated_layer = coeff_mat_reduce_scatter[int(j/len(blocks_reduce_scatter))][j%len(blocks_reduce_scatter)].communicated_layer 
    rs_communicated_layer_np = np.array(rs_communicated_layer)
    layer_checker_np = np.array(layer_checker)
    if(np.sum(rs_communicated_layer_np * layer_checker_np) > 0 ):
        selected_rs_end_time = selection_reduce_scatter[0][j] * reduce_scatter_end_time[j] 
        layer2_rs.append(selected_rs_end_time)
        layer2_selection_rs.append(selection_reduce_scatter[0][j])

layer1_rs_end_time = cvxpy.vstack(layer1_rs)
layer1_rs_max_end_time = cvxpy.max(layer1_rs_end_time)
layer1_ag_start_time = 10 - cvxpy.vstack(layer1_ag)
#layer1_ag_min_start_time = 10 - cvxpy.max(layer1_ag_start_time)
layer1_selection_rs_vstack = cvxpy.vstack(layer1_selection_rs)
layer1_selection_ag_vstack = cvxpy.vstack(layer1_selection_ag)

layer2_rs_end_time = cvxpy.vstack(layer2_rs)
layer2_rs_max_end_time = cvxpy.max(layer2_rs_end_time)
layer2_ag_start_time = 10 - cvxpy.vstack(layer2_ag)
layer2_selection_rs_vstack = cvxpy.vstack(layer2_selection_rs)
layer2_selection_ag_vstack = cvxpy.vstack(layer2_selection_ag)

#rs_ag_constraints.append(layer1_ag_start_time >= layer1_rs_max_end_time)
#rs_ag_constraints.append(layer2_ag_start_time >= layer2_rs_max_end_time)
#rs_ag_constraints.append(constraint1)
#rs_ag_constraints.append(constraint2)

#constraint 4 : handling start/end flag of partitioned reduce scatter and allgather 
#+with reduce scatter <--> all gather dependency 
# init RS without end flag 

# init RS with end flag
# init AG with start flag
# init AG without start flag

layer2_selection_part_ag = []
layer2_selection_part_ag_w_sf = []
layer2_selection_part_rs = []
layer2_selection_part_rs_w_ef = []

#comparison target
layer2_part_ag_start_time = []
layer2_part_ag_w_sf_start_time = []

layer2_part_ag_pseudo_start_time = []
layer2_part_ag_w_sf_pseudo_start_time = []

layer2_part_rs_end_time = []
layer2_part_rs_w_ef_end_time = []

layer_checker = [0,0,0,0,0]
layer_checker[1] = 1

for j in range(5, 5*len(blocks_partitioned_all_gather)):
    ag_communicated_layer = coeff_mat_all_gather[int(j/len(blocks_partitioned_all_gather))][j%len(blocks_partitioned_all_gather)].communicated_layer
    ag_communicated_layer_np = np.array(ag_communicated_layer)

    layer_checker_np = np.array(layer_checker)
    if(np.sum(ag_communicated_layer_np * layer_checker_np) > 0 ):
        selected_ag_start_time =  selection_partitioned_all_gather[0][j]  * part_ag_start_time[j] 
        selected_ag_w_sf_start_time = selection_partitioned_all_gather_start_flag[0][j]  * part_ag_start_time[j]

        selected_ag_pseudo_start_time =  selection_partitioned_all_gather[0][j] * part_ag_pseudo_start_time[j]
        selected_ag_w_sf_pseudo_start_time = selection_partitioned_all_gather_start_flag[0][j] * part_ag_w_sf_pseudo_start_time[j]

        layer2_selection_part_ag.append(selection_partitioned_all_gather[0][j])
        layer2_part_ag_start_time.append(selected_ag_start_time)
        layer2_part_ag_pseudo_start_time.append(selected_ag_pseudo_start_time)

        layer2_selection_part_ag_w_sf.append(selection_partitioned_all_gather_start_flag[0][j])
        layer2_part_ag_w_sf_start_time.append(selected_ag_w_sf_start_time)
        layer2_part_ag_w_sf_pseudo_start_time.append(selected_ag_w_sf_pseudo_start_time)


for j in range(5, 5*len(blocks_partitioned_reduce_scatter)):
    rs_communicated_layer = coeff_mat_reduce_scatter[int(j/len(blocks_partitioned_reduce_scatter))][j%len(blocks_partitioned_reduce_scatter)].communicated_layer 
    rs_communicated_layer_np = np.array(rs_communicated_layer)

    layer_checker_np = np.array(layer_checker)

    if(np.sum(rs_communicated_layer_np * layer_checker_np) > 0 ):
        selected_rs_end_time = selection_partitioned_reduce_scatter[0][j] * part_rs_end_time[j] 
        selected_rs_w_ef_end_time = selection_partitioned_reduce_scatter_end_flag[0][j] * part_rs_w_ef_end_flag[j] 

        layer2_selection_part_rs.append(selection_partitioned_reduce_scatter[0][j])
        layer2_part_rs_end_time.append(selected_rs_end_time)

        layer2_selection_part_rs_w_ef.append(selection_partitioned_reduce_scatter_end_flag[0][j])
        layer2_part_rs_w_ef_end_time.append(selected_rs_w_ef_end_time)


layer2_part_rs_max_end_time = cvxpy.max(cvxpy.vstack(layer2_part_rs_end_time))
layer2_part_rs_w_ef_end_time = cvxpy.vstack(layer2_part_rs_w_ef_end_time)
layer2_part_rs_w_ef_max_end_time = cvxpy.sum(layer2_part_rs_w_ef_end_time)

layer2_part_ag_w_sf_pseudo_start_time = 10 - cvxpy.vstack(layer2_part_ag_w_sf_pseudo_start_time)
layer2_part_ag_w_sf_min_start_time = cvxpy.min(layer2_part_ag_w_sf_pseudo_start_time)

layer2_part_ag_w_sf_start_time = cvxpy.vstack(layer2_part_ag_w_sf_start_time)
layer2_part_ag_w_sf_max_start_time = cvxpy.max(layer2_part_ag_w_sf_start_time)

layer2_part_ag_start_time = 10 - cvxpy.vstack(layer2_part_ag_w_sf_pseudo_start_time)
layer2_part_ag_min_start_time = cvxpy.min(layer2_part_ag_start_time)

rs_ag_constraints.append(layer2_part_rs_max_end_time <= layer2_part_rs_w_ef_max_end_time)
rs_ag_constraints.append(layer2_part_rs_w_ef_max_end_time <= layer2_part_ag_w_sf_min_start_time)
rs_ag_constraints.append(layer2_part_ag_w_sf_max_start_time <= layer2_part_ag_min_start_time)


B_utilized = bandwidth * selection_bw_part_ag + bandwidth * selection_bw_part_ag_w_sf + bandwidth * selection_bw_part_rs + bandwidth * selection_bw_part_rs_w_ef
problem = cvxpy.Problem(cvxpy.Maximize(cvxpy.sum(B_utilized)), constraints=rs_ag_constraints)    
problem.solve(solver=cvxpy.GUROBI)
print(selection_partitioned_reduce_scatter.value)
print(selection_partitioned_reduce_scatter_end_flag.value)

print(selection_partitioned_all_gather.value)
print(selection_partitioned_all_gather_start_flag.value)
print( cvxpy.sum(part_ag_cl * selection_partitioned_all_gather.T +  part_ag_cl_w_sf * selection_partitioned_all_gather_start_flag.T  , axis=1).value)