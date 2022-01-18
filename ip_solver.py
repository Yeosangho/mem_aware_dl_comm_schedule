import cvxpy
import copy
import numpy as np

class partition():
    def __init__(self, start_time, comm_time, memory, layer):
        self.start_time = start_time
        self.comm_time = comm_time
        self.end_time = start_time + comm_time
        self.memory = memory
        self.communicated_layer = layer
memory_limit = [10, 5, 2, 5, 10]

computation_time_per_layers = [0.3, 0.5, 0.7, 0.9, 1.32]
communication_time_per_layers = [0.27, 0.32, 0.43, 0.55, 0.231]
memory_per_layers = [3,4,2,3,2]
latency = 0.0001

partitions = []
for i in range(len(communication_time_per_layers)):
    layer_flags = [0,0,0,0,0]
    comm_time = communication_time_per_layers[i]
    start_time = computation_time_per_layers[i]
    memory = memory_per_layers[i]
    layer_flags[i] = 1
    p = partition(start_time, comm_time, memory, copy.deepcopy(layer_flags))
    partitions.append(p)
    for j in range(len(communication_time_per_layers)-i-1):
        comm_time += communication_time_per_layers[i+j+1] - latency
        memory += memory_per_layers[i+j+1]
        layer_flags[j+i+1] = 1 
        for k in range(len(communication_time_per_layers)-j-i-1):
            start_time = computation_time_per_layers[i+k+j+1]
            p = partition(start_time, comm_time, memory, copy.deepcopy(layer_flags))
            partitions.append(p)
start_times = []
comm_times = []
end_times = []
layers = []
memories = []
partitions.sort(key=lambda x: x.memory, reverse=False)
partitions.sort(key=lambda x: x.start_time, reverse=False)
for p in partitions:
    print(p.start_time)
    print(p.comm_time)
    print(p.end_time)
    print(p.memory)
    print(p.communicated_layer)
    print("#################")
    start_times.append(p.start_time)
    comm_times.append(p.comm_time)
    end_times.append(p.end_time)
    memories.append(p.memory)
    layers.append(p.communicated_layer)
selection = cvxpy.Variable((1,len(partitions)), boolean=True)
print(selection.shape)
np_layers = np.array(layers)
print(np_layers)
np_start_times = np.array(start_times)
np_comm_times = np.array(comm_times)
np_end_times = np.array(end_times)
np_memories = np.array(memories)
np_condition = np.array([1,1,1,1,1])
constraint1 =  cvxpy.sum(selection @ np_layers , axis=0) == np_condition
constraint_scalar = list()
print(selection.value)
#for i in range(5):
#    copied_layers = copy.deepcopy(np_layers[:,i])
#    const = cvxpy.sum(selection * copied_layers, axis=0) == [1]
#    constraint_scalar.append(copy.deepcopy(const))

#for i in range(25):
#    copied_layers = copy.deepcopy(np_layers[:,i])
#    const = cvxpy.sum(selection[i] * copied_layers, axis=0) == [1]
#    constraint_scalar.append(copy.deepcopy(const))

#selected_comms_start_times = selection * np_start_times
#selected_comms_comm_time = selection * np_comm_times
#selected_comms_end_times = selection * np_end_times
#selected_comms_memories = selection * np_memories
scheduled_communications = cvxpy.Variable(len(partitions))
recent_comms_endtime = 0
delay = 0
#a = cvxpy.sum(cvxpy.multiply(selection, np_comm_times))
a = selection @ np_comm_times
b = selection @ np_start_times
c = selection @ np_end_times 
tiled_selection = np.tile(selection, (len(partitions), 1) )
tiled_start_times = np.tile(np_start_times, (len(partitions), 1))
tiled_comm_times = np.tile(np_comm_times, (len(partitions), 1)).transpose()
print(tiled_comm_times)
print(tiled_start_times)
#tiled_start_times = np.transpose(tiled_start_times)

tiled_end_times  = np.tile(np_end_times, (len(partitions), 1))
tiled_end_times = np.transpose(tiled_end_times)
delay_time = np.minimum(tiled_end_times-tiled_start_times, tiled_comm_times)
print(tiled_end_times)
np.fill_diagonal(delay_time, 0 )
delay_time = np.triu(delay_time)
delay_time = delay_time + delay_time.T
print(delay_time)
un_delay_time = copy.deepcopy(delay_time)

delay_time[delay_time < 0] = 0  #delay_time ==> find impact of each communications
print(delay_time)
un_delay_time[delay_time > 0] = 0
#print(un_delay_time)
#print(delay_time.transpose()) #delay_time.transpose() ==> find how much each communication is delayed by others
#delay_time > 0 means this communications are overlapped each other. 
#unconnected 

start_time = selection * np_start_times
selection_transpose = cvxpy.reshape(selection, (len(partitions), 1))
#delayed_time = selection * delay_time * selection_transpose
delayed_time =  cvxpy.quad_form( selection.T,delay_time)

#delayed_time = delayed_time @ selection_transpose
#x = cvxpy.sum(delayed_time) + cvxpy.max(selection * np_end_times)
delay_sum = cvxpy.sum(delayed_time)/2
end_time = cvxpy.max(selection * np_end_times) 
#comm_sum = cvxpy.sum(selection * np_comm_times) 
#x = cvxpy.max(selection * np_end_times)  + cvxpy.sum(delayed_time)
#delay = ((end_time - start_time[0]) - (comm_sum- delay_sum))
obj = end_time + delay_sum

#for i in range(selection.size):
#    
#    if(selection[0][i]) :
#        print(selection[0][i])
#        if i != 0 :
#            delay = recent_comms_endtime - np_start_times[i] 
#        if(delay > 0):
#            recent_comms_endtime += np_comm_times[i] 
#        else :
#            recent_comms_endtime += selection[0][i] * (np_comm_times[i] - delay)
#    #scheduled_communications[i] = recent_comms_endtime
#print(recent_comms_endtime)
#obj = scheduled_communications[-1]   
#obj = recent_comms_endtime 
constraints = [constraint1]
problem = cvxpy.Problem(cvxpy.Minimize(2), constraints=constraints)    
problem.solve(solver=cvxpy.GUROBI)

print(np.sum(selection.value * delay_time * selection.T.value))
y = selection.value * delay_time
z = selection.value * un_delay_time
y = selection.value.transpose() * y 
print(delayed_time.value)
print(selection.value)
#print( ( delay_time * selection_transpose.value)) 
#print(np.max(selection.value * np_end_times))
#print(selection.value)
print(end_time.value)
##print(problem.status)
print(problem.value)
##weights = np.array(selection.value)
#print(selection.value)

#x = np.array([True,True,False])
#a = np.array([[1,0,0],
#              [0,1,0],
#              [0,0,1]])
#print(np.sum(x*a[:,2], axis=0) == 1)
#
#x = cvxpy.Variable()
#y = cvxpy.Variable()
#
## Create two constraints.
#constraints = [x + y == 1,
#               x - y >= 1]
#
## Form objective.
#obj = cvxpy.Minimize((x - y)**2)
#
## Form and solve problem.
#prob = cvxpy.Problem(obj, constraints)
#prob.solve()
#
## The optimal dual variable (Lagrange multiplier) for
## a constraint is stored in constraint.dual_value.
#print("optimal (x + y == 1) dual variable", constraints[0].dual_value)
#print("optimal (x - y >= 1) dual variable", constraints[1].dual_value)
#print("x - y value:", (x - y).value)

# The data for the Knapsack problem
# P is total weight capacity of sack
# weights and utilities are also specified
#P = 165
#weights = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82])
#utilities = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72])
#
## The variable we are solving for
#selection = cvxpy.Variable(len(weights), boolean=True)
#
## The sum of the weights should be less than or equal to P
#weight_constraint = weights * selection <= P
#
## Our total utility is the sum of the item utilities
#total_utility = utilities * selection
#
## We tell cvxpy that we want to maximize total utility 
## subject to weight_constraint. All constraints in 
## cvxpy must be passed as a list
#knapsack_problem = cvxpy.Problem(cvxpy.Maximize(total_utility), [weight_constraint])
#
## Solving the problem
#knapsack_problem.solve(solver=cvxpy.GUROBI)
#print(selection.value)