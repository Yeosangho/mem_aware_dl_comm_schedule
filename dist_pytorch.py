import os
import torch
import torch.distributed as dist
import multiprocessing as mp
def run(rank, size):
    print(4)
    tensor = torch.zeros(1).cuda()
    tensor += 1
    print(5)
    #dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        print('send')
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        print('recv')
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    print(1)
    os.environ['MASTER_ADDR'] = '210.107.197.218'
    os.environ['MASTER_PORT'] = '29500'
    print(2)
    dist.init_process_group(backend, rank=rank, world_size=size)
    print(3)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    init_process(0, 2, run, 'nccl')
    #size = 2
    #processes = []
    #mp.set_start_method("spawn")
    #for rank in range(size):
    #    p = mp.Process(target=init_process, args=(rank, size, run))
    #    p.start()
    #    processes.append(p)
#
    #for p in processes:
    #    p.join()   
 