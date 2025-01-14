import os
import argparse


# Set it correctly for distributed training across nodes
NNODES = 1
NODE_RANK = 0
MASTER_ADDR = '127.0.0.1'
MASTER_PORT = 1666  # 0~65536
NPROC_PER_NODE = 1  # e.g. 4 gpus

print("NNODES, ", NNODES)
print("NODE_RANK, ", NODE_RANK)
print("MASTER_ADDR, ", MASTER_ADDR)
print("MASTER_PORT, ", MASTER_PORT)
print("NPROC_PER_NODE, ", NPROC_PER_NODE)


def get_dist_launch(args):
    # if args.dist == 'f4':
    #     return "CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=4 python3 -m torch.distributed.run --nproc_per_node=4 " \
    #            "--nnodes={:} --node_rank={:} " \
    #            "--master_addr={:} --master_port={:} ".format(NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT)
    # elif args.dist == 'f3':
    #     return "CUDA_VISIBLE_DEVICES=1,2,3 WORLD_SIZE=3 python3 -m torch.distributed.run --nproc_per_node=3 " \
    #            "--nnodes={:} --node_rank={:} " \
    #            "--master_addr={:} --master_port={:} ".format(NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT)
    # elif args.dist.startswith('gpu'):  # use one gpu, --dist "gpu0"
    #     num = int(args.dist[3:])
    #     return "CUDA_VISIBLE_DEVICES={:} WORLD_SIZE=1 python3 -m torch.distributed.run --nproc_per_node=1 " \
    #            "--nnodes={:} --node_rank={:} " \
    #            "--master_addr={:} --master_port={:} ".format(num, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT)
    # else:
    #     raise ValueError
    num = 0
    return "CUDA_VISIBLE_DEVICES={:} WORLD_SIZE=1 python3 -m torch.distributed.run --nproc_per_node=1 " \
               "--nnodes={:} --node_rank={:} " \
               "--master_addr={:} --master_port={:} ".format(num, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT)


def run(args):
    if args.task in ['baseline', 'cmp' ]:
        args.config = 'configs/' + args.task + '.yaml'
        print(args.task, args.config)

        dist_launch = get_dist_launch(args)
        os.system(
            f"{dist_launch} Search.py --config {args.config} --task {args.task} --output_dir {args.output_dir} "
            f"--checkpoint {args.checkpoint} --bs {args.bs} --epo {args.epo} --lr {args.lr} --seed {args.seed} "
            f"{'--evaluate' if args.evaluate else ''}")

    else:
        raise NotImplementedError(f"task == {args.task}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='baseline', type=str)
    parser.add_argument('--dist', default='f4', type=str, help="see func get_dist_launch for details")
    parser.add_argument('--output_dir', default='out/cmp', type=str, help='local path')
    parser.add_argument('--checkpoint', default='./checkpoint/pretrained.pth', type=str)
    parser.add_argument('--bs', default=0, type=int, help="mini batch size")
    parser.add_argument('--epo', default=0, type=int, help="epoch")
    parser.add_argument('--lr', default=0.0, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--evaluate', action='store_true', default=False, help="directly evaluation")
    args = parser.parse_args()

    run(args)
