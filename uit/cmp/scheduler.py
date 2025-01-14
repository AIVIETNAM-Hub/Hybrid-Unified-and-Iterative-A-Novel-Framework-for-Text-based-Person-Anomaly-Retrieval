from torch.optim.lr_scheduler import LambdaLR
import math


def create_scheduler(args, optimizer):
    if 'num_training_steps' not in args:
        args['num_training_steps'] = args['epochs'] * args['step_per_epoch']
    print("### num_training_steps, ", args['num_training_steps'], flush=True)

    if isinstance(args['num_warmup_steps'], float):
        assert 0 <= args['num_warmup_steps'] < 1
        args['num_warmup_steps'] = int(args['num_training_steps'] * args['num_warmup_steps'])
    print("### num_warmup_steps, ", args['num_warmup_steps'], flush=True)

    print('sched:', args.sched, flush=True)

    if args.sched == 'linear':
        def lr_lambda(current_step: int):
            if current_step < args.num_warmup_steps:
                return float(current_step) / float(max(1, args.num_warmup_steps))
            return max(
                0.0, float(args.num_training_steps - current_step) / float(
                    max(1, args.num_training_steps - args.num_warmup_steps))
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    elif args.sched == 'step':
        def lr_lambda(current_step: int):
            if current_step < args.num_warmup_steps:
                return float(current_step) / float(max(1, args.num_warmup_steps))
            elif current_step < args.num_warmup_steps * 4:
                tt = 1
            elif current_step < args.num_warmup_steps * 7:
                tt = 0.5
            else:
                tt = 0.2

            return tt * max(
                0.0, float(args.num_training_steps - current_step) / float(
                    max(1, args.num_training_steps - args.num_warmup_steps))
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    elif args.sched == 'cosine':
        print('Load cosine')
        def lr_lambda(current_step, warmup_steps, total_steps, min_lr=1e-5, max_lr=1e-4):
            if current_step < warmup_steps:
                return current_step / warmup_steps
            else:
                progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return (cosine_decay * (max_lr - min_lr) + min_lr) / max_lr
        warmup_steps = args.num_training_steps * 0.02
        lr_scheduler = LambdaLR(
            optimizer, lr_lambda=lambda step: lr_lambda(step, warmup_steps, args.num_training_steps)
        )
            
    else:
        raise NotImplementedError(f"args.sched == {args.sched}")

    return lr_scheduler


# stepLR
# 
