import argparse 
import os 

def parse_args():
    parser = argparse.ArgumentParser() 
    
    # experiment values 
    parser.add_argument("--exp-name", type=str, default="test1",
        help="name of the experiment")
    parser.add_argument("--seed", type=int, default=63, 
        help="seed of the experiment")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
        help = "gpu or cpu, default 'cuda'")
    parser.add_argument("--n-games", type=int, default=8000,
        help="total games of the experiments")
    parser.add_argument("--n-ppo-games", type=int, default=5000,
        help="total games of the experiments")
    parser.add_argument("--n-fcp-games", type=int, default=10000,
        help="total games of the experiments")
    parser.add_argument("--worker-id", type=int, default=53,
        help = "worker_id for environment")
    parser.add_argument("--cp-every", type=int, default=500,
        help = "checkpoint every n episodes")
    
    # hyperparameters to tune 
    parser.add_argument("--learning-rate", type=float, default=1e-4, 
        help="learning rate of optimizer")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="discounted factor for future rewards")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="lambda for the general advantage estimation")
    parser.add_argument("--eps-clip", type=float, default=0.2,
        help="surrogate clipping coefficient")
    parser.add_argument("--batch-size", type=int, default=64,
        help="batch size of memory")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--fc1-dims", type=int, default=100,
        help="dimension of first layer")
    parser.add_argument("--fc2-dims", type=int, default=100,
        help="dimension of second layer")
    
    args = parser.parse_args() 
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    return args 