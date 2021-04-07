from spinup.utils.run_utils import ExperimentGrid
from spinup import vpg_pytorch, mcvpg_pytorch
import gym_minigrid
import torch

def run_vpg_lava():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=3)
    args = parser.parse_args()

    eg = ExperimentGrid(name='vpg-pt-bench')
    eg.add('env_name', 'MiniGrid-LavaCrossingS9N2-v1', '', True)
    eg.add('seed', [6*i for i in range(args.num_runs)])
    eg.add('epochs', 500)
    eg.add('steps_per_epoch', 5000)
    eg.add('max_ep_len', 200)
    # eg.add('ac_kwargs:hidden_sizes', [(32,), (64,64)], 'hid')
    eg.add('ac_kwargs:hidden_sizes', [(32,)], 'hid')
    # eg.add('ac_kwargs:activation', [torch.nn.Tanh, torch.nn.ReLU], '')
    eg.add('ac_kwargs:activation', [torch.nn.Tanh], '')
    eg.run(vpg_pytorch, num_cpu=args.cpu, datestamp=True)

def run_mcvpg_lava():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=3)
    args = parser.parse_args()

    eg = ExperimentGrid(name='mcvpg-pt-bench')
    eg.add('env_name', 'MiniGrid-LavaCrossingS9N2-v1', '', True)
    eg.add('seed', [6*i for i in range(args.num_runs)])
    eg.add('epochs', 500)
    eg.add('steps_per_epoch', 4000)
    eg.add('max_ep_len', 200)
    # eg.add('ac_kwargs:hidden_sizes', [(32,), (64,64)], 'hid')
    eg.add('ac_kwargs:hidden_sizes', [(32,)], 'hid')
    # eg.add('ac_kwargs:activation', [torch.nn.Tanh, torch.nn.ReLU], '')
    eg.add('ac_kwargs:activation', [torch.nn.Tanh], '')
    eg.run(mcvpg_pytorch, num_cpu=args.cpu, datestamp=True)

if __name__ == '__main__':
    run_mcvpg_lava()
    # run_vpg_lava()
    