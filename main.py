from pathlib import Path

import os
import gym
import d4rl
import numpy as np
import torch
from tqdm import trange

from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
from src.util import return_range, set_seed, sample_batch, torchify, evaluate_policy



def get_env_and_dataset(log, env_name, max_episode_steps):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        min_ret, max_ret = return_range(dataset, max_episode_steps)
        print(f'Dataset returns have range [{min_ret}, {max_ret}]')
        dataset['rewards'] /= (max_ret - min_ret)
        dataset['rewards'] *= max_episode_steps
    elif 'antmaze' in env_name:
        dataset['rewards'] -= 1.

    for k, v in dataset.items():
        dataset[k] = torchify(v)

    return env, dataset


def main(config):


    torch.set_num_threads(1)
    #log = Log(Path(args.log_dir)/args.env_name, vars(args))
    #log(f'Log dir: {log.dir}')

    env, dataset = get_env_and_dataset(log, config.env_name, config.max_episode_steps)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]   # this assume continuous actions
    set_seed(config.seed, env=env)

    if config.deterministic_policy:
        policy = DeterministicPolicy(obs_dim, act_dim, hidden_dim=config.hidden_dim, n_hidden=config.n_hidden)
    else:
        policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=config.hidden_dim, n_hidden=config.n_hidden)
    def eval_policy():
        eval_returns = np.array([evaluate_policy(env, policy, config.max_episode_steps) \
                                 for _ in range(config.n_eval_episodes)])
        normalized_returns = d4rl.get_normalized_score(config.env_name, eval_returns) * 100.0
        log.row({
            'return mean': eval_returns.mean(),
            'return std': eval_returns.std(),
            'normalized return mean': normalized_returns.mean(),
            'normalized return std': normalized_returns.std(),
        })

    iql = ImplicitQLearning(
        qf=TwinQ(obs_dim, act_dim, hidden_dim=config.hidden_dim, n_hidden=config.n_hidden),
        vf=ValueFunction(obs_dim, hidden_dim=config.hidden_dim, n_hidden=config.n_hidden),
        policy=policy,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=config.learning_rate),
        max_steps=config.n_steps,
        tau=config.tau,
        beta=config.beta,
        alpha=config.alpha,
        discount=config.discount
    )

    for step in trange(config.n_steps):
        iql.update(**sample_batch(dataset, config.batch_size))
        if (step+1) % config.eval_period == 0:
            eval_policy()

    torch.save(iql.state_dict(), os.path.join(config.paths['ckpt'], 'final.pt'))
    #log.close()


if __name__ == '__main__':
    from utils.Config import Config, get_parser
    from src.utils import exp_logger
    parser = get_parser(['default'])
    
    parser.add_argument('--env_name', required=True)
    parser.add_argument('--deterministic_policy', action='store_true')
    
    config = Config(parser.parse_args())
    config.load_config("iql", config.cfg_filename, format="yaml")
    #config.add("state_dim",env_list[config.env_name]["state_dim"])
    #config.add("action_dim", env_list[config.env_name]["action_dim"])

    # logger_formats = ["stdout", "log", "csv", "tensorboard"]
    logger_formats = ["stdout", "tensorboard",'csv']
    exp_logger.configure(dir=config.paths["tb"], format_strs=logger_formats, precision=4)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.device)
    os.environ['MUJOCO_GL'] = "egl"

    main(config)