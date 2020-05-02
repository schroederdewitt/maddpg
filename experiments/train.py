from __future__ import division, print_function, unicode_literals
from sacred import Experiment
import numpy as np
import os
import collections
from os.path import dirname, abspath
import pymongo
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
from utils.logging import get_logger, Logger
from utils.dict2namedtuple import convert
import yaml

import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.common.noise import OUNoise
from maddpg.trainer.maddpg import MADDPGAgentTrainer

from tensorflow.contrib import layers


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--env-name", type=str, default="particle", help="name of the environment")
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--test-rate", type=int, default=2000, help="test rate")
    parser.add_argument("--n-tests", type=int, default=10, help="n tests per test")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    parser.add_argument("--score-function", type=str, default="sum", help="score function")
    parser.add_argument("--partial-obs", action="store_true", default=False, help="whether the agent has partial obs")
    # Multiagent Mujoco
    parser.add_argument("--mujoco-name", type=str, default="HalfCheetah-v2", help="name of the mujoco env")
    parser.add_argument("--agent-conf", type=str, default="2x3", help="agent configuration for mujoco multi")
    parser.add_argument("--agent-obsk", type=int, default=1, help="the agent can see the k neareast neighbors")
    parser.add_argument("--k-categories", type=str, default="qpos,qvel|qpos", help="a string describing which properties are observable at which connection distance as comma-separated lists separated by vertical bars")
    parser.add_argument("--env-version", type=int, default=2, help="environment version")
    parser.add_argument("--obs-add-global-pos", action="store_true", help="agent configuration for mujoco multi")
    parser.add_argument("--agent-view-radius", type=float, default=-1, help="view radius of agents")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--discrete-action", action="store_true", default=False, help="use continuous action space by default")
    parser.add_argument("--buffer-warmup", type=int, default=1000, help="number of transitions the replay buffer should at least have")
    parser.add_argument("--learn-interval", type=int, default=1, help="train the network after every fixed number of time steps")
    parser.add_argument("--target-update-tau", type=float, default=0.001, help="soft update param")
    parser.add_argument("--optimizer-epsilon", type=float, default=0.01, help="epsilon value for the optimizer")
    parser.add_argument("--explore-noise", type=str, default="gaussian", help="add gaussian noise to the action selection")
    parser.add_argument("--start-steps", type=int, default=0, help="randomly sample actions from a uniform distribution for better exploration before this number of timesteps")
    parser.add_argument("--ou-stop-episode", type=int, default=100, help="number of episodes to do exploration if selecting ou noise")
    parser.add_argument("--use-global-state", action="store_true", default=False, help="the centralised critic concatenates observations of all agents by default, if set True, it uses global state instead")
    parser.add_argument("--share-weights", action="store_true", default=False)
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")

    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, constrain_out=False, discrete_action=False,
              rnn_cell=None):

    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope+"fc1", reuse=tf.AUTO_REUSE):

        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu, scope=scope)

    with tf.variable_scope(scope + "fc2", reuse=tf.AUTO_REUSE):
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu, scope=scope)
        # out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)

        # NOTE: use this for continuous action space
        if constrain_out and not discrete_action:
            with tf.variable_scope(scope + "fc3", reuse=tf.AUTO_REUSE):
                out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=tf.tanh, scope=scope)
        else:
            with tf.variable_scope(scope + "fc3", reuse=tf.AUTO_REUSE):
                out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None, scope=scope)
        return out

def make_env(env_name, scenario_name, arglist, benchmark=False):

    if env_name == "particle":
        from multiagent.environment import MultiAgentEnv
        import multiagent.scenarios as scenarios
        # load scenario from script
        scenario = scenarios.load(scenario_name + ".py").Scenario()
        # create world
        if not arglist.partial_obs:
            world = scenario.make_world()
        else:
            world = scenario.make_world(args=arglist)
        # create multiagent environment
        if benchmark:
            env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
        else:
            if not arglist.partial_obs:
                env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
            else:
                env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.full_observation)

    elif env_name == "multiagent_mujoco":
        from envs.multiagent_mujoco.mujoco_multi import MujocoMulti

        kwargs = {"scenario": arglist.scenario,
                  "agent_obsk": arglist.agent_obsk,
                  "k_categories": arglist.k_categories,
                  "env_version": arglist.env_version,
                  "agent_conf": arglist.agent_conf,
                  "obs_add_global_pos": arglist.obs_add_global_pos,
                  "episode_limit": arglist.max_episode_len}
        env = MujocoMulti(env_args=kwargs)

    print("ENV TOTAL ACTION SPACE: {}", env.action_space)
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    print("HALLULAH", env.action_space)
    if arglist.use_global_state:
        state_shape = env.get_state().shape
    else:
        state_shape = obs_shape_n[0]
    n_agents = len(range(num_adversaries, env.n))
    for i in range(num_adversaries):
        trainers.append(trainer(
            n_agents, "agent_%d" % i, model, state_shape, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            n_agents, "agent_%d" % i, model, state_shape, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def train(arglist, logger, _config):
    with U.single_threaded_session(frac=0.2):
        # Create environment
        env = make_env(arglist.env_name, arglist.scenario, arglist, arglist.benchmark)

        # Setting the random seed throughout the modules
        np.random.seed(_config["seed"])
        tf.set_random_seed(_config["seed"])
        env.seed(_config["seed"])
        print("seed: ", _config["seed"])

        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        if arglist.use_global_state:
            state = env.get_state()
        else:
            state = obs_n
        episode_step = 0
        train_step = 0
        log_train_stats_t = -100000
        t_start = time.time()

        # add OUNoise objects for each agent
        exploration_noise = [OUNoise(env.action_space[i].shape[0]) for i in range(env.n)]

        print('Starting iterations...')
        while True:
            # get action
            if arglist.explore_noise == "ou":
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]

                # add OUNoise to explore
                if train_step < arglist.max_episode_len * arglist.ou_stop_episode:
                    for _aid in range(env.n):
                        exploration_noise[_aid].reset()
                        ou_noise = exploration_noise[_aid].noise()
                        action_n[_aid] += ou_noise
            elif arglist.explore_noise == "gaussian":
                if train_step >= arglist.start_steps:
                    action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                    action_n += 0.1 * np.random.randn(env.action_space[0].shape[0])
                else:
                    action_n = [env.action_space[i].sample() for i in range(env.n)]

            # now clamp actions to permissible action range (necessary after exploration)
            act_limit = env.action_space[0].high[0]   # assuming all dimensions share the same bound
            for _aid in range(env.n):
                np.clip(action_n[_aid], -act_limit, act_limit, out=action_n[_aid])

            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            if arglist.use_global_state:
                next_state = env.get_state()
            else:
                next_state = new_obs_n

            if done and not terminal:
                done_n = [True for _ in range(len(done_n))]
            else:
                done_n = [False for _ in range(len(done_n))]

            # collect experience
            for i, agent in enumerate(trainers):
            #     print("index i", i)
            #     print("obs_n", obs_n)
            #     print("action_n", action_n)
            #     print("rew_n", rew_n)
            #     print("new_obs_n", new_obs_n)
            #     print("done_n", done_n)
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal,
                                 state=state, next_state=next_state)
            obs_n = new_obs_n
            state = next_state

            for i, rew in enumerate(rew_n):
                # NOTE: We do not sum over all agents' individual rewards again for cooperative env.
                # episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew
            episode_rewards[-1] += rew_n[0]

            if done or terminal:
                obs_n = env.reset()
                if arglist.use_global_state:
                    state = env.get_state()
                else:
                    state = obs_n
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # generate test trajectories
            # if terminal and (train_step % arglist.test_rate == 0):
            if (done or terminal) and (train_step - log_train_stats_t >= arglist.test_rate):
                episode_rewards_test = []
                agent_rewards_test = [[] for _ in trainers]
                for _ in range(arglist.n_tests):
                    episode_rewards_test.append(0)
                    for i, agent in enumerate(trainers):
                        agent_rewards_test[i].append(0)
                    episode_test_step = 0
                    while True:
                        action_n = [agent.action_test(obs) for agent, obs in zip(trainers,obs_n)]

                        # now clamp actions to permissible action range (necessary after exploration)
                        act_limit = env.action_space[0].high[0]  # assuming all dimensions share the same bound
                        for _aid in range(env.n):
                            np.clip(action_n[_aid], -act_limit, act_limit, out=action_n[_aid])

                        # environment step
                        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                        episode_test_step += 1
                        done = all(done_n)
                        terminal = (episode_test_step >= arglist.max_episode_len)
                        obs_n = new_obs_n

                        for i, rew in enumerate(rew_n):
                            # NOTE: we do not sum over all agents' individual rewards again for cooperative env.
                            # episode_rewards_test[-1] += rew
                            agent_rewards_test[i][-1] += rew
                        episode_rewards_test[-1] += rew_n[0]
                        if done or terminal:
                            obs_n = env.reset()
                            break
                # save them to sacred
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))

                prefix = "test"
                logger.log_stat(prefix + "_return_mean", np.mean(episode_rewards_test), train_step)
                for _i, rew in enumerate(agent_rewards_test):
                    final_ep_ag_rewards.append(np.mean(rew))
                    logger.log_stat(prefix + "_return_mean_agent{}".format(_i), np.mean(rew), train_step)

                log_train_stats_t = train_step

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                # NOTE: Make sure we use the same values for these parameters as used in pymarl.
                if len(agent.replay_buffer) > arglist.buffer_warmup and len(agent.replay_buffer) >= arglist.batch_size:
                    if train_step % arglist.learn_interval == 0:
                        loss = agent.update(trainers, train_step)

            # save model, display training output
            prefix = ""  # not sure if test or train wtf
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(os.path.join(arglist.save_dir, arglist.exp_name, "state", str(len(episode_rewards))), saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                    logger.log_stat(prefix + "episodes", len(episode_rewards), train_step)
                    logger.log_stat(prefix + "mean_episode_reward", np.mean(episode_rewards[-arglist.save_rate:]), train_step)
                    logger.log_stat(prefix + "time", round(time.time()-t_start, 3), train_step)
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))

                logger.log_stat(prefix + "return_mean", np.mean(episode_rewards[-arglist.save_rate:]), train_step)
                for _i, rew in enumerate(agent_rewards):
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))
                    logger.log_stat(prefix + "return_mean_agent{}".format(_i), np.mean(rew[-arglist.save_rate:]), train_step)

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break


SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")

mongo_client = None


# Function to connect to a mongodb and add a Sacred MongoObserver
def setup_mongodb(db_url, db_name):
    client = None
    mongodb_fail = True

    # Try 5 times to connect to the mongodb
    for tries in range(5):
        # First try to connect to the central server. If that doesn't work then just save locally
        maxSevSelDelay = 10000  # Assume 10s maximum server selection delay
        try:
            # Check whether server is accessible
            logger.info("Trying to connect to mongoDB '{}'".format(db_url))
            client = pymongo.MongoClient(db_url, ssl=True, serverSelectionTimeoutMS=maxSevSelDelay)
            client.server_info()
            # If this hasn't raised an exception, we can add the observer
            ex.observers.append(MongoObserver.create(url=db_url, db_name=db_name, ssl=True)) # db_name=db_name,
            logger.info("Added MongoDB observer on {}.".format(db_url))
            mongodb_fail = False
            break
        except pymongo.errors.ServerSelectionTimeoutError:
            logger.warning("Couldn't connect to MongoDB on try {}".format(tries + 1))

    if mongodb_fail:
        logger.error("Couldn't connect to MongoDB after 5 tries!")
        # TODO: Maybe we want to end the script here sometimes?

    return client


@ex.main
def my_main(_run, _config, _log):
    global mongo_client

    import datetime
    unique_token = "{}__{}".format(_config["name"], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # run the framework
    # run(_run, _config, _log, mongo_client, unique_token)
    arglist = parse_args()

    logger = Logger(_log)
    # configure tensorboard logger
    unique_token = "{}__{}".format(arglist.exp_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    use_tensorboard = False
    if use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)
    logger.setup_sacred(_run)

    train(arglist, logger, _config)
    # arglist = convert(_config)
    #train(arglist)

    # force exit
    os._exit(0)

def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

if __name__ == '__main__':
    import os

    arglist = parse_args()

    from copy import deepcopy
    # params = deepcopy(sys.argv)

    # scenario_name = None
    # for _i, _v in enumerate(params):
    #     if _v.split("=")[0] == "--scenario":
    #         #scenario_name = _v.split("=")[1]
    #         scenario_name = params[_i + 1]
    #         del params[_i:_i+2]
    #         break
    #
    # name = None
    # for _i, _v in enumerate(params):
    #     if _v.split("=")[0] == "--name":
    #         #scenario_name = _v.split("=")[1]
    #         name = params[_i + 1]
    #         del params[_i:_i+2]
    #         break

    # now add all the config to sacred
    # ex.add_config({"scenario":scenario_name,
    #                "name":name})
    ex.add_config({"name":arglist.exp_name})

    # Check if we don't want to save to sacred mongodb
    no_mongodb = True

    # for _i, _v in enumerate(params):
    #     if "no-mongo" in _v:
    #     # if "--no-mongo" == _v:
    #         del params[_i]
    #         no_mongodb = True
    #         break

    config_dict={}
    config_dict["db_url"] = "mongodb://pymarlOwner:EMC7Jp98c8rE7FxxN7g82DT5spGsVr9A@gandalf.cs.ox.ac.uk:27017/pymarl"
    config_dict["db_name"] = "pymarl"

    # If there is no url set for the mongodb, we cannot use it
    if not no_mongodb and "db_url" not in config_dict:
        no_mongodb = True
        logger.error("No 'db_url' to use for Sacred MongoDB")

    if not no_mongodb:
        db_url = config_dict["db_url"]
        db_name = config_dict["db_name"]
        mongo_client = setup_mongodb(db_url, db_name)

    # Save to disk by default for sacred, even if we are using the mongodb
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline("")

