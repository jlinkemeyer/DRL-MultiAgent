import argparse
import progressbar as pb
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
from collections import deque

from mlagents_envs.environment import UnityEnvironment, ActionTuple

from agent import DDPGAgent
from buffer import ReplayBuffer

# TODO: adjust for multi-agent
def setup_environment(file_name, log_dir, verbose=True):
    """
    Creates UnityEnvironment object from Unity environment binary
    and extracts 
    """
    # create environment
    env = UnityEnvironment(
        file_name,
        seed=1,
        side_channels=[],
        log_folder=log_dir
    )
    env.reset()

    # get behavior and agent spec
    behavior_name = list(env.behavior_specs)[0] # behavior_specs = brains?
    agent_spec = env.behavior_specs[behavior_name]

    if verbose:
        # TODO: observation and action shapes
        print("\n")
        print("SPEC INFO:")
        print(f"-- Number of observations: {len(agent_spec.observation_specs)}")
        if agent_spec.action_spec.is_continuous():
            print("-- Action space: continuous\n")
        elif agent_spec.action_spec.is_discrete():
            print(f"-- Action space: discrete (size={agent_spec.action_spec.discrete_size})\n")

    return env, behavior_name, agent_spec 


def train_single_agent(env_path, log_dir, config):
    env, behavior_name, agent_spec = setup_environment(env_path, log_dir, verbose=True)

    # read/set hyperparameters
    # set random seed
    # set save interval
    # set noise

    # setup performance metrics
    agent_scores = []
    agent_scores_last_100 = deque(maxlen = 100)
    agent_scores_avg, previous_agent_scores_avg = np.zeros(1), np.zeros(1)

    # create model directory

    env.reset()
    buffer = ReplayBuffer(size=config['buffer_size'], n_steps=config['n_steps'], 
        discount_rate=config['discount_rate'])
    agent = DDPGAgent(action_size=config['action_size'])

    logger = SummaryWriter(logdir=log_dir)
    # widget = ['episode: ', pb.Counter(), '/', str(config['number_of_episodes']), ' ',
    #     pb.DynamicMessage('avg_score'), ' ',
    #     pb.DynamicMessage('buffer_size'), ' ',
    #     pb.ETA(), ' ',
    #     pb.Bar(marker=pb.RotatingMarker()), ' ']
    # timer = pb.ProgressBar(widgets=widget, maxval=config['number_of_episodes']).start()

    # training loop
    for episode in range(config['number_of_episodes']):
        # timer.update(episode, avg_score=agent_scores_avg, buffer_size=len(buffer))
        # reset n-step bootstraps after every episode
        buffer.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        tracked_agent = -1
        reward_current_episode = np.zeros(1)
        done = False

        observation = decision_steps[0].obs
        observation = np.concatenate((observation[0], observation[1]))

        # for t in range(config['episode_length']):
        while not done:
            if tracked_agent == -1 and len(decision_steps) >= 1:
                tracked_agent = decision_steps.agent_id[0]

            # take action
            action = agent.act(tf.expand_dims(observation, 0))
            action_tuple = ActionTuple()
            action_tuple.add_discrete(action)
            env.set_actions(behavior_name, action_tuple)
            env.step()

            # get next observation, reward and done info
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            next_observation = decision_steps[tracked_agent].obs
            reward = decision_steps[tracked_agent].reward
            done = tracked_agent in terminal_steps

            # add data to buffer
            transition = ([observation, action, reward, next_observation, done])
            buffer.push(transition)

            observation = np.concatenate((next_observation[0], next_observation[1]))

        # update episode scores

        # train agent

    env.close()
    logger.close()

        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_mode", nargs="?", type=str, 
        default="single", help="Either 'single' or 'multi'")
    parser.add_argument("--env_path", nargs="?", type=str, 
        default="./builds/DRL-Unity_Windows_x86_64", help="Path to Unity Exe")
    parser.add_argument("--log_dir", nargs="?", type=str, 
        default="./logs", help="Path directory where log files are stored")
    args = parser.parse_args()

    # TODO: read configs from file instead of hardcoding
    config = {
        'buffer_size': 50000,
        'n_steps': 5,
        'discount_rate': 0.99,
        'number_of_episodes': 1,
        'episode_length': 10,
        'action_size': 5
    }

    if (args.agent_mode == "single"):
        train_single_agent(args.env_path, args.log_dir, config)
    elif (args.agent_mode == "multi"):
        raise NotImplementedError
    else:
        print(f"Agent mode {args.agent_mode} invalid! Must be 'single' or 'multi'")


