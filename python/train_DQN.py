import argparse
import progressbar as pb
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
from collections import deque

from mlagents_envs.environment import UnityEnvironment, ActionTuple
from DQN_agent import DeepQAgent


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

    # create DQN agent
    agent = DeepQAgent(
        config['action_size'], 
        config['state_size'], 
        epsilon=0.2, 
        brain=agent_spec,
        buffer_size=config['buffer_size'], 
        batch_size=config['batch_size'],
        episodes=config['train_episodes'],
        gamma=config['discount_rate'])

    # set up cumulative rewards
    returns = []
    losses = []

    # TODO: move inside loop?
    step = 0
    loss = -1

    # training loop
    for episode in range(config['number_of_episodes']):

        print(f'EPISODE {episode}')

        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        done = False 
        tracked_agent = -1
        reward_sum = 0

        observation = decision_steps[0].obs
        observation = np.concatenate((observation[0], observation[1]))

        # TODO: move into loop
        agent.update_target('hard')

        while not done:
            if tracked_agent == -1 and len(decision_steps) >= 1:
                tracked_agent = decision_steps.agent_id[0]

            # choose greedy action based on Q(s, a; theta)
            action = agent.choose_action(tf.expand_dims(observation, 0))
            action_tuple = ActionTuple()
            action_tuple.add_discrete(action)

            # update the environment based on the chosen action
            env.set_actions(behavior_name, action_tuple)
            env.step()

            # get next observation, reward and done info
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            if tracked_agent in decision_steps:
                next_observation = decision_steps[tracked_agent].obs
                reward = decision_steps[tracked_agent].reward
            if tracked_agent in terminal_steps:
                next_observation = terminal_steps[tracked_agent].obs
                reward = terminal_steps[tracked_agent].reward
            done = tracked_agent in terminal_steps

            reward_sum += reward

            # add trajectory to buffer
            next_observation = np.concatenate((next_observation[0], next_observation[1]))
            transition = ([observation, action, reward, next_observation, done])
            agent.memory.push(transition)

            observation = next_observation

            # if target update frequency: update target network -> TODO: make it work inside the loop
            # if step % config['target_update_frequency']:
            #     agent.update_target('hard')

            # if udate frequency: train q-network
            if step % config['train_frequency']:
                loss = agent.learn()

            step += 1

        returns.append(reward_sum)
        losses.append(loss)
        print(f"--> rewards: {reward_sum}")
        print(f"--> loss: {loss}")

    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_mode", nargs="?", type=str, 
        default="single", help="Either 'single' or 'multi'")
    parser.add_argument("--env_path", nargs="?", type=str, 
        default="./builds/DRL-Unity_Windows", help="Path to Unity Exe")
    parser.add_argument("--log_dir", nargs="?", type=str, 
        default="./logs", help="Path directory where log files are stored")
    args = parser.parse_args()

    # TODO: read configs from file instead of hardcoding
    config = {
        'buffer_size': 50000,
        'discount_rate': 0.99,
        'number_of_episodes': 100,
        'episode_length': 10,
        'action_size': 5,
        'batch_size': 16,
        'learn_steps_per_env_step': 3,
        'state_size': 108, 
        'train_frequency': 5,
        'target_update_frequency': 20,
        'train_episodes': 3
    }

    if (args.agent_mode == "single"):
        train_single_agent(args.env_path, args.log_dir, config)
    elif (args.agent_mode == "multi"):
        raise NotImplementedError
    else:
        print(f"Agent mode {args.agent_mode} invalid! Must be 'single' or 'multi'")