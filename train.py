import argparse
import numpy as np
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from distutils.util import strtobool

from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
# from DQN_agent import DeepQAgent
from DDQN_agent import DoubleDeepQAgent


def setup_environment(file_name, log_dir, verbose=True):
    """
    Creates UnityEnvironment object from Unity environment binary
    and extracts 
    """

    channel = EngineConfigurationChannel()

    # create environment
    env = UnityEnvironment(
        file_name,
        seed=1,
        side_channels=[channel],
        log_folder=log_dir,
        no_graphics=False
    )
    channel.set_configuration_parameters(time_scale=2.0)
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


def train_single_agent(env_path, train, log_dir, incr_batch, decr_lr, config):
    """
    Training function. Contains the main training loop.

    :param env_path: Path to the Unity executable
    :param log_dir: directory to save console logs in
    :param incr_batch:
    :param decr_lr:
    :param config: training configuration information
    :return:
    """
    # Use the gym wrapper to create a controllable environment
    env, behavior_name, agent_spec = setup_environment(env_path, log_dir, verbose=True)

    # Create an agent with given parameters
    agent = DoubleDeepQAgent(
        action_size=config['action_size'],
        epsilon=config['epsilon'], 
        epsilon_min=config['epsilon_min'],
        epsilon_decay=config['epsilon_decay'],
        brain=agent_spec,
        buffer_size=config['buffer_size'], 
        batch_size=config['batch_size'],
        epochs=config['train_epochs'],
        gamma=config['discount_rate'],
        alpha=config['learning_rate'],
        batch_factor=config['batch_factor'],
        decr_lr=decr_lr,
        lr_decay_steps=config['lr_decay_steps'],
        lr_decay_rate=config['lr_decay_rate']
    )

    # set up cumulative rewards
    returns = deque(maxlen=100)
    means, losses = [], []
    if incr_batch:
        batch_sizes = []
    if decr_lr:
        learn_rates = []
    step = 0
    loss = -1

    # Main training loop - generate one episode per iteration
    for episode in range(config['number_of_episodes']):

        # Start episode by resetting environment (set agent to initial position, random goal position, random
        # instruction (score with red or blue) and initializing sum of rewards
        env.reset()
        decision_steps, _ = env.get_steps(behavior_name)
        reward_sum = 0
        tracked_agent = -1  # The tracked agent in the scene

        # Get the current observation of the agent
        observation = decision_steps[0].obs
        observation = np.concatenate((observation[0], observation[1], observation[2]))  # TODO

        # TODO (batch increase also in learn function?)
        if incr_batch and agent.sufficient_experience() and episode % config['batch_incr_freq'] == 0 and not episode == 0:
            agent.increase_batch_size()
            
        while True:
            # Update the target network in the set update frequency, but only if there are enough samples in the replay
            # buffer since else no weight updates took place yet
            if train and step % config['target_update_frequency'] == 0 and agent.sufficient_experience():
                agent.update_target('hard')
                print("--------->TARGET UPDATE")

            # Access the current agent in the scene
            if tracked_agent == -1 and len(decision_steps) >= 1:
                tracked_agent = decision_steps.agent_id[0]

            # TODO is the action selection the problem?
            # Determine the action based on the observation
            # action = agent.choose_action(tf.expand_dims(observation, 0))
            # action_tuple = ActionTuple()
            # action_tuple.add_discrete(action)

            # -------------- TODO from old code
            # choose greedy action based on Q(s, a; theta)
            actions = []
            for env_nr in range(config['n_envs']):
                action = agent.choose_action(tf.expand_dims(observation, 0))
                actions.append(action)
            actions = [np.squeeze(action) for action in actions]
            actions = np.array(actions)
            actions = np.expand_dims(actions, 1)
            action_tuple = ActionTuple()
            action_tuple.add_discrete(actions)
            # ---------------

            # Perform the determined action
            env.set_actions(behavior_name, action_tuple)
            env.step()

            # Store the reward for the action and the next observation
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            if tracked_agent in decision_steps:
                next_observation = decision_steps[tracked_agent].obs
                reward = decision_steps[tracked_agent].reward
            if tracked_agent in terminal_steps:
                next_observation = terminal_steps[tracked_agent].obs
                reward = terminal_steps[tracked_agent].reward
            done = tracked_agent in terminal_steps  # Check whether the environment is marked as done

            # Get the next observation
            next_observation = np.concatenate((next_observation[0], next_observation[1], next_observation[2]))

            # Add transition to replay buffer
            agent.memory.push(observation, action, reward, next_observation, done)

            # set new observation to current observation for determining next action
            observation = next_observation

            # Update sum of rewards
            reward_sum += reward

            # Train q-network according to training frequency, but only if the memory buffer contains enough samples
            if train and step % config['train_frequency'] == 0:
                if agent.sufficient_experience():
                    l = []
                    for _ in range(agent.epochs):
                        l.append(agent.learn())
                    loss = np.mean(np.array(l))

            step += 1

            # Break when the end of an episode is reached
            if done:
                break

        # Epsilon decay to decrease exploration over time
        if train:
            agent.decay_epsilon()

        # Save checkpoint
        if train and episode % config['checkpoint_save_frequency'] == 0:
            agent.manager.save()

        # Update all lists to track progress over time
        returns.append(reward_sum)
        mean_reward = np.mean(np.array(returns))
        means.append(mean_reward)

        # Track progress
        losses.append(loss)
        if decr_lr:
            learn_rates.append(agent.get_learning_rate())
        if incr_batch:
            batch_sizes.append(agent.get_batch_size())
        print(f" -- episode: {episode} | reward sum: {reward_sum} | mean reward sum: {mean_reward} | loss: {loss}")

        # Create plots to allow visual progress tracking
        if train and episode % config['plot_frequency'] == 0 and episode > 0:

            visualize(data=means, save_path=f'./output/mean_reward_episode_{episode}.png', title='Mean Reward')
            visualize(data=means, save_path=f'./output/score_episode_{episode}.png', title='Scores', data2=returns)
            visualize(data=losses, save_path=f'./output/loss_episode_{episode}.png', title='MSE Loss')

            if incr_batch:
                visualize(data=batch_sizes, save_path=f'./output/batch_size_episode_{episode}.png', title='Batch size')

            if decr_lr:
                visualize(data=learn_rates, save_path=f'./output/learn_rate_episode_{episode}.png',
                          title='Learning rate')

    env.close()


def visualize(data, title, save_path, data2=None):
    """
    Visualizes one-dimensional data in a plot

    :param data: 1-d array or list
    :param title: title is used for the plot and y-axis description
    :param save_path: path where to save the plot as a .png file
    :param data2: special case when two arrays should be plotted in the same image # TODO not working properly
    """
    try:
        plt.figure()
        plt.plot(data, color='darkblue')
        if data2:
            plt.plot(data2, color='darkblue', alpha=0.5)
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel(title)
        plt.savefig(save_path)
        plt.close()
    except ValueError:
        pass


if __name__ == "__main__":

    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_mode", nargs="?", type=str, 
        default="single", help="Either 'single' or 'multi'")
    parser.add_argument("--env_path", nargs="?", type=str, 
        default="./builds/Newest", help="Path to Unity Exe")
    parser.add_argument("--log_dir", nargs="?", type=str, 
        default="./logs", help="Path directory where log files are stored")
    parser.add_argument("--incr_batch", nargs="?", type=strtobool, 
        default=False, help="Whether to gradually increase the batch size, either 'True' or 'False'")
    parser.add_argument("--decr_lr", nargs="?", type=strtobool, 
        default=False, help="Whether to gradually decay the learning rate, either 'True' or 'False'")
    parser.add_argument("--train", nargs="?", type=strtobool, 
        default=True, help="Whether to train the agent, either 'True' or 'False'")
    args = parser.parse_args()

    # TODO: read configs from file instead of hardcoding
    config = {
        'buffer_size': 50000,
        'learning_rate': 0.001,
        'discount_rate': 0.99,
        'number_of_episodes': 5000,
        'action_size': 5,
        'batch_size': 256,
        'train_frequency': 5,
        'target_update_frequency': 5000,
        'train_epochs': 3,
        'n_envs': 1,
        'epsilon': 0.1,
        'epsilon_min': 0.001,
        'epsilon_decay': 0.99,
        'plot_frequency': 50,
        'save_frequency': 10,
        'batch_incr_freq': 500,
        'batch_factor': 2,
        'lr_decay_steps': 10000,
        'lr_decay_rate': 0.99,
        'checkpoint_save_frequency': 20
    }

    # Calling different training functions depending on which agent mode (single- or multi-agent) is chosen by the user.
    # Currently, only 'single' is possible. Due to time constraints, the 'multi' option is not implemented
    if args.agent_mode == "single":
        train_single_agent(args.env_path, args.train, args.log_dir, args.incr_batch, args.decr_lr, config)
    elif args.agent_mode == "multi":
        raise NotImplementedError
    else:
        print(f"Agent mode {args.agent_mode} invalid! Must be 'single' or 'multi'")