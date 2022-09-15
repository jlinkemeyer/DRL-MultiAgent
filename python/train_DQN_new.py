import argparse
import progressbar as pb
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from collections import deque

from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from DQN_agent import DeepQAgent
from DDQN_agent import DoubleDeepQAgent


# TODO: adjust for multi-agent
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
        no_graphics=True
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


def train_single_agent(env_path, log_dir, config):
    env, behavior_name, agent_spec = setup_environment(env_path, log_dir, verbose=True)

    # create DQN agent
    agent = DoubleDeepQAgent(
        config['action_size'], 
        config['state_size'], 
        epsilon=config['epsilon'], 
        epsilon_min=config['epsilon_min'],
        epsilon_decay=config['epsilon_decay'],
        brain=agent_spec,
        buffer_size=config['buffer_size'], 
        batch_size=config['batch_size'],
        episodes=config['train_episodes'],
        gamma=config['discount_rate'])

    # set up cumulative rewards
    scores = deque(maxlen=100)
    losses = []
    means = []

    # TODO: move inside loop?
    step = 0
    loss = -1

    # training loop
    for episode in range(config['number_of_episodes']):

        # print(f'EPISODE {episode}')

        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        done = False 
        tracked_agent = -1
        score = 0

        observations = []
        for env_nr in range (config['n_envs']):
            observation = decision_steps[env_nr].obs
            observation = np.concatenate((observation[0], observation[1], observation[2]))
            observations.append(observation)
            
        while not done:
            if step % config['target_update_frequency'] == 0 and agent.sufficient_experience():
                agent.update_target('hard')
                print("--------->TARGET UPDATE")
            # choose greedy action based on Q(s, a; theta)
            actions = []
            for env_nr in range(config['n_envs']):
                action = agent.choose_action(tf.expand_dims(observations[env_nr], 0))
                actions.append(action)
            actions = [np.squeeze(action) for action in actions]
            actions = np.array(actions)
            actions = np.expand_dims(actions, 1)

            action_tuple = ActionTuple()
            action_tuple.add_discrete(actions)

            # update the environment based on the chosen action
            env.set_actions(behavior_name, action_tuple)
            env.step()

            # get next observation, reward and done info
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            if len(terminal_steps) != 0:
                done = True

            next_observations = []
            reward_all_areas = 0
            for env_nr in range (config['n_envs']):
                if env_nr in decision_steps:
                    next_observation = decision_steps[env_nr].obs
                    reward = decision_steps[env_nr].reward
                if env_nr in terminal_steps:
                    next_observation = terminal_steps[env_nr].obs
                    reward = terminal_steps[env_nr].reward

                # add trajectory to buffer
                try:
                    next_observation = np.concatenate((next_observation[0], next_observation[1], next_observation[2]))
                    next_observations.append(next_observation)
                    #transition = ([observations[env_nr], actions[env_nr], reward, next_observation, done])
                    agent.memory.push(observations[env_nr], actions[env_nr], reward, next_observation, done)
                except ValueError:
                    pass

                reward_all_areas += reward

            score += reward_all_areas / config['n_envs']
            observations = next_observations

            # if target update frequency: update target network -> TODO: make it work inside the loop
            # if step % config['target_update_frequency']:
            #     agent.update_target('hard')

            # if udate frequency: train q-network
            if step % config['train_frequency'] == 0:
                output = agent.learn()

                if output:
                    loss, td_error, predictions = output

            step += 1

        """
        try:
            print('td_error: ', td_error)
            print('predictions: ', predictions)
        except:
            pass
        """

        scores.append(score)
        mean_reward = np.mean(np.array(scores))
        means.append(mean_reward)
        losses.append(loss)
        print(f" -- episode: {episode} | score: {score} | mean reward: {mean_reward} | loss: {loss}")

        if episode % config['plot_frequency'] == 0 and episode > 0:
            x = np.arange(episode + 1)

            try:
                plt.figure()
                plt.plot(x, means, color='darkblue')
                plt.title('Mean Reward')
                plt.xlabel('Episode')
                plt.ylabel('Mean Reward')
                plt.savefig(f'./output/mean_reward_episode_{episode}.png')
            except ValueError:
                pass


            try:
                plt.figure()
                x_values = np.arange(len(scores))
                plt.plot(x_values, scores, color='darkblue', alpha=0.5)
                plt.plot(x_values, means, color='darkblue')
                plt.title('Scores')
                plt.xlabel('Episode')
                plt.ylabel('Score')
                plt.savefig(f'./output/score_episode_{episode}.png')
            except ValueError:
                pass

            try:
                plt.figure()
                plt.plot(x, losses, color='darkblue')
                plt.title('Loss')
                plt.xlabel('Episode')
                plt.ylabel('MSE Loss')
                plt.savefig(f'./output/loss_episode_{episode}.png')
            except ValueError:
                pass

        if episode % config['save_frequency'] == 0 and episode > 0:
            agent.q_network.save(f'./models/q_network_ep_{episode}')


        #agent.decay_epsilon()

    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_mode", nargs="?", type=str, 
        default="single", help="Either 'single' or 'multi'")
    parser.add_argument("--env_path", nargs="?", type=str, 
        default="./builds/Newest", help="Path to Unity Exe")
    parser.add_argument("--log_dir", nargs="?", type=str, 
        default="./logs", help="Path directory where log files are stored")
    args = parser.parse_args()

    # TODO: read configs from file instead of hardcoding
    config = {
        'buffer_size': 50000,
        'discount_rate': 0.99,
        'number_of_episodes': 5000,
        'action_size': 5,
        'batch_size': 128,
        'state_size': 108, 
        'train_frequency': 5,
        'target_update_frequency': 5000,
        'train_episodes': 3,
        'n_envs': 1,
        'epsilon': 0.1,
        'epsilon_min': 0.001,
        'epsilon_decay': 0.9,
        'plot_frequency': 50,
        'save_frequency': 10
    }

    if (args.agent_mode == "single"):
        train_single_agent(args.env_path, args.log_dir, config)
    elif (args.agent_mode == "multi"):
        raise NotImplementedError
    else:
        print(f"Agent mode {args.agent_mode} invalid! Must be 'single' or 'multi'")