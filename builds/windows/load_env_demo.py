from mlagents_envs.environment import UnityEnvironment

env = UnityEnvironment(
    file_name="DRL-Unity", 
    seed=1, 
    side_channels=[],
    log_folder='./logs')

env.reset()

# Get action & observation info
behavior_name = list(env.behavior_specs)[0]
spec = env.behavior_specs[behavior_name]

print("SPEC INFO:")
print(f"-- Number of observations: {len(spec.observation_specs)}")
if spec.action_spec.is_continuous():
    print("-- Action space: continuous")
elif spec.action_spec.is_discrete():
    print(f"-- Action space: discrete (size={spec.action_spec.discrete_size})")

# "training loop"
for episode in range(3):
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    tracked_agent = -1
    done = False
    episode_rewards = 0

    while not done:
        # track agent
        if tracked_agent == -1 and len(decision_steps) >= 1:
            tracked_agent = decision_steps.agent_id[0]
        
        # create and take action
        action = spec.action_spec.random_action(len(decision_steps))
        env.set_actions(behavior_name, action)
        env.step()

        # set reward 
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        if tracked_agent in decision_steps:
            episode_rewards += decision_steps[tracked_agent].reward

        # check for terminal state
        if tracked_agent in terminal_steps:
            episode_rewards += decision_steps[tracked_agent].reward
            done = True

    print(f"Total rewards for episode {episode} is {episode_rewards}")

env.close()