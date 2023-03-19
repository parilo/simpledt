

# class ObsDictWrapper(gym.Wrapper):
#     def __init__(self, env: gym.Env, obs_key='observation'):
#         super().__init__(env)
#         self.obs_key = obs_key
#         self.observation_space = self._get_observation_space()

#     def _get_observation_space(self) -> gym.spaces.Dict:
#         # Get the original observation space
#         obs_space = self.env.observation_space
#         # If the observation space is already a dictionary, return it
#         if isinstance(obs_space, gym.spaces.Dict):
#             return obs_space
#         # Otherwise, create a new observation space that is a dictionary
#         return gym.spaces.Dict({self.obs_key: obs_space})

#     def step(self, action):
#         # Call the parent environment's step function
#         obs, reward, done, info = self.env.step(action)
#         # Wrap the observation in a dictionary if it is not already a dictionary
#         if not isinstance(obs, dict):
#             obs = {self.obs_key: obs}
#         # Return the wrapped observation, along with the other step outputs
#         return obs, reward, done, info

#     def reset(self, **kwargs):
#         # Call the parent environment's reset function
#         ret = self.env.reset(**kwargs)
#         obs = ret[0]
#         # Wrap the observation in a dictionary if it is not already a dictionary
#         if not isinstance(obs, dict):
#             obs = {self.obs_key: obs}
#         # Return the wrapped observation
#         return (obs,) + ret[1:]
