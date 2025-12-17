"""
Vectorized Flight Environment for Stable-Baselines3

Modern implementation using Gymnasium interface, compatible with SB3.
This replaces the legacy FlightEnvVec that used stable-baselines (v2) VecEnv.

Purpose:
- To wrap the C++ flightgym environment and provide a Gymnasium interface
- To support vectorized environments natively through the flightgym C++ backend
- To support Stable-Baselines3, Gymnasium, Unity rendering, curriculum learning, extra info, and episode tracking
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List

from ..core import BaseFlightEnv


class FlightEnvVec(BaseFlightEnv):
    """ 
    Gymnasium-compatible wrapper for Flightmare vectorized environments.
    
    This class wraps the C++ flightgym environments and provides a Gymnasium interface
    that is compatible with Stable-Baselines3. It supports vectorized environments
    natively through the flightgym C++ backend.
    
    Args:
        impl: The flightgym environment implementation (C++ backend)
        max_episode_steps: Maximum steps per episode (default: 300)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, impl, max_episode_steps: int = 300):
        # Get num_envs from C++ backend before calling super().__init__()
        num_envs = impl.getNumOfEnvs()
        super().__init__(num_envs=num_envs)
        
        self.wrapper = impl     #### C++ backend, doesn't care about type, just interface 
        self.num_obs = self.wrapper.getObsDim() #### Get from C++, returns the number of observations 
        self.num_acts = self.wrapper.getActDim() #### Get from C++, returns the number of actions
        self.max_episode_steps = max_episode_steps #### Maximum steps per episode
        
        # Define observation and action spaces
        self.observation_space = spaces.Box( #### Continuous space
            low=-np.inf,
            high=np.inf,
            shape=(self.num_obs,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box( ### Normalized to [-1, 1]
            low=-1.0, ### Minimum action value
            high=1.0, ### Maximum action value
            shape=(self.num_acts,), ### Number of actions
            dtype=np.float32 ### 32-bit float
        )
        
        # Internal state buffers ### Pre-allocated buffers to avoid memory allocation in hot loop
        self._observation = np.zeros([self._num_envs, self.num_obs], dtype=np.float32) ### 2D array for vectorized environments
        self._reward = np.zeros(self._num_envs, dtype=np.float32)
        self._terminated = np.zeros(self._num_envs, dtype=np.bool_)
        self._truncated = np.zeros(self._num_envs, dtype=np.bool_)
        
        # Extra info handling
        self._extra_info_names = self.wrapper.getExtraInfoNames()
        self._extra_info = np.zeros(
            [self._num_envs, len(self._extra_info_names)],
            dtype=np.float32
        )
        
        # Episode tracking for each environment
        self.episode_rewards = [[] for _ in range(self._num_envs)]
        self._episode_lengths = np.zeros(self._num_envs, dtype=np.int32)
        
    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed for reproducibility."""
        super().seed(seed)
        if seed is not None:
            self.wrapper.setSeed(seed)
    
    def reset(
        ### Overrides the reset method in gym.Env
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options (not used)
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        if seed is not None:
            self.seed(seed)
        
        self._reward.fill(0.0)
        self._terminated.fill(False)
        self._truncated.fill(False)
        self._episode_lengths.fill(0)
        self.episode_rewards = [[] for _ in range(self._num_envs)]
        
        self.wrapper.reset(self._observation)
        
        info = {}
        return self._observation[0].copy(), info
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation: Next observation
            reward: Reward received
            terminated: Whether episode ended due to task completion/failure
            truncated: Whether episode ended due to time limit
            info: Additional information
        """
        # Ensure action is 2D for vectorized environments
        if action.ndim == 1:
            action = action.reshape(1, -1)
        
        # Call C++ backend
        # Note: Legacy wrapper uses _done as a single boolean array (deprecated in Gymnasium)
        # We'll use it for terminated and handle truncation separately
        done_array = np.zeros(self._num_envs, dtype=np.bool_)
        self.wrapper.step(
            action.astype(np.float32),
            self._observation,
            self._reward,
            done_array,
            self._extra_info
        )
        
        # Update episode tracking
        self._episode_lengths += 1
        for i in range(self._num_envs):
            self.episode_rewards[i].append(self._reward[i])
        
        # Build info dict
        info = {}
        
        # Add extra info if available
        if len(self._extra_info_names) > 0:
            info['extra_info'] = {
                name: self._extra_info[0, j]
                for j, name in enumerate(self._extra_info_names)
            }
        
        # Check for episode completion (only for first env in vectorized case)
        terminated = bool(done_array[0]) ### Episode ended due to task completion/failure
        truncated = self._episode_lengths[0] >= self.max_episode_steps ### Episode ended due to time limit
        
        if terminated or truncated:
            # Calculate episode statistics
            episode_reward = sum(self.episode_rewards[0])
            episode_length = self._episode_lengths[0]
            
            info['episode'] = {
                'r': episode_reward,
                'l': episode_length,
            }
            
            # Reset tracking for this environment
            self.episode_rewards[0].clear()
            self._episode_lengths[0] = 0
        
        return (
            self._observation[0].copy(),
            float(self._reward[0]),
            terminated,
            truncated,
            info
        )
    
    def render(self):
        """Render the environment (handled by Unity)"""
        # Rendering is handled by the Unity bridge
        pass
    
    def close(self):
        """Close the environment and clean up resources"""
        if hasattr(self, 'wrapper'):
            self.wrapper.close()
    
    # Additional utility methods for Unity integration
    
    def connect_unity(self):
        """Connect to Unity renderer"""
        if hasattr(self.wrapper, 'connectUnity'):
            self.wrapper.connectUnity()
    
    def disconnect_unity(self):
        """Disconnect from Unity renderer"""
        if hasattr(self.wrapper, 'disconnectUnity'):
            self.wrapper.disconnectUnity()
    
    def step_unity(self, action: np.ndarray, send_id: int):
        """
        Step with Unity synchronization.
        
        Args:
            action: Action to take
            send_id: ID for synchronization
            
        Returns:
            receive_id: Received ID from Unity
        """
        if hasattr(self.wrapper, 'stepUnity'):
            return self.wrapper.stepUnity(
                action.astype(np.float32),
                self._observation,
                self._reward,
                self._terminated,
                self._extra_info,
                send_id
            )
        return send_id
    
    def curriculum_callback(self):
        """Curriculum learning callback"""
        if hasattr(self.wrapper, 'curriculumUpdate'):
            self.wrapper.curriculumUpdate()
    
    # Properties for vectorized environment support
    # num_envs property inherited from BaseFlightEnv
    
    @property
    def extra_info_names(self) -> List[str]:
        """Names of extra info fields"""
        return self._extra_info_names


class FlightEnvVecSB3(gym.Env):
    """
    Alternative wrapper optimized for SB3 VecEnv usage.
    
    This wrapper is designed to work with SB3's vectorization utilities
    (DummyVecEnv, SubprocVecEnv) for better multi-processing support.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, impl, env_index: int = 0):
        """
        Initialize single environment from vectorized backend.
        
        Args:
            impl: The flightgym environment implementation
            env_index: Index of this environment in the vectorized backend
        """
        super().__init__()
        
        self.wrapper = impl
        self.env_index = env_index
        self.num_obs = self.wrapper.getObsDim()
        self.num_acts = self.wrapper.getActDim()
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_obs,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_acts,),
            dtype=np.float32
        )
        
        # Buffers (allocate for all envs but only use one)
        num_envs = self.wrapper.getNumOfEnvs()
        self._obs_buffer = np.zeros([num_envs, self.num_obs], dtype=np.float32)
        self._reward_buffer = np.zeros(num_envs, dtype=np.float32)
        self._done_buffer = np.zeros(num_envs, dtype=np.bool_)
        
        extra_info_names = self.wrapper.getExtraInfoNames()
        self._extra_info_buffer = np.zeros(
            [num_envs, len(extra_info_names)],
            dtype=np.float32
        )
        self._extra_info_names = extra_info_names
        
        self.episode_reward = 0.0
        self.episode_length = 0
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment"""
        if seed is not None:
            self.wrapper.setSeed(seed)
        
        self.wrapper.reset(self._obs_buffer)
        self.episode_reward = 0.0
        self.episode_length = 0
        
        return self._obs_buffer[self.env_index].copy(), {}
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step"""
        # Prepare action for all envs (but only this one matters)
        action_array = np.zeros([self.wrapper.getNumOfEnvs(), self.num_acts], dtype=np.float32)
        action_array[self.env_index] = action
        
        self.wrapper.step(
            action_array,
            self._obs_buffer,
            self._reward_buffer,
            self._done_buffer,
            self._extra_info_buffer
        )
        
        obs = self._obs_buffer[self.env_index].copy()
        reward = float(self._reward_buffer[self.env_index])
        terminated = bool(self._done_buffer[self.env_index])
        truncated = False  # Handled by TimeLimit wrapper
        
        self.episode_reward += reward
        self.episode_length += 1
        
        info = {}
        if terminated:
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.episode_length,
            }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render (handled by Unity)"""
        pass
    
    def close(self):
        """Close environment"""
        pass
