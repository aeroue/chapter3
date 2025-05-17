# weak_communication/active_inference.py

import numpy as np
import copy
from .aif_functions_wrapper import AIFWrapper

# try:
#     # Assuming AIFWrapper is in the same package or accessible
#     from .aif_functions_wrapper import AIFWrapper
# except ImportError:
#     # Fallback for standalone execution or if aif_functions_wrapper is in a different relative path
#     try:
#         from aif_functions_wrapper import AIFWrapper
#     except ImportError:
#         print("CRITICAL ERROR: AIFWrapper class not found. Ensure aif_functions_wrapper.py is accessible.")
#         # Define a dummy AIFWrapper to allow parsing, this will not function correctly.
#         class AIFWrapper:
#             def __init__(self, agent_params): self.agent_params = agent_params; self.current_belief_q = np.array([0.5, 0.5])
#             def choose_action_to_minimize_efe(self, *args, **kwargs): return 0.0, 0.0, 0.0, self.current_belief_q # vel, head, efe, belief
#             def update_belief_q(self, new_q): self.current_belief_q = new_q
#             def get_likelihood_for_system_config(self, *args, **kwargs): return self.current_belief_q


class ActiveInferenceAgent:
    """
    Represents a single spacecraft agent performing active inference
    for decision-making in a weak communication environment.
    """

    def __init__(self, agent_id: int, agent_specific_params: dict, all_agent_types: list[str]):
        """
        Initializes the Active Inference Agent.

        Args:
            agent_id (int): The unique ID of this agent.
            agent_specific_params (dict): Parameters for this agent, structured for AIFWrapper.
                                          This should include keys like 'goals', 'velocity_options',
                                          'heading_options', 'reward_configs', 'observation_error_std',
                                          'dt', 'env_size', 'max_distance_measure', etc.,
                                          AND 'agent_id', 'num_agents'.
            all_agent_types (list[str]): List of types for all agents in the system,
                                         e.g., ['A', 'B', 'A']. The type for this
                                         agent_id should correspond.
        """
        self.agent_id = agent_id
        self.params = copy.deepcopy(agent_specific_params)

        # Ensure essential parameters for AIFWrapper are present
        self.params['agent_id'] = self.agent_id
        self.params['agent_types'] = all_agent_types # AIFWrapper needs the list of all agent types
        # num_agents should be part of agent_specific_params or derived
        if 'num_agents' not in self.params:
            self.params['num_agents'] = len(all_agent_types)

        self.aif_wrapper = AIFWrapper(self.params)
        self.current_state_estimate = np.array(self.params['agent_positions'][self.agent_id, :]) # Own position [x,y,theta]
        # The belief Q over system configurations (e.g., task assignments) is managed by aif_wrapper
        # self.belief_over_system_config_q = self.aif_wrapper.current_belief_q

        self.reasoning_mode = self.params.get('reasoning_mode', 'higher_order') # Default, can be overridden


    def update_internal_state_estimate(self, new_own_position: np.ndarray):
        """
        Updates the agent's estimate of its own physical state.
        Args:
            new_own_position (np.ndarray): New [x, y, theta] for this agent.
        """
        if new_own_position.shape != (3,): # Assuming x, y, theta
            raise ValueError("New position must be of shape (3,)")
        self.current_state_estimate = new_own_position.copy()

    def update_belief_based_on_observations(self, all_observed_positions: np.ndarray):
        """
        Updates the agent's belief Q over system configurations based on
        observations of all agents' positions.

        Args:
            all_observed_positions (np.ndarray): Observed positions of all agents in the system,
                                                 shape (num_agents, 3) -> [[x,y,theta]_0, ..., [x,y,theta]_N-1].
        """
        if all_observed_positions.shape != (self.params['num_agents'], 3):
            raise ValueError(f"all_observed_positions shape mismatch. Expected ({self.params['num_agents']}, 3)")

        # The get_likelihood_for_system_config method in AIFWrapper updates its internal belief (self.current_belief_q)
        # and returns the new posterior.
        new_posterior_q = self.aif_wrapper.get_likelihood_for_system_config(
            observed_agent_positions=all_observed_positions,
            reasoning_mode=self.reasoning_mode
        )
        # self.belief_over_system_config_q is now updated within aif_wrapper
        # print(f"Agent {self.agent_id} updated belief Q: {np.round(self.aif_wrapper.current_belief_q,2)}")


    def decide_action(self, all_current_agent_positions: np.ndarray) -> tuple[float, float, float]:
        """
        Decides the next action (velocity, heading_change) by minimizing EFE.

        Args:
            all_current_agent_positions (np.ndarray): Current estimated/observed positions
                                                      of all agents in the system.

        Returns:
            tuple: (selected_velocity, selected_heading_change, efe_score_of_action)
        """
        if all_current_agent_positions.shape != (self.params['num_agents'], 3):
            raise ValueError(f"all_current_agent_positions shape mismatch. Expected ({self.params['num_agents']}, 3)")

        # Ensure own position is consistent with the input for decision making
        # self.current_state_estimate = all_current_agent_positions[self.agent_id, :].copy() # Redundant if already updated

        velocity, heading_delta, efe_score, updated_q = self.aif_wrapper.choose_action_to_minimize_efe(
            current_agent_positions=all_current_agent_positions,
            reasoning_mode=self.reasoning_mode
        )
        # The belief self.aif_wrapper.current_belief_q is updated by choose_action_to_minimize_efe
        # print(f"Agent {self.agent_id} chose action: Vel={velocity:.2f}, HeadChg={heading_delta:.2f} with EFE={efe_score:.3f}")
        return velocity, heading_delta, efe_score

    def get_current_belief(self) -> np.ndarray:
        """Returns the agent's current belief over system configurations."""
        return self.aif_wrapper.current_belief_q.copy()

    def set_reasoning_mode(self, mode: str):
        """
        Sets the reasoning mode for the agent.
        Args:
            mode (str): 'higher_order', 'first_order', or 'zero_order'.
        """
        if mode not in ['higher_order', 'first_order', 'zero_order']:
            raise ValueError("Invalid reasoning mode. Must be 'higher_order', 'first_order', or 'zero_order'.")
        self.reasoning_mode = mode
        self.params['reasoning_mode'] = mode # Update in params too for AIFWrapper re-init if ever needed
        # Re-initialize belief structure if mode changes how belief is structured (1D vs 2D)
        # Current AIFWrapper._initialize_belief handles this based on reasoning_mode in params.
        # We might need to explicitly reset/reinitialize the belief in the wrapper or here.
        # For now, assume AIFWrapper handles prior shape correctly on next call, or re-init wrapper.
        self.aif_wrapper = AIFWrapper(self.params) # Re-initialize wrapper to reset belief structure


if __name__ == '__main__':
    print("Active Inference Agent Example")

    # --- Mocking necessary parameters and classes for testing ---
    num_agents_test = 2
    num_goals_test = 2
    env_size_test = 20.0

    # Define goals and initial positions
    goals_test_np = np.array([[5.0, 5.0], [15.0, 15.0]])
    initial_positions_np = np.array([
        [1.0, 1.0, 0.0],    # Agent 0: x, y, theta
        [18.0, 18.0, np.pi] # Agent 1: x, y, theta
    ])
    agent_types_list = ['A', 'B']

    # Define reward_configs (possible system goal assignments)
    # For 2 agents, 2 goals, if exclusive assignment (e.g., tasks)
    # reward_configs_test = [(0,1), (1,0)]
    # For rendezvous (both agents to same goal)
    reward_configs_test = [(0,0), (1,1)]


    # Base parameters shared by all agents for the AIF core
    # This structure should align with what aif_functions_isobeliefs_convergent.py's
    # parse_args_by_agent would create from a global args dictionary.
    # The ActiveInferenceAgent will receive its *specific slice* of these params,
    # plus some global info like all_agent_types and num_agents.

    agent0_specific_params = {
        # 'agent_id' will be set by ActiveInferenceAgent constructor
        # 'num_agents' will be set by ActiveInferenceAgent constructor
        # 'agent_types' will be set by ActiveInferenceAgent constructor
        'goals': goals_test_np,
        'agent_positions': initial_positions_np, # Used for initial state in AIFWrapper if needed
        'velocity_options': np.array([0.0, 0.5, 1.0]),
        'heading_options': np.array([-np.pi/8, 0, np.pi/8]),
        'num_actions': 3 * 3,
        'observation_error_std': 0.2,
        'reward_configs': reward_configs_test,
        'dt': 1.0,
        'env_size': env_size_test,
        'max_distance_measure': env_size_test * 1.5,
        'reasoning_mode': 'higher_order' # Default for this agent
    }

    # Create Agent 0
    agent0 = ActiveInferenceAgent(
        agent_id=0,
        agent_specific_params=agent0_specific_params,
        all_agent_types=agent_types_list
    )
    print(f"Agent {agent0.agent_id} initialized. Reasoning: {agent0.reasoning_mode}")
    print(f"Agent {agent0.agent_id} initial belief Q:\n{np.round(agent0.get_current_belief(), 3)}")

    # --- Simulate a step ---
    # Assume agent0 observes the current positions of all agents
    current_system_positions = initial_positions_np.copy()

    # 1. Agent updates its belief based on current observations
    agent0.update_belief_based_on_observations(current_system_positions)
    print(f"Agent {agent0.agent_id} belief Q after initial observation:\n{np.round(agent0.get_current_belief(), 3)}")

    # 2. Agent decides an action
    vel, head_delta, efe = agent0.decide_action(current_system_positions)
    print(f"Agent {agent0.agent_id} decided action: Vel={vel:.3f}, HeadDelta={head_delta:.3f}, EFE={efe:.3f}")
    print(f"Agent {agent0.agent_id} belief Q after action decision:\n{np.round(agent0.get_current_belief(), 3)}")

    # Simulate applying the action (simplified)
    new_x = agent0.current_state_estimate[0] + vel * np.cos(agent0.current_state_estimate[2] + head_delta) * agent0.params['dt']
    new_y = agent0.current_state_estimate[1] + vel * np.sin(agent0.current_state_estimate[2] + head_delta) * agent0.params['dt']
    new_theta = (agent0.current_state_estimate[2] + head_delta + np.pi) % (2 * np.pi) - np.pi # Wrap to Pi
    
    agent0.update_internal_state_estimate(np.array([new_x, new_y, new_theta]))
    current_system_positions[0] = agent0.current_state_estimate # Update system view for next step

    print(f"Agent {agent0.agent_id} new estimated state: {np.round(agent0.current_state_estimate,3)}")

    # --- Test changing reasoning mode ---
    print("\n--- Changing Agent 0 to First-Order Reasoning ---")
    agent0.set_reasoning_mode('first_order')
    print(f"Agent {agent0.agent_id} reasoning mode: {agent0.reasoning_mode}")
    print(f"Agent {agent0.agent_id} belief Q after mode change:\n{np.round(agent0.get_current_belief(), 3)}") # Should be reset by AIFWrapper re-init

    agent0.update_belief_based_on_observations(current_system_positions) # Observe again
    vel_fo, head_delta_fo, efe_fo = agent0.decide_action(current_system_positions)
    print(f"Agent {agent0.agent_id} (First-Order) decided action: Vel={vel_fo:.3f}, HeadDelta={head_delta_fo:.3f}, EFE={efe_fo:.3f}")
    print(f"Agent {agent0.agent_id} (First-Order) belief Q after action decision:\n{np.round(agent0.get_current_belief(), 3)}")