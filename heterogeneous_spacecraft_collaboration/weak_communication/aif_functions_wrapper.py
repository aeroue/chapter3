# weak_communication/aif_functions_wrapper.py

import numpy as np
import copy
import itertools
from . import aif_functions_isobeliefs_convergent as aif_core
# Attempt to import the core AIF functions.
# This assumes 'aif_functions_isobeliefs_convergent.py' is either in the same directory,
# or in a directory added to sys.path, or installed as a module.
# For cleaner project structure, it might be good to place it within weak_communication
# and rename it to, e.g., aif_core_logic.py

# # weak_communication/aif_functions_wrapper.py
# import os
# import sys
# print(f"Current working directory: {os.getcwd()}")
# print(f"Sys.path: {sys.path}")
# print(f"Contents of weak_communication directory: {os.listdir(os.path.dirname(__file__))}")
class AIFWrapper:
    """
    Wrapper for aif_functions_isobeliefs_convergent.py to provide a structured
    interface for active inference, supporting different orders of reasoning.
    """

    def __init__(self, agent_params: dict):
        """
        Initializes the AIF Wrapper with agent-specific parameters.

        Args:
            agent_params (dict): A dictionary containing parameters for the agent,
                                 similar to what's used in main.py and parsed by
                                 aif_core.parse_args_by_agent. Expected keys include:
                                 'agent_id', 'num_agents', 'agent_types', 'goals',
                                 'velocity_options', 'heading_options', 'num_actions',
                                 'observation_error_std', 'reward_configs', 'dt',
                                 'max_distance_measure', etc.
        """
        self.agent_params = agent_params
        self.agent_id = agent_params['agent_id']
        self.num_agents = agent_params['num_agents']
        self.agent_types = agent_params['agent_types'] # List of types for all agents
        self.goals = np.array(agent_params['goals'])
        self.reward_configs = agent_params['reward_configs']

        # Ensure prior is correctly shaped for different reasoning levels
        # The core `get_likelihood` in `aif_core` expects `prior` to be
        # 1D for non-EP and 2D (num_agents, num_reward_configs) for EP.
        # We'll manage the initial prior state here or expect it to be managed externally.
        self.current_belief_q = self._initialize_belief()


    def _initialize_belief(self) -> np.ndarray:
        """
        Initializes the belief (posterior q) based on the reasoning mode.
        """
        num_reward_configs = len(self.agent_params['reward_configs'])
        reasoning_mode = self.agent_params.get('reasoning_mode', 'higher_order') # 'higher_order', 'first_order', 'zero_order'

        if reasoning_mode == 'higher_order': # Corresponds to use_ep=True
            # For higher-order, belief is often per agent about the system config
            # aif_core.get_likelihood with use_ep=True returns a (num_agents, num_reward_configs) array
            # So, the prior/belief Q should also be of this shape or be broadcastable.
            # The initial prior in aif_core.parse_args_by_agent for use_ep=True is:
            # np.tile(1/len(args['reward_configs']), (len(args['agent_positions']), len(args['reward_configs'])))
            initial_q = np.tile(1.0 / num_reward_configs, (self.num_agents, num_reward_configs))
        else: # First-order or Zero-order (use_ep=False in aif_core)
            # aif_core.get_likelihood with use_ep=False returns a 1D array
            initial_q = np.full(num_reward_configs, 1.0 / num_reward_configs)
        return initial_q


    def update_belief_q(self, new_q: np.ndarray):
        """Updates the agent's current belief Q."""
        self.current_belief_q = new_q


    def get_likelihood_for_system_config(self,
                                         observed_agent_positions: np.ndarray, # All agent positions [N_agents, 3] (x,y,theta)
                                         reasoning_mode: str = 'higher_order'  # 'higher_order', 'first_order'
                                        ) -> np.ndarray:
        """
        Calculates the likelihood of system configurations (P(o|s)) using aif_core.get_likelihood.
        Manages the `use_ep` flag based on `reasoning_mode`.

        Args:
            observed_agent_positions (np.ndarray): Array of current observed positions
                                                  for all agents, shape (num_agents, 3).
            reasoning_mode (str): 'higher_order' (for use_ep=True in aif_core) or
                                  'first_order' (for use_ep=False in aif_core).

        Returns:
            np.ndarray: Posterior probability distribution over reward_configs.
                        Shape depends on reasoning_mode (1D or 2D for higher_order).
        """
        use_ep_flag = True if reasoning_mode == 'higher_order' else False
        # The 'observations' structure for aif_core.get_likelihood needs to be
        # a list of dicts, where each dict has 'position', 'type'.
        observations_for_aif = []
        for i in range(self.num_agents):
            sim_type = self.agent_types[i]
            if i == self.agent_id: # Self-observation, typically handled as 's' type in aif_core
                sim_type = 's' # or agent_params['agent_types'][self.agent_id] if 's' is for sim only

            observations_for_aif.append(
                aif_core.simulate_observation(
                    true_position=observed_agent_positions[i],
                    observation_error_std=0 if i == self.agent_id else self.agent_params['observation_error_std'],
                    sim_type=sim_type, # 's' for self, actual type for others for evidence calc
                    observed_agent=i
                )
            )
        
        # The prior passed to get_likelihood is the current belief Q
        # `aif_core.get_likelihood` then calculates P(o|s) * P(s) and normalizes
        posterior_q = aif_core.get_likelihood(
            robot_id=self.agent_id,
            observations=observations_for_aif,
            goals=self.goals,
            agent_vars=self.agent_params, # This needs all necessary fields like 'reward_configs', 'agent_types'
            prior=self.current_belief_q,    # Pass current belief Q as prior
            use_ep=use_ep_flag,
            consensus=use_ep_flag # Typically consensus is also used with EP
        )
        self.update_belief_q(posterior_q) # Update internal belief
        return posterior_q

    def choose_action_to_minimize_efe(self,
                                      current_agent_positions: np.ndarray, # All agent positions
                                      reasoning_mode: str = 'higher_order'
                                     ) -> tuple[float, float, float, np.ndarray]:
        """
        Chooses an action (velocity, heading change) by minimizing Expected Free Energy (EFE),
        using the choice_heuristic from aif_core.

        Args:
            current_agent_positions (np.ndarray): Current positions of all agents.
            reasoning_mode (str): 'higher_order', 'first_order', or 'zero_order' (greedy).

        Returns:
            tuple: (best_velocity, best_heading_change, min_efe_score, updated_belief_q)
        """
        use_ep_flag = True if reasoning_mode == 'higher_order' else False
        greedy_flag = True if reasoning_mode == 'zero_order' else False
        
        # The choice_heuristic in aif_core needs a list of observation dicts
        observations_for_aif = []
        for i in range(self.num_agents):
            sim_type = self.agent_types[i]
            # if i == self.agent_id: sim_type = 's' # Self, for evidence calc

            observations_for_aif.append(
                aif_core.simulate_observation(
                    true_position=current_agent_positions[i],
                    # Error std for observations passed to choice_heuristic should reflect true sensing
                    observation_error_std=self.agent_params['observation_error_std'],
                    sim_type=sim_type,
                    observed_agent=i
                )
            )

        # Override agent_params for this specific call if greedy is needed
        temp_agent_params = self.agent_params.copy()
        temp_agent_params['greedy'] = greedy_flag
        # Ensure 'use_ep' in temp_agent_params reflects the current reasoning_mode for choice_heuristic
        temp_agent_params['use_ep'] = use_ep_flag


        best_velocity, best_heading_change, min_efe_score = aif_core.choice_heuristic(
            current_positions=current_agent_positions,
            observations=observations_for_aif, # Correctly formatted observations
            prior=self.current_belief_q,          # Pass current belief Q as prior
            agent_params=temp_agent_params,     # Pass full agent_params
            use_ep=use_ep_flag,                 # Control epistemic reasoning
            consensus=use_ep_flag               # Control consensus mechanism
        )
        
        # The choice_heuristic internally calls get_likelihood, which updates Q.
        # To get the Q *after* this decision process (i.e., the posterior used for EFE calc),
        # we might need get_likelihood to return it or re-calculate it based on the chosen action.
        # For now, assume self.current_belief_q is updated by the last call to get_likelihood inside choice_heuristic.
        # This depends on aif_core.choice_heuristic's internal calls updating the prior it receives.
        # More robustly, choice_heuristic should return the posterior it used for EFE calc,
        # or we simulate one step with the chosen action and then update belief.

        # Let's simulate one step with the chosen action and get the resulting posterior
        # to ensure self.current_belief_q is the belief *after* the action consideration.
        
        # 1. Predict this agent's next position based on chosen action
        my_predicted_pos = aif_core.predict_agent_position(
            agent_position=current_agent_positions[self.agent_id],
            velocity=best_velocity,
            heading=best_heading_change, # This is a delta heading in aif_core
            dt=self.agent_params['dt']
        )
        
        # 2. Create new set of all agent positions for likelihood calculation
        next_all_agent_positions = current_agent_positions.copy()
        next_all_agent_positions[self.agent_id] = my_predicted_pos
        
        # 3. Get likelihood/posterior based on this predicted state
        # This is the posterior Q(G | a*)
        final_posterior_q_for_action = self.get_likelihood_for_system_config(
            observed_agent_positions=next_all_agent_positions,
            reasoning_mode=reasoning_mode
        )
        # self.current_belief_q is updated inside get_likelihood_for_system_config

        return best_velocity, best_heading_change, min_efe_score, self.current_belief_q


    # --- Passthrough or wrapped utility functions from aif_core ---
    def softmax(self, x: np.ndarray) -> np.ndarray:
        return aif_core.softmax(x)

    def calculate_kl_divergence(self, q_distribution: np.ndarray) -> float:
        """Calculates KL divergence of q w.r.t. a preference (typically an ideal/target distribution).
           The aif_core.calculate_kl_divergence seems to take only 'q' and assumes 'p' is deterministic max.
        """
        return aif_core.calculate_kl_divergence(q_distribution)

    def calculate_shannon_entropy(self, p_distribution: np.ndarray) -> float:
        return aif_core.calculate_shannon_entropy(p_distribution)

    def get_expected_free_energy(self, posterior_q_for_action: np.ndarray,
                                 # Optional: add parameters if EFE calculation needs more context
                                ) -> float:
        """
        Calculates the Expected Free Energy (EFE) for a given posterior belief Q(G|action)
        EFE = Expected_KL_Divergence (Pragmatic Value) + Expected_Entropy (Epistemic Value)
        The aif_core.choice_heuristic already calculates this. This is for external use if needed.
        Your draft (source 1800) defines G(a_i(t)) which is the EFE.
        It has D_KL[Q(s|a,pi)||P(s|C)] - E_Q[ln P(o|s)].
        The aif_core.choice_heuristic calculates free_energies = entropies + kl_divergences.
        This seems to be: H[Q(s|a,pi)] + D_KL[Q(s|a,pi)||P_desired(s)].
        Let's assume the EFE score from choice_heuristic is what we need.
        If a direct EFE calculation is needed based on your draft's formula:
         pragmatic_value = self.calculate_kl_divergence(posterior_q_for_action) # Measures deviation from preferred dist
         epistemic_value = self.calculate_shannon_entropy(posterior_q_for_action) # Measures uncertainty reduction
        return pragmatic_value + epistemic_value 
        """
        # This function might be redundant if choice_heuristic already returns the EFE score.
        # The 'min_efe_score' from choose_action_to_minimize_efe is the EFE value.
        kl_div = self.calculate_kl_divergence(posterior_q_for_action)
        entropy = self.calculate_shannon_entropy(posterior_q_for_action)
        # Note: The exact EFE formulation can vary.
        # aif_core seems to use H[Q] + KL[Q || P_desired] where P_desired is one-hot max of Q.
        # Your draft's formulation is D_KL[Q(s|a)||P(s|C)] - E_Q[ln P(o|s)]
        # The first term is risk/divergence from goal, second is ambiguity/expected surprise.
        # For now, let's use the sum of entropy and KL as a common simplified EFE proxy.
        return entropy + kl_div


if __name__ == '__main__':
    print("AIF Wrapper Example")

    # Example agent_params (must match structure expected by aif_core functions)
    num_agents_test = 2
    num_goals_test = 2
    env_size_test = 30
    goals_test = np.random.uniform(1, env_size_test - 1, size=(num_goals_test, 2))
    agent_positions_test = np.random.uniform(0, env_size_test, size=(num_agents_test, 3)) # x,y,theta
    agent_types_test = np.random.choice(['A', 'B'], num_agents_test)

    # Define reward_configs (goal assignments for the system)
    # For 2 agents, 2 goals, exclusive assignment: (0,1) or (1,0)
    # For rendezvous, reward_configs would be e.g. [(0,0), (1,1)]
    example_reward_configs = list(itertools.permutations(range(num_goals_test), num_agents_test))
    if not example_reward_configs and num_agents_test > 0 and num_goals_test > 0: # e.g. 1 agent, 2 goals
        example_reward_configs = [(g,) for g in range(num_goals_test)]
    elif num_agents_test == 0 :
         example_reward_configs = []


    base_params_for_agent0 = {
        'agent_id': 0,
        'num_agents': num_agents_test,
        'agent_types': agent_types_test,
        'goals': goals_test,
        'agent_positions': agent_positions_test, # This is used by aif_core for initial state if needed
        'velocity_options': np.linspace(0.0, 1.0, 4),
        'heading_options': np.linspace(-np.pi / 4, np.pi / 4, 8),
        'num_actions': 4 * 8,
        'observation_error_std': 0.1, # Smaller error for testing
        'reward_configs': example_reward_configs,
        'dt': 1.0,
        'max_distance_measure': env_size_test + 1,
        'env_size': env_size_test,
        # 'prior': np.ones(len(example_reward_configs)) / len(example_reward_configs) # Will be handled by wrapper
    }

    print(f"Test goals: {goals_test}")
    print(f"Test agent types: {agent_types_test}")
    print(f"Test reward_configs: {example_reward_configs}")


    if not example_reward_configs:
        print("Skipping test due to empty reward_configs (likely num_agents > num_goals for permutations).")
    else:
        # --- Test Higher-Order Reasoning ---
        print("\n--- Testing Higher-Order Reasoning ---")
        params_agent0_ep = base_params_for_agent0.copy()
        params_agent0_ep['reasoning_mode'] = 'higher_order' # This will set use_ep=True internally for core calls

        aif_wrapper_ep = AIFWrapper(params_agent0_ep)
        print(f"Initial Belief Q (Agent 0 for EP): {aif_wrapper_ep.current_belief_q.shape}\n{np.round(aif_wrapper_ep.current_belief_q,3)}")

        # Simulate getting likelihood
        current_poses = agent_positions_test.copy()
        likelihood_ep = aif_wrapper_ep.get_likelihood_for_system_config(current_poses, reasoning_mode='higher_order')
        print(f"Likelihood/Posterior Q (EP) after obs: {likelihood_ep.shape}\n{np.round(likelihood_ep,3)}")

        # Simulate choosing action
        vel_ep, head_ep, efe_ep, q_after_action_ep = aif_wrapper_ep.choose_action_to_minimize_efe(current_poses, reasoning_mode='higher_order')
        print(f"Action (EP): Vel={vel_ep:.2f}, HeadChg={head_ep:.2f}, EFE={efe_ep:.3f}")
        print(f"Belief Q after action (EP): {q_after_action_ep.shape}\n{np.round(q_after_action_ep,3)}")

        # --- Test First-Order Reasoning ---
        print("\n--- Testing First-Order Reasoning ---")
        params_agent0_first_order = base_params_for_agent0.copy()
        params_agent0_first_order['reasoning_mode'] = 'first_order'

        aif_wrapper_first = AIFWrapper(params_agent0_first_order)
        print(f"Initial Belief Q (Agent 0 for First-Order): {aif_wrapper_first.current_belief_q.shape}\n{np.round(aif_wrapper_first.current_belief_q,3)}")
        
        likelihood_first = aif_wrapper_first.get_likelihood_for_system_config(current_poses, reasoning_mode='first_order')
        print(f"Likelihood/Posterior Q (First-Order) after obs: {likelihood_first.shape}\n{np.round(likelihood_first,3)}")

        vel_first, head_first, efe_first, q_after_action_first = aif_wrapper_first.choose_action_to_minimize_efe(current_poses, reasoning_mode='first_order')
        print(f"Action (First-Order): Vel={vel_first:.2f}, HeadChg={head_first:.2f}, EFE={efe_first:.3f}")
        print(f"Belief Q after action (First-Order): {q_after_action_first.shape}\n{np.round(q_after_action_first,3)}")

        # --- Test Zero-Order (Greedy) Reasoning ---
        print("\n--- Testing Zero-Order (Greedy) Reasoning ---")
        params_agent0_zero_order = base_params_for_agent0.copy()
        params_agent0_zero_order['reasoning_mode'] = 'zero_order'

        aif_wrapper_zero = AIFWrapper(params_agent0_zero_order)
        print(f"Initial Belief Q (Agent 0 for Zero-Order): {aif_wrapper_zero.current_belief_q.shape}\n{np.round(aif_wrapper_zero.current_belief_q,3)}")

        # For greedy, get_likelihood might not be explicitly called before action choice by external logic
        # but choice_heuristic will call it if not set to greedy internally.
        # The wrapper's choose_action sets the greedy flag in agent_params for the core call.
        vel_zero, head_zero, efe_zero, q_after_action_zero = aif_wrapper_zero.choose_action_to_minimize_efe(current_poses, reasoning_mode='zero_order')
        print(f"Action (Zero-Order): Vel={vel_zero:.2f}, HeadChg={head_zero:.2f}, Score (might not be EFE for pure greedy)={efe_zero:.3f}")
        print(f"Belief Q after action (Zero-Order): {q_after_action_zero.shape}\n{np.round(q_after_action_zero,3)}")