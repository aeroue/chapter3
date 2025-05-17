# strong_communication/utility_calculator.py

import numpy as np

class UtilityCalculator:
    """
    Calculates the expected utility for a spacecraft joining or being in a coalition
    to perform a task, considering task type uncertainty, risk costs, and fuel costs.
    """

    def __init__(self,
                 task_properties_by_type: dict, # {type_idx: {"revenue": val, "risk": val, "name": "str"}}
                 fuel_cost_calculator: callable, # function: (agent_id, agent_state, task_info) -> fuel_cost
                 n_task_types: int
                ):
        """
        Initializes the UtilityCalculator.

        Args:
            task_properties_by_type (dict): A dictionary where keys are task type indices (0 to K-1)
                                            and values are dictionaries containing "revenue" and "risk"
                                            for that type. E.g., {0: {"revenue": 100, "risk": 5}, ...}.
            fuel_cost_calculator (callable): A function that takes agent_id, current agent_state,
                                             and task_info (e.g., {'id': task_id, 'position': ...})
                                             and returns the estimated fuel cost.
            n_task_types (int): The total number of distinct task types (K).
        """
        if not isinstance(task_properties_by_type, dict):
            raise TypeError("task_properties_by_type must be a dictionary.")
        self.task_properties_by_type = task_properties_by_type
        
        if not callable(fuel_cost_calculator):
            raise TypeError("fuel_cost_calculator must be a callable function.")
        self.fuel_cost_calculator = fuel_cost_calculator
        
        if not isinstance(n_task_types, int) or n_task_types <= 0:
            raise ValueError("n_task_types must be a positive integer.")
        self.n_task_types = n_task_types

        # Pre-extract revenue and risk for faster lookup, with defaults
        self.revenues_by_type_arr = np.zeros(n_task_types)
        self.risks_by_type_arr = np.zeros(n_task_types)
        for i in range(n_task_types):
            props = self.task_properties_by_type.get(i, {})
            self.revenues_by_type_arr[i] = props.get("revenue", 0)
            self.risks_by_type_arr[i] = props.get("risk", 0)


    def calculate_expected_utility(self,
                                   agent_id: int,
                                   agent_state: np.ndarray, # Current state [x,y,vx,vy] for fuel calculation
                                   task_info: dict,         # Min: {'id': task_id (actual, e.g., 1-indexed), 'position': np.array}
                                   agent_belief_for_task: np.ndarray, # Belief distribution [beta_k] for this task. Shape (K,).
                                   num_agents_in_coalition_if_joined: int # Expected size of coalition *if agent joins*.
                                  ) -> float:
        """
        Calculates the expected utility for agent_id if it joins/is in the coalition for task_id.
        This matches the formulation in user's draft (source 1794):
        u_i(C_j) = (Expected_Total_Revenue_for_Task_j / num_agents_in_coalition_if_joined)
                   - Expected_Risk_Cost_for_agent_i_for_Task_j
                   - Fuel_Cost_for_agent_i_to_Task_j

        Args:
            agent_id (int): The ID of the agent.
            agent_state (np.ndarray): Current physical state of the agent [x,y,vx,vy].
            task_info (dict): Dict with task info, must include 'id' and 'position'.
            agent_belief_for_task (np.ndarray): Agent's belief (probabilities) over task types.
                                                Shape (n_task_types,).
            num_agents_in_coalition_if_joined (int): The number of agents that would be in the
                                                     coalition if this agent is part of it. Must be > 0.

        Returns:
            float: The expected utility.
        """
        if not isinstance(agent_belief_for_task, np.ndarray) or \
           agent_belief_for_task.shape != (self.n_task_types,):
            raise ValueError(f"Agent belief must be a numpy array of shape ({self.n_task_types},)")
        if not np.isclose(np.sum(agent_belief_for_task), 1.0):
            # Attempt to normalize if very close, otherwise raise error
            if abs(np.sum(agent_belief_for_task) - 1.0) < 1e-5 :
                agent_belief_for_task = agent_belief_for_task / np.sum(agent_belief_for_task)
            else:
                raise ValueError(f"Agent belief for task {task_info.get('id','unknown')} must sum to 1. "
                                 f"Got: sum={np.sum(agent_belief_for_task)}, belief={agent_belief_for_task}")
        if num_agents_in_coalition_if_joined <= 0:
            # print(f"Warning: num_agents_in_coalition_if_joined is {num_agents_in_coalition_if_joined} for task {task_info.get('id')}. Utility will be -inf.")
            return -np.inf # Or a large negative number, as division by zero/non-positive is undefined.

        # 1. Calculate Expected Total Revenue of the task ( \sum_k beta_k * V(k) )
        #    This is V_tilde_i^j from draft (source 1791), but without division by |C_j| yet.
        expected_total_task_revenue = np.dot(agent_belief_for_task, self.revenues_by_type_arr)

        # Shared expected revenue for this agent
        shared_expected_revenue = expected_total_task_revenue / num_agents_in_coalition_if_joined

        # 2. Calculate Expected Risk Cost for this agent for this task ( \sum_k beta_k * O_i(k) )
        #    This is O_tilde_i^j from draft (source 1791).
        #    Assuming risks_by_type_arr are O_i(k) for this agent_id.
        #    If O_i(k) depends on agent_id, this needs more complex lookup.
        expected_risk_cost = np.dot(agent_belief_for_task, self.risks_by_type_arr)

        # 3. Calculate Fuel Cost S_i(C_j)
        #    The fuel_cost_calculator function is passed during __init__.
        #    It expects task_info to contain at least 'id' and 'position'.
        if 'position' not in task_info or task_info['position'] is None:
            # print(f"Warning: Task {task_info.get('id')} has no position for fuel calculation. Assuming high fuel cost.")
            fuel_cost = np.inf # Or some large penalty
        else:
            fuel_cost = self.fuel_cost_calculator(agent_id, agent_state, task_info)

        # 4. Calculate total expected utility
        total_expected_utility = shared_expected_revenue - expected_risk_cost - fuel_cost

        return total_expected_utility

if __name__ == '__main__':
    # Define parameters for testing
    task_props_by_type_test = {
        0: {"revenue": 100, "risk": 5, "name": "Type 0 (Easy)"},
        1: {"revenue": 200, "risk": 25, "name": "Type 1 (Hard)"}
    }
    num_types_test = 2

    # Dummy fuel calculator for testing
    def dummy_fuel_calc(agent_id, agent_state, task_info_dict, base_cost=1.0, dist_factor=0.1):
        agent_pos = agent_state[:2]
        task_pos = task_info_dict.get('position')
        if task_pos is None: return np.inf
        distance = np.linalg.norm(agent_pos - task_pos)
        return base_cost + distance * dist_factor

    utility_calc_test = UtilityCalculator(
        task_properties_by_type=task_props_by_type_test,
        fuel_cost_calculator=dummy_fuel_calc,
        n_task_types=num_types_test
    )

    print(f"Calculator internal revenues: {utility_calc_test.revenues_by_type_arr}")
    print(f"Calculator internal risks: {utility_calc_test.risks_by_type_arr}")

    # Test case 1: Agent 0 considers Task 1, would be the only member
    agent0_id = 0
    agent0_state = np.array([0.0, 0.0, 0.0, 0.0])
    task1_info_test = {'id': 1, 'position': np.array([10.0, 0.0])}
    agent0_belief_task1 = np.array([0.7, 0.3]) # 70% Type 0, 30% Type 1
    num_agents_if_joined_1 = 1

    utility1 = utility_calc_test.calculate_expected_utility(
        agent0_id, agent0_state, task1_info_test, agent0_belief_task1, num_agents_if_joined_1
    )
    print(f"\nAgent {agent0_id} for Task {task1_info_test['id']} (as 1st member): Expected Utility = {utility1:.2f}")
    # Expected Total Revenue for Task: (0.7 * 100) + (0.3 * 200) = 70 + 60 = 130
    # Shared Revenue: 130 / 1 = 130
    # Expected Risk: (0.7 * 5) + (0.3 * 25) = 3.5 + 7.5 = 11
    # Fuel Cost: 1.0 + (0.1 * 10.0) = 2.0
    # Total Utility: 130 - 11 - 2.0 = 117.0. Correct.

    # Test case 2: Agent 0 considers Task 1, another agent is already there (total 2)
    num_agents_if_joined_2 = 2
    utility2 = utility_calc_test.calculate_expected_utility(
        agent0_id, agent0_state, task1_info_test, agent0_belief_task1, num_agents_if_joined_2
    )
    print(f"Agent {agent0_id} for Task {task1_info_test['id']} (as 2nd member): Expected Utility = {utility2:.2f}")
    # Shared Revenue: 130 / 2 = 65
    # Total Utility: 65 - 11 - 2.0 = 52.0. Correct.

    # Test case 3: Belief sums to nearly 1
    agent0_belief_task1_norm = np.array([0.70001, 0.29998])
    utility3 = utility_calc_test.calculate_expected_utility(
        agent0_id, agent0_state, task1_info_test, agent0_belief_task1_norm, num_agents_if_joined_1
    )
    print(f"Agent {agent0_id} for Task {task1_info_test['id']} (belief near sum 1): Expected Utility = {utility3:.2f}")
    # Should be close to 117.0

    # Test case 4: Belief sums far from 1 (should raise error)
    agent0_belief_task1_bad = np.array([0.6, 0.2])
    try:
        utility4 = utility_calc_test.calculate_expected_utility(
            agent0_id, agent0_state, task1_info_test, agent0_belief_task1_bad, num_agents_if_joined_1
        )
    except ValueError as e:
        print(f"Error for bad belief sum: {e}")