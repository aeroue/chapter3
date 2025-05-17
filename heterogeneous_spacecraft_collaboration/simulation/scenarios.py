# simulation/scenarios.py

import numpy as np
import random
import itertools # 


def get_scenario(scenario_name: str, random_seed: int = None) -> dict:
    """
    Returns a dictionary containing parameters for a predefined simulation scenario.

    Args:
        scenario_name (str): The name of the scenario to load.
        random_seed (int, optional): A random seed for scenarios with random elements.
                                     If None, randomization will be based on current state.

    Returns:
        dict: A dictionary of scenario parameters.
              Keys might include: 'num_agents', 'num_tasks', 'agent_initial_states',
              'task_definitions', 'sim_time_step', 'orbital_n', 'communication_type',
              'k_max_lengths', 'agent_type_properties', 'task_properties_by_type',
              'num_task_types', 'env_extents', 'max_sim_duration', 'comm_graph', etc.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    base_params = {
        "sim_time_step": 1.0,
        "orbital_n": 0.00113,  # rad/s (approx LEO)
        "num_task_types": 2,   # E.g., Type 0 (Low Val), Type 1 (High Val)
        # "task_properties_by_type": { # Ground truth properties
        #     0: {"revenue": 100, "risk": 10, "name": "Type 0 (Low Value)"},
        #     1: {"revenue": 250, "risk": 30, "name": "Type 1 (High Value)"}
        # }
        "task_properties_by_type": {
        0: {"revenue": 100, "risk": 10, "name": "低风险低收益"},
        1: {"revenue": 250, "risk": 30, "name": "高风险高收益"}
        },
        "env_extents": {"x_lim": (-50, 50), "y_lim": (-50, 50)}, # For visualization and initial placement
        "max_sim_duration": 200.0, # Max simulation time in seconds
        "convergence_patience_ksc": 3, # For KSC coalition formation
        "mpc_horizon_N": 10,
        "mpc_Q_diag": [1.0, 1.0, 0.1, 0.1], # MPC state weights [x,y,vx,vy]
        "mpc_R_diag": [0.01, 0.01],       # MPC control weights [ax,ay]
        "mpc_u_max": 0.05,                 # Max thrust for MPC
        "fuel_calc_params": {"fuel_per_unit_distance": 0.05, "fixed_maneuver_cost": 0.5}, # For UtilityCalculator
        # In scenarios.py, within a scenario definition:
        "agent_observation_p_corr": {0: 0.85, 1: 0.9, 2: 0.8}, # agent_id: p_corr
        "aif_params": { # Parameters for ActiveInferenceAgent and AIFWrapper
            "velocity_options": np.array([0.0, 0.2, 0.5, 0.8]), # m/s relative to target for simple AIF vel
            "heading_options": np.array([-np.pi/6, -np.pi/12, 0, np.pi/12, np.pi/6]), # rad, relative heading change
            "observation_error_std": 0.5, # std dev for observing other agents' positions
            "dt": 1.0, # time step for AIF internal prediction (should match sim_time_step generally)
            # reward_configs will be scenario-specific
        }
    }


    if scenario_name == "strong_3agents_2tasks":
        num_agents = 3
        num_tasks = 2
        agent_ids = list(range(num_agents))
        task_ids_actual = list(range(1, num_tasks + 1))

        scenario = {
            **base_params,
            "scenario_name": scenario_name,
            "num_agents": num_agents,
            "num_tasks": num_tasks,
            "agent_ids": agent_ids,
            "task_ids_actual": task_ids_actual,
            "agent_initial_states": { # agent_id: [x, y, vx, vy]
                0: np.array([0, 0, 0.1, 0]),
                1: np.array([5, 5, 0, -0.1]),
                2: np.array([-5, 5, -0.1, -0.1]),
            },
            "task_definitions": [ # List of dicts for Task constructor
                {"task_id": 1, "position": np.array([20, 15]), "true_type": 0},
                {"task_id": 2, "position": np.array([-20, -10]), "true_type": 1},
            ],
            "communication_type": "strong", # "strong", "weak", "mixed"
            "k_max_lengths": {aid: 2 for aid in agent_ids}, # KSC K-value
            "agent_type_properties": { # For heterogeneity if needed
                aid: {"sensor_type": "omniscient_strong", "isp": 300, "initial_mass":100, "dry_mass":80} for aid in agent_ids
            },
            "comm_graph": {0:[1,2], 1:[0,2], 2:[0,1]}, # Fully connected
            "initial_belief_pseudo_counts": 1.0, # For BeliefManager
        }
        # AIF specific params for this scenario if different from base
        scenario["aif_params"]["reward_configs"] = _generate_reward_configs(num_agents, num_tasks, exclusive=True)
        scenario["aif_params"]["max_distance_measure"] = np.max(np.abs(base_params["env_extents"]["x_lim"])) * 2 # approx
        return scenario

    elif scenario_name == "weak_2agents_2tasks_rendezvous":
        num_agents = 2
        num_tasks = 2 # These are potential rendezvous points
        agent_ids = list(range(num_agents))

        scenario = {
            **base_params,
            "scenario_name": scenario_name,
            "num_agents": num_agents,
            "num_tasks": num_tasks, # Here, tasks are potential goals for AIF
            "agent_ids": agent_ids,
            "task_ids_actual": [], # No "tasks" in the KSC sense, goals are for AIF
            "agent_initial_states": {
                0: np.array([-10, 0, 0.0, 0.05]),
                1: np.array([10, 0, 0.0, -0.05]),
            },
            "task_definitions": [], # AIF goals will be defined separately or within aif_params
            "aif_goals": np.array([[0, 15], [0, -15]]), # Two potential rendezvous points
            "communication_type": "weak",
            "agent_type_properties": {
                0: {"sensor_type": "A_dist", "aif_reasoning_mode": "higher_order"},
                1: {"sensor_type": "B_angle", "aif_reasoning_mode": "higher_order"},
            },
            "comm_graph": {0:[], 1:[]}, # No communication
        }
        # AIF reward_configs for rendezvous: agents go to the same goal
        # e.g., config (0,0) means agent0->goal0, agent1->goal0
        scenario["aif_params"]["reward_configs"] = [(g,g) for g in range(num_tasks)]
        scenario["aif_params"]["max_distance_measure"] = np.max(np.abs(base_params["env_extents"]["x_lim"])) * 2
        scenario["aif_params"]["agent_types_for_aif"] = [p["sensor_type"] for p in scenario["agent_type_properties"].values()] # For AIFWrapper
        return scenario

    elif scenario_name == "weak_3agents_3tasks_exclusive":
        num_agents = 3
        num_tasks = 3 # These are tasks/goals for AIF
        agent_ids = list(range(num_agents))

        scenario = {
            **base_params,
            "scenario_name": scenario_name,
            "num_agents": num_agents,
            "num_tasks": num_tasks, # Tasks/goals for AIF
            "agent_ids": agent_ids,
            "task_ids_actual": [],
            "agent_initial_states": {
                0: np.array([-15, 10, 0.0, 0.0]),
                1: np.array([0, -15, 0.0, 0.0]),
                2: np.array([15, 10, 0.0, 0.0]),
            },
            "task_definitions": [],
            "aif_goals": np.array([[0,0], [-10, 10], [10,10]]),
            "communication_type": "weak",
            "agent_type_properties": {
                0: {"sensor_type": "A_dist", "aif_reasoning_mode": "higher_order"},
                1: {"sensor_type": "A_dist", "aif_reasoning_mode": "higher_order"},
                2: {"sensor_type": "B_angle", "aif_reasoning_mode": "first_order"}, # Test mixed reasoning
            },
            "comm_graph": {0:[], 1:[], 2:[]},
        }
        scenario["aif_params"]["reward_configs"] = _generate_reward_configs(num_agents, num_tasks, exclusive=True)
        scenario["aif_params"]["max_distance_measure"] = np.max(np.abs(base_params["env_extents"]["x_lim"])) * 2
        scenario["aif_params"]["agent_types_for_aif"] = [p["sensor_type"] for p in scenario["agent_type_properties"].values()]
        return scenario

    # Add more scenarios here:
    # elif scenario_name == "mixed_5agents_3tasks":
    #   ...

    else:
        raise ValueError(f"Unknown scenario name: {scenario_name}")

def _generate_reward_configs(num_agents, num_goals, exclusive=False):
    """
    Helper to generate reward_configs for AIF.
    If exclusive, agents must go to different goals (permutations).
    If not exclusive (e.g. rendezvous), all agents can go to any goal (product).
    """
    if exclusive:
        if num_agents > num_goals:
            # Cannot assign exclusively if more agents than goals
            # Fallback: generate assignments where some agents might be unassigned (not handled here)
            # Or, allow multiple agents per goal but try to maximize distinctness.
            # For simplicity, if exclusive and N_agents > N_goals, this will be problematic for permutations.
            # This should be handled by the problem setup (e.g. more goals or non-exclusive tasks).
            # print(f"Warning: Cannot generate exclusive reward_configs for {num_agents} agents and {num_goals} goals.")
            # Default to product allowing shared goals if permutation is not possible
            return list(itertools.product(range(num_goals), repeat=num_agents))

        configs = list(itertools.permutations(range(num_goals), num_agents))
    else: # Non-exclusive, e.g. rendezvous (all agents to same goal is one type of non-exclusive)
          # Or general non-exclusive: each agent can choose any task independently
        configs = list(itertools.product(range(num_goals), repeat=num_agents))
    return configs


if __name__ == "__main__":
    print("Available scenarios can be requested via get_scenario(scenario_name).")

    try:
        scen1_params = get_scenario("strong_3agents_2tasks", random_seed=42)
        print(f"\n--- Scenario: {scen1_params['scenario_name']} ---")
        # print(scen1_params)
        print(f"  Num Agents: {scen1_params['num_agents']}, Num Tasks: {scen1_params['num_tasks']}")
        print(f"  Comm Type: {scen1_params['communication_type']}")
        print(f"  Agent 0 initial state: {scen1_params['agent_initial_states'][0]}")
        print(f"  Task 1 def: {scen1_params['task_definitions'][0]}")
        print(f"  AIF reward_configs for strong (example): {scen1_params['aif_params']['reward_configs']}")


        scen2_params = get_scenario("weak_2agents_2tasks_rendezvous", random_seed=123)
        print(f"\n--- Scenario: {scen2_params['scenario_name']} ---")
        print(f"  Num Agents: {scen2_params['num_agents']}")
        print(f"  Comm Type: {scen2_params['communication_type']}")
        print(f"  AIF Goals: \n{scen2_params['aif_goals']}")
        print(f"  Agent 0 properties: {scen2_params['agent_type_properties'][0]}")
        print(f"  AIF reward_configs for weak rendezvous: {scen2_params['aif_params']['reward_configs']}")

        scen3_params = get_scenario("weak_3agents_3tasks_exclusive", random_seed=10)
        print(f"\n--- Scenario: {scen3_params['scenario_name']} ---")
        print(f"  Num Agents: {scen3_params['num_agents']}")
        print(f"  AIF Goals: \n{scen3_params['aif_goals']}")
        print(f"  Agent 2 properties: {scen3_params['agent_type_properties'][2]}")
        print(f"  AIF reward_configs for weak exclusive: {scen3_params['aif_params']['reward_configs']}")


    except ValueError as e:
        print(f"Error getting scenario: {e}")