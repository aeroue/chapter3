# simulation/run_simulation.py

import numpy as np
import time
import random
import copy
import matplotlib.pyplot as plt
import os
import json # For saving history data

# ... (其他导入与之前版本相同) ...
try:
    from common.dynamics import CWDynamics
    from common.mpc_controller import MPCController
    from strong_communication.utility_calculator import UtilityCalculator
    from strong_communication.belief_manager import BeliefManager as StrongCommsBeliefManager
    from strong_communication.ksc_coalition_formation import KSCCoalitionFormation
    from weak_communication.active_inference import ActiveInferenceAgent
    from simulation.environment import SimulationEnvironment, SpacecraftAgent, Task
    from simulation.scenarios import get_scenario
    from simulation.visualizer import SimulationVisualizer
except ImportError as e:
    print(f"Error importing modules in run_simulation.py: {e}")
    exit(1)

env_instance_for_fuel_calc = None

def get_fuel_calculator_with_env_access(params_dict):
    try:
        from strong_communication.utility_calculator import example_fuel_calculator as base_fuel_calc
    except ImportError:
        def base_fuel_calc(agent_id, agent_state, task_info, fuel_per_unit_distance, fixed_maneuver_cost):
            agent_pos = agent_state[:2] if agent_state is not None else np.array([0,0])
            task_pos = task_info.get('position', agent_pos)
            distance = np.linalg.norm(agent_pos - task_pos) if task_pos is not None else 0
            return fuel_per_unit_distance * distance + fixed_maneuver_cost
    fuel_params = params_dict.get("fuel_calc_params", {})
    fuel_per_dist = fuel_params.get("fuel_per_unit_distance", 0.05)
    fixed_cost = fuel_params.get("fixed_maneuver_cost", 0.5)
    def fuel_calculator_for_utility(agent_id_fc, agent_state_fc, task_info_fc_minimal):
        task_id_lookup = task_info_fc_minimal.get('id')
        task_position_fc = agent_state_fc[:2] if agent_state_fc is not None else np.array([0,0])
        if env_instance_for_fuel_calc and task_id_lookup != 0 and \
           env_instance_for_fuel_calc.tasks and task_id_lookup in env_instance_for_fuel_calc.tasks:
            task_position_fc = env_instance_for_fuel_calc.tasks[task_id_lookup].position
        enriched_task_info = {'id': task_id_lookup, 'position': task_position_fc}
        return base_fuel_calc(agent_id_fc, agent_state_fc, enriched_task_info, fuel_per_dist, fixed_cost)
    return fuel_calculator_for_utility

def run_simulation(scenario_name: str, random_seed: int = None, visualize: bool = True, animation_filename_prefix: str = None):
    global env_instance_for_fuel_calc
    print(f"Starting simulation for scenario: {scenario_name} with seed: {random_seed}")
    start_time_total = time.time()

    try:
        params = get_scenario(scenario_name, random_seed)
    except ValueError as e:
        print(f"Error loading scenario '{scenario_name}': {e}")
        return None

    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    env = SimulationEnvironment(sim_time_step=params["sim_time_step"])
    env_instance_for_fuel_calc = env
    shared_dynamics_model = CWDynamics(n=params["orbital_n"], ts=params["sim_time_step"])

    for agent_id in params["agent_ids"]:
        initial_state = np.array(params["agent_initial_states"][agent_id])
        agent_props = params["agent_type_properties"].get(agent_id, {})
        k_max_val = params.get("k_max_lengths", {}).get(agent_id, 1)
        agent = SpacecraftAgent(agent_id, initial_state, shared_dynamics_model, agent_props, k_max_val)
        env.add_agent(agent)

    for task_def in params["task_definitions"]:
        task_props_from_type = params["task_properties_by_type"].get(task_def["true_type"], {})
        final_task_props = {**task_props_from_type, **task_def.get("properties", {})}
        task = Task(task_def["task_id"], np.array(task_def["position"]), task_def["true_type"], final_task_props)
        env.add_task(task)

    env.set_communication_graph(params.get("comm_graph", {}))
    print(f"Environment initialized with {len(env.agents)} agents and {len(env.tasks)} tasks.")

    mpc_controllers = {}
    mpc_u_min = np.array([-params["mpc_u_max"]] * shared_dynamics_model.control_dim)
    mpc_u_max = np.array([params["mpc_u_max"]] * shared_dynamics_model.control_dim)
    Ad, Bd = shared_dynamics_model.get_discrete_matrices()
    for agent_id in params["agent_ids"]:
        mpc_controllers[agent_id] = MPCController(
            Ad, Bd, np.diag(params["mpc_Q_diag"]), np.diag(params["mpc_R_diag"]),
            params["mpc_horizon_N"], mpc_u_min, mpc_u_max
        )

    strong_comms_belief_mgr = None
    ksc_coalition_former = None
    if params["communication_type"] == "strong":
        print("Initializing modules for STRONG communication.")
        num_actual_tasks_for_bm = len(params.get("task_ids_actual", []))
        strong_comms_belief_mgr = StrongCommsBeliefManager(
            params["num_agents"], num_actual_tasks_for_bm, params["num_task_types"],
            params.get("initial_belief_pseudo_counts", 1.0)
        )
        task_revenues_for_calc = {type_idx: props.get("revenue", 0) for type_idx, props in params["task_properties_by_type"].items() if isinstance(props, dict)}
        risk_costs_for_calc = {type_idx: props.get("risk", 0) for type_idx, props in params["task_properties_by_type"].items() if isinstance(props, dict)}
        current_fuel_calculator = get_fuel_calculator_with_env_access(params)
        utility_calc_strong = UtilityCalculator(task_revenues_for_calc, risk_costs_for_calc, current_fuel_calculator, params["num_task_types"])
        ksc_coalition_former = KSCCoalitionFormation(
            params["agent_ids"], params.get("task_ids_actual", []), utility_calc_strong,
            strong_comms_belief_mgr, params.get("k_max_lengths",{aid:1 for aid in params["agent_ids"]}),
            env.communication_graph, params.get("ksc_max_agents_in_chain", 2)
        )

    active_inference_agents = {}
    if params["communication_type"] == "weak":
        # (AIF initialization - no changes from previous version needed here for this request)
        print("Initializing modules for WEAK communication (Active Inference).")
        base_aif_params = params["aif_params"].copy()
        base_aif_params['goals'] = np.array(params.get("aif_goals", []))
        base_aif_params['reward_configs'] = params["aif_params"].get("reward_configs", [])
        base_aif_params['num_agents'] = params["num_agents"]
        sorted_agent_ids_init_aif = sorted(params["agent_ids"])
        initial_agent_states_for_aif_wrapper = []
        for aid_init in sorted_agent_ids_init_aif:
            full_state = np.array(params["agent_initial_states"][aid_init])
            vel_norm = np.linalg.norm(full_state[2:])
            theta = np.arctan2(full_state[3], full_state[2]) if vel_norm > 1e-4 else 0.0
            initial_agent_states_for_aif_wrapper.append([full_state[0], full_state[1], theta])
        for agent_id in params["agent_ids"]:
            agent_props = params["agent_type_properties"].get(agent_id, {})
            aif_agent_specific_params = base_aif_params.copy()
            aif_agent_specific_params['agent_id'] = agent_id
            aif_agent_specific_params['agent_positions'] = np.array(initial_agent_states_for_aif_wrapper)
            aif_agent_specific_params['agent_types'] = params["aif_params"].get("agent_types_for_aif", [props.get("sensor_type","A_dist") for props in params["agent_type_properties"].values()])
            aif_agent_specific_params['reasoning_mode'] = agent_props.get("aif_reasoning_mode", "higher_order")
            active_inference_agents[agent_id] = ActiveInferenceAgent(agent_id, aif_agent_specific_params, aif_agent_specific_params['agent_types'])


    viz = None
    if visualize:
        print("Initializing visualizer...")
        viz_title_prefix = f"{params.get('scenario_name', 'Simulation')} (Seed: {random_seed}): "
        viz = SimulationVisualizer(env, params["num_task_types"], params["task_properties_by_type"], # Pass full props
                                   params["env_extents"]["x_lim"], params["env_extents"]["y_lim"],
                                   main_title_prefix=viz_title_prefix)
        if params.get("task_ids_actual") and params.get("agent_ids") and len(params["agent_ids"]) > 0 and len(params["task_ids_actual"]) > 0:
            agent_for_belief_plot = params["agent_ids"][0]
            task_for_belief_plot_id = params["task_ids_actual"][0]
            if task_for_belief_plot_id in env.tasks:
                viz.setup_dynamic_belief_plot(agent_id_to_show=agent_for_belief_plot, task_id_to_show=task_for_belief_plot_id)
        
        viz.record_state(agent_controls={}, strong_belief_manager=strong_comms_belief_mgr, aif_agents_weak=active_inference_agents)


    print("\n--- Starting Simulation Loop ---")
    max_steps = int(params["max_sim_duration"] / params["sim_time_step"])
    decision_frequency = params.get("decision_update_frequency_steps", 5)
    ksc_form_max_iters = params.get("ksc_formation_max_iters", 10)
    ksc_form_patience = params.get("ksc_convergence_patience_ksc", 2)

    current_coalition_structure_strong = {}
    if params["communication_type"] == "strong" and ksc_coalition_former and params["agent_ids"]:
         current_coalition_structure_strong = ksc_coalition_former.local_coalition_structures[params["agent_ids"][0]]

    mpc_initial_u_guesses = { ag_id: np.zeros((params["mpc_horizon_N"], shared_dynamics_model.control_dim)) for ag_id in params["agent_ids"]}

    for step in range(max_steps):
        current_time_sec = step * params["sim_time_step"]
        if step % 10 == 0 or step == max_steps - 1:
            print(f"\rSim Step: {step+1}/{max_steps} (Time: {current_time_sec:.1f}s)            ", end="")

        agent_target_states_for_mpc = {}
        current_physical_states_for_decision = env.get_agent_states_for_utility_calc() # For KSC

        if step % decision_frequency == 0:
            if params["communication_type"] == "strong":
                if ksc_coalition_former:
                    # --- Belief Update Logic for Strong Comms ---
                    if strong_comms_belief_mgr and step > 0: # Avoid update at step 0
                        num_bm_tasks = strong_comms_belief_mgr.num_tasks
                        num_bm_types = strong_comms_belief_mgr.num_task_types
                        aggregated_obs = np.zeros((num_bm_tasks, num_bm_types))
                        p_corr_map = params.get("agent_observation_p_corr", {})

                        for ag_id_obs, agent_obj_obs in env.agents.items():
                            assigned_task_id = agent_obj_obs.assigned_task_id
                            p_corr = p_corr_map.get(ag_id_obs, 0.8) # Default p_corr

                            if assigned_task_id != 0 and assigned_task_id in env.tasks:
                                task_obj = env.tasks[assigned_task_id]
                                true_type_idx = task_obj.true_type
                                task_idx_for_bm_obs = assigned_task_id - 1 # Convert to 0-indexed

                                if 0 <= task_idx_for_bm_obs < num_bm_tasks:
                                    observed_type = true_type_idx # Default to true
                                    if random.random() > p_corr : # Mis-observation
                                        possible_false_types = [t for t in range(num_bm_types) if t != true_type_idx]
                                        if possible_false_types: observed_type = random.choice(possible_false_types)
                                    
                                    if 0 <= observed_type < num_bm_types:
                                        aggregated_obs[task_idx_for_bm_obs, observed_type] += 1
                        
                        if np.sum(aggregated_obs) > 0:
                            strong_comms_belief_mgr.update_beliefs_from_aggregated_observations(aggregated_obs)
                            # print(f"\nStrong comms beliefs updated at step {step+1}")
                    # --- End Belief Update ---

                    current_coalition_structure_strong = ksc_coalition_former.form_coalitions(
                        current_physical_states_for_decision, ksc_form_max_iters, ksc_form_patience
                    )
                    for task_id_assigned, agent_ids_in_task in current_coalition_structure_strong.items():
                        for ag_id_ksc in agent_ids_in_task:
                            env.agents[ag_id_ksc].assigned_task_id = task_id_assigned
                    
                    for ag_id_mpc, agent_obj_mpc in env.agents.items():
                        assigned_tid = agent_obj_mpc.assigned_task_id
                        if assigned_tid != 0 and assigned_tid in env.tasks:
                            task_pos = env.tasks[assigned_tid].position
                            agent_target_states_for_mpc[ag_id_mpc] = np.array([task_pos[0], task_pos[1], 0.0, 0.0])
                        else:
                            agent_target_states_for_mpc[ag_id_mpc] = agent_obj_mpc.state.copy()

            elif params["communication_type"] == "weak":
                # (AIF logic as before)
                current_all_aif_positions_obs = env.get_all_agent_positions_for_aif()
                sorted_agent_ids_aif = sorted(list(env.agents.keys()))
                for agent_id_actual in sorted_agent_ids_aif:
                    if agent_id_actual not in active_inference_agents: continue
                    aif_agent = active_inference_agents[agent_id_actual]
                    aif_agent.update_belief_based_on_observations(current_all_aif_positions_obs)
                    aif_vel, aif_head_delta, _ = aif_agent.decide_action(current_all_aif_positions_obs)
                    current_physical_state_agent = env.agents[agent_id_actual].state
                    projected_target = current_physical_state_agent.copy()
                    agent_idx_in_aif_array = sorted_agent_ids_aif.index(agent_id_actual) # Get index for current_all_aif_positions_obs
                    current_agent_heading = current_all_aif_positions_obs[agent_idx_in_aif_array][2]
                    target_heading = (current_agent_heading + aif_head_delta + np.pi) % (2 * np.pi) - np.pi
                    projected_target[0] += aif_vel * np.cos(target_heading) * params["sim_time_step"]
                    projected_target[1] += aif_vel * np.sin(target_heading) * params["sim_time_step"]
                    projected_target[2] = aif_vel * np.cos(target_heading)
                    projected_target[3] = aif_vel * np.sin(target_heading)
                    agent_target_states_for_mpc[agent_id_actual] = projected_target
            
            for ag_id_default in params["agent_ids"]:
                if ag_id_default not in agent_target_states_for_mpc:
                    agent_target_states_for_mpc[ag_id_default] = env.agents[ag_id_default].state.copy()

        actual_controls_to_apply = {}
        for agent_id, agent_obj in env.agents.items():
            current_physical_state = agent_obj.state
            mpc_target_s = agent_target_states_for_mpc.get(agent_id, current_physical_state)
            ref_traj_for_mpc = np.vstack([current_physical_state, np.tile(mpc_target_s, (params["mpc_horizon_N"], 1))])
            u_first_mpc, u_opt_seq, _ = mpc_controllers[agent_id].compute_control(
                current_physical_state, ref_traj_for_mpc, mpc_initial_u_guesses[agent_id]
            )
            actual_controls_to_apply[agent_id] = u_first_mpc
            mpc_initial_u_guesses[agent_id] = np.vstack([u_opt_seq[1:,:], u_opt_seq[-1,:]])

        env.apply_control_inputs(actual_controls_to_apply)
        env.step_simulation_time()

        if params["communication_type"] == "strong" and current_coalition_structure_strong:
             for task_id_assigned_check, agent_ids_in_task_check in current_coalition_structure_strong.items():
                if task_id_assigned_check != 0 and task_id_assigned_check in env.tasks:
                    task_obj_check = env.tasks[task_id_assigned_check]
                    if not task_obj_check.is_completed:
                        for ag_id_in_task_check in agent_ids_in_task_check:
                            if np.linalg.norm(env.agents[ag_id_in_task_check].state[:2] - task_obj_check.position) < params.get("task_completion_distance", 1.5):
                                task_obj_check.is_completed = True; break
                        if task_obj_check.is_completed: continue
        
        if visualize and viz:
            viz.record_state(actual_controls_to_apply, strong_comms_belief_mgr, active_inference_agents)

        all_tasks_done = env.tasks and all(t.is_completed for t in env.tasks.values())
        if params["communication_type"] == "strong" and all_tasks_done:
            print(f"\nAll tasks completed at step {step+1}!")
            break
        
        if params["communication_type"] == "weak" and "aif_goals" in params and active_inference_agents:
            # (Convergence check for AIF as before)
            if params.get("aif_scenario_type") == "rendezvous":
                all_aif_pos_check = env.get_all_agent_positions_for_aif()
                aif_goals_check_np = np.array(params["aif_goals"])
                if aif_goals_check_np.ndim == 1: aif_goals_check_np = aif_goals_check_np.reshape(1,-1)
                num_aif_goals_check = aif_goals_check_np.shape[0]
                if num_aif_goals_check > 0:
                    converged_to_aif_goal_counts = [0] * num_aif_goals_check
                    for i_agent_check in range(params["num_agents"]):
                        agent_pos_xy_check = all_aif_pos_check[i_agent_check, :2]
                        for g_idx_check, goal_pos_check in enumerate(aif_goals_check_np):
                            if np.linalg.norm(agent_pos_xy_check - goal_pos_check) < params.get("aif_convergence_radius", 1.0):
                                converged_to_aif_goal_counts[g_idx_check]+=1
                    if any(count == params["num_agents"] for count in converged_to_aif_goal_counts):
                        print(f"\nWeak communication: All agents rendezvoused at an AIF goal at step {step+1}!")
                        break
        if step == max_steps -1 : print("")

    print(f"\n--- Simulation Ended at step {step+1} (Time: {env.current_time:.1f}s) ---")
    end_time_total = time.time()
    print(f"Total simulation runtime: {end_time_total - start_time_total:.2f} seconds.")

    # Prepare data for saving
    history_to_save = viz.history if viz else {} # Get history from visualizer if it exists
    # Convert numpy arrays in history to lists for JSON serialization
    serializable_history = {}
    if history_to_save:
        for key, value in history_to_save.items():
            if isinstance(value, dict):
                serializable_history[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list) and sub_value and isinstance(sub_value[0], np.ndarray):
                        serializable_history[key][sub_key] = [arr.tolist() for arr in sub_value]
                    elif isinstance(sub_value, list) and sub_value and isinstance(sub_value[0], dict): # e.g. aif_system_beliefs_weak might be list of dicts
                        # Check if dicts contain numpy arrays
                        processed_sub_list = []
                        for item_in_list in sub_value:
                            if isinstance(item_in_list, np.ndarray):
                                processed_sub_list.append(item_in_list.tolist())
                            elif isinstance(item_in_list, dict):
                                processed_dict = {k_dict: v_dict.tolist() if isinstance(v_dict, np.ndarray) else v_dict 
                                                  for k_dict, v_dict in item_in_list.items()}
                                processed_sub_list.append(processed_dict)
                            else:
                                processed_sub_list.append(item_in_list)
                        serializable_history[key][sub_key] = processed_sub_list
                    else:
                        serializable_history[key][sub_key] = sub_value
            elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                serializable_history[key] = [arr.tolist() for arr in value]
            else:
                serializable_history[key] = value
    
    output_plot_dir = "results"
    plot_scenario_prefix = scenario_name
    if animation_filename_prefix:
        output_plot_dir = os.path.dirname(animation_filename_prefix) if os.path.dirname(animation_filename_prefix) else "results"
        plot_scenario_prefix = os.path.basename(animation_filename_prefix)
    if not os.path.exists(output_plot_dir): os.makedirs(output_plot_dir, exist_ok=True)

    # Save history data to JSON
    history_json_path = os.path.join(output_plot_dir, f"{plot_scenario_prefix}_history.json")
    try:
        with open(history_json_path, 'w') as f:
            json.dump(serializable_history, f, indent=4)
        print(f"Simulation history saved to: {history_json_path}")
    except Exception as e:
        print(f"Error saving history to JSON: {e}")


    if visualize and viz:
        print("Generating animation...")
        try:
            ani = viz.create_animation(interval=max(50, int(params["sim_time_step"]*1000 / 2.0)),
                                   output_filename=animation_filename_prefix)
            if animation_filename_prefix is None and ani is not None:
                 plt.show()
        except Exception as e:
            print(f"Could not generate or display animation: {e}")
        
        print("Generating static plots...")
        viz.plot_final_static_charts(results_dir=output_plot_dir, scenario_name_prefix=plot_scenario_prefix)
    
    plt.close('all')
    final_snapshot = env.get_system_snapshot()
    env_instance_for_fuel_calc = None
    return final_snapshot


if __name__ == "__main__":
    # scenario_to_run = "strong_3agents_2tasks"
    scenario_to_run = "weak_2agents_2tasks_rendezvous"
    # scenario_to_run = "weak_3agents_3tasks_exclusive"
    
    seed_for_run = 42

    results_output_dir_main = "results"
    if not os.path.exists(results_output_dir_main):
        os.makedirs(results_output_dir_main, exist_ok=True)
    
    output_file_base_main = scenario_to_run
    if seed_for_run is not None:
        output_file_base_main += f"_seed{seed_for_run}"
    
    animation_path_prefix_main = os.path.join(results_output_dir_main, output_file_base_main)

    results_data = run_simulation(
        scenario_name=scenario_to_run,
        random_seed=seed_for_run,
        visualize=True, # Set to False for headless runs
        animation_filename_prefix=animation_path_prefix_main
    )
    print("\nRun_simulation finished.")