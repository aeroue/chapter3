# simulation/environment.py

import numpy as np
import random
import copy

try:
    from common.dynamics import CWDynamics
except ImportError:
    print("Warning: CWDynamics not found in common.dynamics. Assuming a placeholder for Environment.")
    # Placeholder if CWDynamics is not yet available or path is not set
    class CWDynamics:
        def __init__(self, n, ts): self.n = n; self.ts = ts; self.state_dim = 4; self.control_dim = 2
        def step(self, state, control): return state + np.hstack((state[2:], control)) * self.ts # Simple integrator
        def get_discrete_matrices(self): return np.eye(4), np.eye(4,2)*self.ts # Placeholder


class Task:
    """Represents a task in the environment."""
    def __init__(self, task_id: int, position: np.ndarray, true_type: int,
                 properties: dict = None):
        """
        Args:
            task_id (int): Unique identifier for the task.
            position (np.ndarray): 2D position [x, y] of the task.
            true_type (int): The actual, underlying type of the task (0 to K-1).
            properties (dict, optional): Additional task properties (e.g., value, risk from type).
        """
        self.id = task_id
        self.position = np.array(position, dtype=float)
        self.true_type = true_type # Ground truth
        self.properties = properties if properties is not None else {}
        self.assigned_coalition_id = None # Which coalition is working on it
        self.is_completed = False

    def __repr__(self):
        return f"Task(id={self.id}, pos={self.position}, type={self.true_type}, completed={self.is_completed})"


class SpacecraftAgent:
    """Represents a spacecraft agent in the simulation."""
    def __init__(self, agent_id: int, initial_state: np.ndarray,
                 dynamics_model: CWDynamics,
                 agent_type_properties: dict = None, # For heterogeneity
                 k_max_length: int = 1 # For KSC
                ):
        """
        Args:
            agent_id (int): Unique identifier for the agent.
            initial_state (np.ndarray): Initial physical state [x, y, vx, vy].
            dynamics_model (CWDynamics): The dynamics model for this agent.
            agent_type_properties (dict, optional): e.g., {'sensor_type': 'A', 'max_thrust': 0.1}.
            k_max_length (int): k-value for KSC algorithm for this agent.
        """
        self.id = agent_id
        self.state = np.array(initial_state, dtype=float) # [x, y, vx, vy]
        self.dynamics_model = dynamics_model
        self.properties = agent_type_properties if agent_type_properties is not None else {}
        self.k_max_length = k_max_length # Specific K for this agent

        # Beliefs and decision-making modules will be attached externally or managed by higher-level controllers
        self.assigned_task_id = 0 # 0 for virtual/unassigned task
        self.current_control_input = np.zeros(dynamics_model.control_dim)

    def update_state(self, control_input: np.ndarray):
        """Updates the agent's state based on control input and dynamics."""
        self.current_control_input = np.array(control_input, dtype=float)
        self.state = self.dynamics_model.step(self.state, self.current_control_input)

    def get_position_for_aif(self) -> np.ndarray:
        """Returns position [x, y, theta] for AIF. Theta is vx for now if not available."""
        # AIF often expects [x,y,theta]. CW state is [x,y,vx,vy].
        # We need a consistent way to get/represent heading if AIF depends on it.
        # For 2D CW, absolute heading isn't inherent. Relative heading to velocity vector is.
        # Using vx as a proxy for heading component or assuming a fixed heading for simplicity for AIF.
        # Or, if AIF's choice_heuristic is purely based on relative positions, then theta might not be critical.
        # Let's assume vx direction can infer a heading, or use 0 if vx is near zero.
        theta = np.arctan2(self.state[3], self.state[2]) if np.linalg.norm(self.state[2:]) > 1e-3 else 0.0
        return np.array([self.state[0], self.state[1], theta])


    def __repr__(self):
        state_str = ", ".join([f"{x:.2f}" for x in self.state])
        return f"Agent(id={self.id}, state=[{state_str}], task={self.assigned_task_id})"


class SimulationEnvironment:
    """Manages the simulation environment, including agents and tasks."""

    def __init__(self, sim_time_step: float):
        self.agents: dict[int, SpacecraftAgent] = {}
        self.tasks: dict[int, Task] = {}
        self.sim_time_step = sim_time_step # Corresponds to ts for CWDynamics
        self.current_time = 0.0
        self.communication_graph = {} # {agent_id: [neighbor_ids]}

    def add_agent(self, agent: SpacecraftAgent):
        if agent.id in self.agents:
            raise ValueError(f"Agent with ID {agent.id} already exists.")
        self.agents[agent.id] = agent

    def add_task(self, task: Task):
        if task.id in self.tasks:
            raise ValueError(f"Task with ID {task.id} already exists.")
        if task.id == 0:
            raise ValueError("Task ID 0 is reserved for the virtual/unassigned task.")
        self.tasks[task.id] = task

    def set_communication_graph(self, graph: dict[int, list[int]]):
        """Sets the communication graph for agents."""
        # Validate graph (e.g., all agent_ids exist)
        for agent_id, neighbors in graph.items():
            if agent_id not in self.agents:
                raise ValueError(f"Agent ID {agent_id} in communication graph not in environment agents.")
            for neighbor_id in neighbors:
                if neighbor_id not in self.agents:
                    raise ValueError(f"Neighbor ID {neighbor_id} for agent {agent_id} not in environment agents.")
        self.communication_graph = copy.deepcopy(graph)


    def get_agent_states_for_utility_calc(self) -> dict[int, np.ndarray]:
        """Returns a dictionary of agent_id to their current physical state [x,y,vx,vy]."""
        return {ag_id: ag.state.copy() for ag_id, ag in self.agents.items()}

    def get_all_agent_positions_for_aif(self) -> np.ndarray:
        """
        Returns a numpy array of all agent positions suitable for AIFWrapper,
        shape (num_agents, 3) where each row is [x, y, theta].
        Agents are ordered by their sorted IDs.
        """
        sorted_agent_ids = sorted(list(self.agents.keys()))
        positions = np.zeros((len(sorted_agent_ids), 3))
        for i, agent_id in enumerate(sorted_agent_ids):
            positions[i, :] = self.agents[agent_id].get_position_for_aif()
        return positions

    def get_all_agent_full_states(self) -> dict[int, np.ndarray]:
        """Returns all agents' full CW states."""
        return {id: agent.state.copy() for id, agent in self.agents.items()}

    def get_task_info_for_utility(self, task_id: int) -> dict | None:
        """
        Provides necessary info about a task for the UtilityCalculator.
        Mainly 'id' and 'position'.
        """
        if task_id == 0: # Virtual task
            return {'id': 0, 'position': np.array([np.inf, np.inf])} # Or some convention
        task = self.tasks.get(task_id)
        if task:
            return {'id': task.id, 'position': task.position.copy()}
        return None

    def apply_control_inputs(self, control_inputs_dict: dict[int, np.ndarray]):
        """
        Applies control inputs to each specified agent and steps their dynamics.
        Args:
            control_inputs_dict (dict[int, np.ndarray]): {agent_id: control_vector}
        """
        for agent_id, control in control_inputs_dict.items():
            if agent_id in self.agents:
                self.agents[agent_id].update_state(control)
            else:
                print(f"Warning: Control input provided for non-existent agent ID {agent_id}.")

    def step_simulation_time(self):
        """Advances simulation time by one sim_time_step."""
        self.current_time += self.sim_time_step

    def get_system_snapshot(self) -> dict:
        """Returns a snapshot of the current environment state."""
        return {
            "time": self.current_time,
            "agent_states": {id: ag.state.copy() for id, ag in self.agents.items()},
            "agent_assignments": {id: ag.assigned_task_id for id, ag in self.agents.items()},
            "task_states": {id: {"completed": t.is_completed, "position": t.position.copy()} for id, t in self.tasks.items()}
        }

if __name__ == '__main__':
    # --- Example Usage ---
    # Define CW Dynamics parameters (should be consistent for all agents using it)
    N_ORBITAL = 0.00113  # rad/s (approx LEO)
    SIM_TS = 1.0      # Simulation time step in seconds

    # Create dynamics model instance
    # In a real scenario, this might be passed to each agent or agents might have different dynamics
    shared_dynamics = CWDynamics(n=N_ORBITAL, ts=SIM_TS)

    # Create Environment
    env = SimulationEnvironment(sim_time_step=SIM_TS)

    # Add Agents
    agent0 = SpacecraftAgent(agent_id=0, initial_state=np.array([10.0, 0.0, 0.1, 0.05]), dynamics_model=shared_dynamics)
    agent1 = SpacecraftAgent(agent_id=1, initial_state=np.array([-5.0, 5.0, 0.0, -0.1]), dynamics_model=shared_dynamics)
    agent2 = SpacecraftAgent(agent_id=2, initial_state=np.array([0.0, -10.0, -0.05, 0.0]), dynamics_model=shared_dynamics)
    env.add_agent(agent0)
    env.add_agent(agent1)
    env.add_agent(agent2)

    # Add Tasks (Task ID 0 is virtual/unassigned)
    task1 = Task(task_id=1, position=np.array([50.0, 20.0]), true_type=0) # Type 0
    task2 = Task(task_id=2, position=np.array([-30.0, -15.0]), true_type=1) # Type 1
    env.add_task(task1)
    env.add_task(task2)

    # Set communication graph (fully connected for this example)
    env.set_communication_graph({
        0: [1, 2],
        1: [0, 2],
        2: [0, 1]
    })

    print("Initial Environment State:")
    print(f"Time: {env.current_time}")
    for ag_id, agent_obj in env.agents.items():
        print(agent_obj)
    for task_id, task_obj in env.tasks.items():
        print(task_obj)
    print(f"Communication Graph: {env.communication_graph}")


    # Simulate a few steps with dummy controls
    dummy_controls = {
        0: np.array([0.001, 0.0005]),
        1: np.array([-0.0005, 0.001]),
        2: np.array([0.0, -0.0005])
    }
    num_sim_steps = 3
    print(f"\nSimulating {num_sim_steps} steps...")
    for i in range(num_sim_steps):
        env.apply_control_inputs(dummy_controls)
        env.step_simulation_time()
        print(f"\nAfter step {i+1}, Time: {env.current_time:.1f}s")
        for ag_id, agent_obj in env.agents.items():
            print(f"  {agent_obj}")

    print("\nSnapshot:")
    print(env.get_system_snapshot())

    print("\nAgent positions for AIF:")
    print(env.get_all_agent_positions_for_aif())