# strong_communication/ksc_coalition_formation.py
# 文件路径: strong_communication/ksc_coalition_formation.py

import numpy as np
import random
import copy
import itertools # 用于在更复杂的链搜索中生成组合 (当前版本简化了搜索)
import matplotlib.pyplot as plt # 仅用于 __main__ 测试块中的绘图
import os # 仅用于 __main__ 测试块中的绘图保存

class KSCCoalitionFormation:
    """
    实现 K-Serial 稳定联盟 (KSS) 形成算法。
    智能体迭代地提出和评估“变换链”(transformation chains)，以改善联盟结构的整体效用。
    该算法是分布式的，依赖于局部通信和共识机制。
    结合了 KS-COAL 的 K-串行稳定性概念和 "Self-Learning"（通过外部信念管理器更新的信念）。

    主要参考:
    - 用户初稿 (定义 3.1-3.3, 算法 3.1) (source: 4051, 4052)
    - Huo et al., "A Self-Learning Approach..." (分布式算法, Nash稳定性, 信念更新) (source: 3341, 3343, 3386, 3465)
    - Chen et al., "Accelerated K-Serial Stable Coalition..." (KSS 定义, 链式变换) (source: 3138, 3145)
    - KS_COAL中文版 (KSS 定义, 链式变换, 分布式消息传递) (source: 2273, 2274, 2275)
    """

    def __init__(self,
                 agent_ids: list[int],
                 task_ids_actual: list[int], # 实际任务ID列表 (例如 [1, 2, 3])
                 utility_calculator,        # UtilityCalculator 实例
                 belief_manager,            # BeliefManager 实例
                 k_max_lengths: dict[int, int], # {agent_id: k_val} 每个智能体考虑的最大链长度
                 communication_graph: dict[int, list[int]], # 通信图: {agent_id: [neighbor_ids]}
                 max_distinct_agents_in_chain: int = 2, # 在一次变换链搜索中涉及的最多不同智能体数量
                 ksc_utility_history_for_plot: list = None # 用于记录KSC收益演化的列表 (外部传入)
                ):
        """
        初始化 KSC 联盟形成过程。

        Args:
            agent_ids (list[int]): 唯一的智能体ID列表。
            task_ids_actual (list[int]): 唯一的实际任务ID列表 (例如 [1, 2, 3])。任务0被视为虚拟/未分配任务。
            utility_calculator (UtilityCalculator): 用于计算智能体效用的实例。
            belief_manager (BeliefManager): 用于获取智能体信念以计算效用的实例。
            k_max_lengths (dict[int, int]): 每个智能体考虑的最大变换链长度。
            communication_graph (dict[int, list[int]]): 定义智能体间的通信能力。
            max_distinct_agents_in_chain (int): 在单个智能体的变换链搜索中，允许参与切换的最多不同智能体数量。
            ksc_utility_history_for_plot (list, optional): 从外部传入的列表，用于记录每次KSC主要迭代后的全局效用。
        """
        self.agent_ids = sorted(list(agent_ids)) # 确保顺序一致
        self.num_agents = len(agent_ids)
        self.task_ids_actual = sorted(list(task_ids_actual))
        self.all_task_options_for_switch = [0] + self.task_ids_actual # 0 代表未分配

        if utility_calculator is None or belief_manager is None:
            raise ValueError("UtilityCalculator 和 BeliefManager 必须被提供。")
        self.utility_calculator = utility_calculator
        self.belief_manager = belief_manager # BeliefManager 用于获取信念

        self.k_max_lengths = k_max_lengths
        self.communication_graph = communication_graph
        # 确保 max_distinct_agents_in_chain 至少为1且不超过智能体总数
        self.max_distinct_agents_in_chain = min(max(1, max_distinct_agents_in_chain), self.num_agents)

        # 用于绘图的KSC效用历史记录
        self.ksc_utility_history = ksc_utility_history_for_plot if ksc_utility_history_for_plot is not None else []

        # 每个智能体的内部状态:
        # local_coalition_structures: dict {agent_id_viewer: {task_id: set_of_agent_ids_in_task}}
        # task_id 0 代表未分配的智能体。
        self.local_coalition_structures: dict[int, dict[int, set[int]]] = {}
        self.local_structure_versions: dict[int, int] = {} # 用于共识的版本号
        self.local_structure_timestamps: dict[int, float] = {} # 用于共识的时间戳
        # current_assignments_map: dict {agent_id_viewer: {agent_id_assigned: task_id_assigned_to}}
        # 这个映射是从 local_coalition_structures 派生出来的，用于快速查找。
        self.current_assignments_map: dict[int, dict[int, int]] = {}

        self._initialize_structures() # 初始化本地视图

    def _initialize_structures(self):
        """为每个智能体的本地视图初始化联盟结构，所有智能体最初都在虚拟联盟 (任务0) 中。"""
        initial_structure_template = {0: set(self.agent_ids)} # 所有智能体最初都在任务0
        for task_id in self.task_ids_actual:
            initial_structure_template[task_id] = set()

        for agent_id in self.agent_ids:
            self.local_coalition_structures[agent_id] = copy.deepcopy(initial_structure_template)
            self.local_structure_versions[agent_id] = 0
            self.local_structure_timestamps[agent_id] = random.random() # 用于版本号相同时的共识打破僵局
            # 初始化当前分配映射
            self.current_assignments_map[agent_id] = {aid: 0 for aid in self.agent_ids}

    def _get_agent_current_task_from_structure(self, structure: dict[int, set[int]], agent_id_to_find: int) -> int:
        """从给定的联盟结构中查找一个智能体当前分配的任务。"""
        for task_id, agents_in_task in structure.items():
            if agent_id_to_find in agents_in_task:
                return task_id
        # 如果智能体未在任何实际任务联盟中找到，则假定它在虚拟任务0中
        if 0 in structure and agent_id_to_find in structure.get(0, set()): # 检查是否在任务0中
            return 0
        # print(f"警告: 智能体 {agent_id_to_find} 未在结构 {structure} 的任何任务中找到。默认返回任务0。")
        return 0 # 如果结构不一致或agent_id_to_find不存在，回退到未分配

    def _calculate_overall_system_utility(self,
                                     structure: dict[int, set[int]],
                                     agent_physical_states: dict[int, np.ndarray],
                                     belief_context_agent_id: int # 必须提供一个用于评估的智能体信念上下文
                                    ) -> float:
        """
        计算给定联盟结构的总系统效用。
        效用 = 所有智能体个体期望效用之和。
        所有任务的期望都基于 belief_context_agent_id 的信念进行评估，以保证比较的一致性。
        这符合用户初稿中最大化 sum(sum(u_i(C_j))) 的目标 (source: 4050)。

        Args:
            structure (dict): 要评估的联盟结构 {task_id: set_of_agent_ids}。
            agent_physical_states (dict): {agent_id: state_vector} 航天器的物理状态，用于燃料计算。
            belief_context_agent_id (int): 用于评估所有任务期望效用的智能体ID。

        Returns:
            float: 该联盟结构的总（或平均）系统效用。
        """
        total_utility_sum = 0.0
        
        # 临时构建一个从智能体到任务的分配映射，以便于查找
        agent_to_task_map_eval = {}
        for task_id_eval, agents_in_task_eval in structure.items():
            for agent_id_eval in agents_in_task_eval:
                agent_to_task_map_eval[agent_id_eval] = task_id_eval

        for agent_id_iter in self.agent_ids: # 遍历系统中的所有智能体
            assigned_task_id = agent_to_task_map_eval.get(agent_id_iter, 0) # 默认未分配
            agent_state_for_fuel = agent_physical_states.get(agent_id_iter, np.zeros(4)) # 默认状态

            if assigned_task_id == 0: # 未分配任务的智能体效用通常为0
                total_utility_sum += 0.0
                continue

            # 任务ID在BeliefManager中是0索引的。实际任务ID是1-indexed。
            task_idx_for_belief = assigned_task_id - 1 
            if not (0 <= task_idx_for_belief < self.belief_manager.num_tasks):
                # print(f"警告(效用计算): 任务ID {assigned_task_id} (索引 {task_idx_for_belief}) "
                #       f"超出BeliefManager范围({self.belief_manager.num_tasks})。智能体 {agent_id_iter} 的效用计为负无穷。")
                total_utility_sum += -np.inf # 对于无效的任务分配给予大的负效用
                continue
                
            # 使用指定的 belief_context_agent_id 的信念来评估任务
            agent_belief_dist = self.belief_manager.get_expected_belief_distribution(
                belief_context_agent_id, task_idx_for_belief
            )
            
            # 燃料计算器需要任务ID，它内部会查找任务位置
            task_info_for_fuel = {'id': assigned_task_id}

            num_agents_in_this_coalition = len(structure.get(assigned_task_id, set()))
            if num_agents_in_this_coalition == 0: # 不应发生，因为 agent_id_iter 在这个联盟里
                # print(f"警告: 智能体 {agent_id_iter} 被分配到任务 {assigned_task_id}，但联盟为空。")
                continue # 或者给予负效用

            individual_expected_utility = self.utility_calculator.calculate_expected_utility(
                agent_id=agent_id_iter,
                agent_state=agent_state_for_fuel,
                task_info=task_info_for_fuel,
                agent_belief_for_task=agent_belief_dist, # 使用 belief_context_agent_id 的信念
                num_agents_in_coalition_if_joined=num_agents_in_this_coalition
            )
            total_utility_sum += individual_expected_utility
            
        return total_utility_sum

    def _apply_transformation_chain(self, base_structure: dict[int, set[int]],
                                    chain_of_switches: list[tuple[int, int]]) -> dict[int, set[int]]:
        """
        将一系列切换操作 (变换链) 应用于基础联盟结构。
        一个切换是 (要移动的智能体ID, 该智能体的新任务ID)。
        返回一个新的结构 (深拷贝)。
        """
        new_structure = copy.deepcopy(base_structure)
        for agent_to_switch, new_task_id_for_agent in chain_of_switches:
            current_task_of_agent = self._get_agent_current_task_from_structure(new_structure, agent_to_switch)

            if current_task_of_agent != new_task_id_for_agent: # 仅当任务确实改变时才操作
                if agent_to_switch in new_structure.get(current_task_of_agent, set()):
                    new_structure[current_task_of_agent].remove(agent_to_switch)
                
                new_structure.setdefault(new_task_id_for_agent, set()).add(agent_to_switch)
        return new_structure

    def _find_improving_transformation_chain(self,
                                             focal_agent_id: int,
                                             agent_physical_states: dict[int, np.ndarray]
                                            ) -> tuple[dict | None, float, list | None]:
        """
        焦点智能体 (focal_agent_id) 尝试找到一个变换链。
        该链根植于自身，可能涉及邻居，长度不超过 k_max_lengths[focal_agent_id]，
        并且能够改善其当前局部联盟结构的效用 (从focal_agent_id的信念角度评估)。

        Returns:
            (最佳新结构, 最佳新效用, 最佳切换链元组列表) 或 (None, 当前效用, None) 如果没有找到改进。
        """
        current_local_struct = self.local_coalition_structures[focal_agent_id]
        current_utility = self._calculate_overall_system_utility(current_local_struct, agent_physical_states, focal_agent_id)

        best_improved_structure = None
        best_utility_found = current_utility
        best_chain_ops_list_of_tuples = None 

        max_k_len_for_focal = self.k_max_lengths.get(focal_agent_id, 1) # 获取该智能体的K值

        # 使用深度优先搜索 (DFS) 来探索变换链
        # 栈元素: (current_chain_operations_dicts, last_switched_agent)
        # current_chain_operations_dicts: [{'agent': id, 'new_task': tid, 'old_task': tid}, ...]
        dfs_stack = [([], None)] # 初始链为空，上一个切换者为None

        while dfs_stack:
            chain_dicts_so_far, last_switched_agent = dfs_stack.pop()  # 只解包两个值

            # 1. 基于当前链 (chain_dicts_so_far) 从初始结构 current_local_struct 构建新结构并评估
            # 每次都从原始的 current_local_struct 应用完整的链
            current_chain_tuples_for_apply = [(op['agent'], op['new_task']) for op in chain_dicts_so_far]
            structure_after_current_chain = self._apply_transformation_chain(current_local_struct, current_chain_tuples_for_apply)

            if chain_dicts_so_far: # 只有当链非空（即至少发生了一次切换）时才评估和比较
                utility_of_this_chain_outcome = self._calculate_overall_system_utility(
                    structure_after_current_chain, agent_physical_states, focal_agent_id
                )
                if utility_of_this_chain_outcome > best_utility_found + 1e-7: # 增加容差
                    best_utility_found = utility_of_this_chain_outcome
                    best_improved_structure = structure_after_current_chain
                    best_chain_ops_list_of_tuples = current_chain_tuples_for_apply

            # 2. 如果链长度未达到上限，尝试扩展链
            if len(chain_dicts_so_far) < max_k_len_for_focal:
                agents_eligible_for_next_switch = []
                if not chain_dicts_so_far: # 链的第一个切换操作必须由焦点智能体发起
                    agents_eligible_for_next_switch = [focal_agent_id]
                else: # 后续切换操作可以由链中最后一个切换的智能体的邻居发起，或其自身再次切换
                    if last_switched_agent is not None:
                         agents_eligible_for_next_switch.append(last_switched_agent) # 允许自己再次切换到不同任务
                         agents_eligible_for_next_switch.extend(self.communication_graph.get(last_switched_agent, []))
                         agents_eligible_for_next_switch = list(set(agents_eligible_for_next_switch))
                    else: # 不应发生
                        agents_eligible_for_next_switch = [focal_agent_id]
                
                # 限制参与链的独特智能体数量 (可选的复杂性控制)
                # current_distinct_agents_in_chain = {op['agent'] for op in chain_dicts_so_far}
                # if len(current_distinct_agents_in_chain) >= self.max_distinct_agents_in_chain and \
                #    agent_for_next_switch not in current_distinct_agents_in_chain:
                #    continue # 如果达到独特智能体数量上限，且下一个切换者是新智能体，则跳过

                for agent_for_next_switch in agents_eligible_for_next_switch:
                    # 获取该智能体在当前链作用下的任务 (即 structure_after_current_chain 中的任务)
                    original_task_of_agent = self._get_agent_current_task_from_structure(structure_after_current_chain, agent_for_next_switch)
                    
                    for new_task_for_agent in self.all_task_options_for_switch:
                        if new_task_for_agent == original_task_of_agent: continue # 不是有效的切换

                        next_op_dict = {'agent': agent_for_next_switch,
                                        'new_task': new_task_for_agent,
                                        'old_task': original_task_of_agent}
                        
                        # 避免简单地立即撤销链中的上一个操作
                        if chain_dicts_so_far and \
                           chain_dicts_so_far[-1]['agent'] == agent_for_next_switch and \
                           chain_dicts_so_far[-1]['new_task'] == original_task_of_agent and \
                           next_op_dict['new_task'] == chain_dicts_so_far[-1]['old_task']:
                            continue
                        
                        # 检查一个智能体是否在一个链中切换了太多次 (可选)
                        # num_switches_this_agent = sum(1 for op in chain_dicts_so_far if op['agent'] == agent_for_next_switch)
                        # if num_switches_this_agent >= some_agent_switch_limit_in_chain: continue

                        temp_extended_chain_dicts = chain_dicts_so_far + [next_op_dict]
                        dfs_stack.append((temp_extended_chain_dicts, agent_for_next_switch))  # 只添加两个值
        
        return best_improved_structure, best_utility_found, best_chain_ops_list_of_tuples

    def run_iteration(self, agent_physical_states: dict[int, np.ndarray]) -> tuple[bool, float]:
        """
        运行 KSC 联盟形成算法的一次主迭代。
        包括每个智能体的本地改进尝试和共识阶段。

        Args:
            agent_physical_states (dict[int, np.ndarray]): 所有智能体的当前物理状态。

        Returns:
            tuple[bool, float]: (是否有任何智能体的本地结构在本轮迭代中发生改变, agent_ids[0] 的最终局部效用)
        """
        changed_overall_in_iteration = False
        newly_adopted_structures_phase1 = {} # 存储在阶段1中采纳的新结构

        # --- 阶段 1: 本地改进 ---
        for agent_id in self.agent_ids:
            improved_structure, new_utility_from_chain, _ = self._find_improving_transformation_chain(
                agent_id, agent_physical_states
            )
            current_view_utility = self._calculate_overall_system_utility(
                 self.local_coalition_structures[agent_id], agent_physical_states, agent_id
            )
            if improved_structure is not None and new_utility_from_chain > current_view_utility + 1e-7: # 仅当严格更优时
                newly_adopted_structures_phase1[agent_id] = improved_structure
                # 版本号和时间戳在实际应用此结构时更新

        # 应用在阶段1中找到的改进结构到各自智能体的局部视图
        for agent_id, new_struct in newly_adopted_structures_phase1.items():
            if self.local_coalition_structures[agent_id] != new_struct: # 确保结构真的改变了
                self.local_coalition_structures[agent_id] = new_struct # new_struct 已经是深拷贝
                self.local_structure_versions[agent_id] += 1
                self.local_structure_timestamps[agent_id] = random.random()
                changed_overall_in_iteration = True
                # 更新该智能体对其自身视图中所有智能体分配的映射
                for t_id, ags_in_t in new_struct.items():
                    for ag_in_set in ags_in_t: # 使用不同的变量名避免覆盖外部循环变量
                        self.current_assignments_map[agent_id][ag_in_set] = t_id

        # --- 阶段 2: 共识 ---
        max_consensus_passes = len(self.agent_ids) # 启发式：允许信息在网络中传播足够的轮次
        for _ in range(max_consensus_passes):
            made_change_in_this_consensus_pass = False
            for agent_id_viewer in self.agent_ids:
                current_viewer_version = self.local_structure_versions[agent_id_viewer]
                current_viewer_timestamp = self.local_structure_timestamps[agent_id_viewer]

                for neighbor_id in self.communication_graph.get(agent_id_viewer, []):
                    neighbor_version = self.local_structure_versions[neighbor_id]
                    neighbor_timestamp = self.local_structure_timestamps[neighbor_id]

                    should_adopt_neighbor_view = False
                    if neighbor_version > current_viewer_version:
                        should_adopt_neighbor_view = True
                    elif neighbor_version == current_viewer_version and neighbor_timestamp > current_viewer_timestamp:
                        should_adopt_neighbor_view = True
                    
                    if should_adopt_neighbor_view:
                        # 仅当版本号或时间戳更新，或结构本身不同时才执行复制和更新
                        # （避免不必要的深拷贝）
                        is_different_struct = self.local_coalition_structures[agent_id_viewer] != self.local_coalition_structures[neighbor_id]
                        if is_different_struct or \
                           self.local_structure_versions[agent_id_viewer] != neighbor_version or \
                           self.local_structure_timestamps[agent_id_viewer] != neighbor_timestamp:
                            
                            self.local_coalition_structures[agent_id_viewer] = copy.deepcopy(self.local_coalition_structures[neighbor_id])
                            self.local_structure_versions[agent_id_viewer] = neighbor_version
                            self.local_structure_timestamps[agent_id_viewer] = neighbor_timestamp
                            
                            current_viewer_version = neighbor_version 
                            current_viewer_timestamp = neighbor_timestamp
                            
                            made_change_in_this_consensus_pass = True
                            changed_overall_in_iteration = True
                            
                            new_struct_for_viewer = self.local_coalition_structures[agent_id_viewer]
                            for t_id_cons, ags_in_t_cons in new_struct_for_viewer.items():
                                for ag_in_set_cons in ags_in_t_cons:
                                    self.current_assignments_map[agent_id_viewer][ag_in_set_cons] = t_id_cons
            
            if not made_change_in_this_consensus_pass:
                break # 如果一整轮共识传递都没有发生变化，则认为共识在本轮迭代已稳定
                        
        final_utility_agent0_view = 0.0
        if self.agent_ids: # 确保 agent_ids 列表不为空
            final_utility_agent0_view = self._calculate_overall_system_utility(
                self.local_coalition_structures[self.agent_ids[0]],
                agent_physical_states,
                self.agent_ids[0] # 使用第一个智能体的信念作为参考来记录效用历史
            )
        return changed_overall_in_iteration, final_utility_agent0_view

    def form_coalitions(self, agent_physical_states: dict[int, np.ndarray],
                        max_iterations: int = 20, 
                        convergence_patience: int = 3) -> dict[int, set[int]]:
        """
        运行联盟形成过程直到收敛或达到最大迭代次数。

        Args:
            agent_physical_states (dict[int, np.ndarray]): 所有智能体的当前物理状态。
            max_iterations (int): 允许的最大迭代次数。
            convergence_patience (int): 在声明收敛前，没有变化的迭代次数。

        Returns:
            dict[int, set[int]]: 从 agent_ids[0] 的角度看，最终的（或收敛的）联盟结构。
        """
        self.ksc_utility_history.clear() # 清除上一轮 KSC 调用的历史
        
        # 记录 KSC 开始前的初始效用
        if self.agent_ids:
            initial_utility_for_history = self._calculate_overall_system_utility(
                 self.local_coalition_structures[self.agent_ids[0]], agent_physical_states, self.agent_ids[0]
            )
            self.ksc_utility_history.append(initial_utility_for_history)

        no_change_streak = 0
        for i in range(max_iterations):
            changed_this_iter, current_iter_utility = self.run_iteration(agent_physical_states)
            self.ksc_utility_history.append(current_iter_utility) # 记录每轮KSC迭代后的效用

            if not changed_this_iter:
                no_change_streak += 1
            else:
                no_change_streak = 0

            if no_change_streak >= convergence_patience:
                # print(f"KSC 在 {i + 1} 次迭代后收敛 (其中 {convergence_patience} 次迭代稳定)。")
                break
        # else: # 如果循环正常结束（未通过break退出）
            # print(f"KSC 达到最大迭代次数 ({max_iterations})。")

        return self.local_coalition_structures[self.agent_ids[0]] if self.agent_ids else {}


if __name__ == '__main__':
    # --- 模拟类定义 (用于独立测试 KSCCoalitionFormation) ---
    class MockBeliefManager:
        def __init__(self, num_s, num_t_actual, num_k_types):
            self.num_spacecraft = num_s; self.num_tasks = num_t_actual; self.num_task_types = num_k_types
            self.alpha_parameters = np.ones((num_s, num_t_actual, num_k_types)) # 初始化alpha为1
        def get_expected_belief_distribution(self, agent_id, task_idx_0_based):
            if not (0 <= task_idx_0_based < self.num_tasks):
                # print(f"MockBM 警告: task_idx_0_based {task_idx_0_based} 超出任务数量 {self.num_tasks} 的范围")
                return np.full(self.num_task_types, 1.0/self.num_task_types if self.num_task_types > 0 else 0)
            alphas = self.alpha_parameters[agent_id, task_idx_0_based, :]
            return alphas / np.sum(alphas) if np.sum(alphas) > 1e-9 else np.full(self.num_task_types, 1.0/self.num_task_types if self.num_task_types > 0 else 0)

    class MockUtilityCalculator:
        def __init__(self, task_props_by_type, fuel_calc_func, n_types):
            self.task_properties_by_type = task_props_by_type
            self.fuel_cost_calculator = fuel_calc_func
            self.n_task_types = n_types
            # 确保revenues_by_type_arr和risks_by_type_arr的索引与task_type一致
            self.revenues_by_type_arr = np.zeros(n_types)
            self.risks_by_type_arr = np.zeros(n_types)
            for type_idx, props in task_props_by_type.items():
                 if isinstance(props, dict) and 0 <= type_idx < n_types:
                    self.revenues_by_type_arr[type_idx] = props.get("revenue",0)
                    self.risks_by_type_arr[type_idx] = props.get("risk",0)

        def calculate_expected_utility(self, agent_id, agent_state, task_info, agent_belief_for_task, num_agents_in_coalition_if_joined):
            if num_agents_in_coalition_if_joined <= 0: return -np.inf
            expected_total_task_revenue = np.dot(agent_belief_for_task, self.revenues_by_type_arr)
            shared_expected_revenue = expected_total_task_revenue / num_agents_in_coalition_if_joined
            expected_risk_cost = np.dot(agent_belief_for_task, self.risks_by_type_arr)
            fuel_cost = self.fuel_cost_calculator(agent_id, agent_state, task_info)
            return shared_expected_revenue - expected_risk_cost - fuel_cost

    # 固定的测试任务位置
    test_task_positions_for_main = {
        1: np.array([10.0, 0.0]),
        2: np.array([0.0, 10.0])
    }
    def mock_fuel_calculator_main(agent_id, agent_state, task_info_minimal):
        task_id = task_info_minimal.get('id')
        task_pos = test_task_positions_for_main.get(task_id, agent_state[:2] if agent_state is not None else np.array([0.0,0.0]))
        distance = np.linalg.norm((agent_state[:2] if agent_state is not None else np.array([0.0,0.0])) - task_pos)
        return 0.1 + distance * 0.01 # 基础成本 + 距离相关成本

    # --- 测试参数 ---
    main_test_agent_ids = [0, 1, 2]
    main_test_task_ids_actual = [1, 2] # 任务ID从1开始
    main_test_num_k_types = 2
    main_test_agent_states = {
        0: np.array([0.0, 0.0, 0.0, 0.0]),
        1: np.array([1.0, 1.0, 0.0, 0.0]),
        2: np.array([-1.0, -1.0, 0.0, 0.0])
    }
    main_test_task_props_by_type = { # 0-indexed type
        0: {"revenue": 200, "risk": 1, "name":"A型任务 (高收益低风险)"},
        1: {"revenue": 50,  "risk": 5, "name":"B型任务 (低收益高风险)"}
    }

    main_test_belief_mgr = MockBeliefManager(len(main_test_agent_ids), len(main_test_task_ids_actual), main_test_num_k_types)
    # 为测试设置一些非均匀的初始信念
    main_test_belief_mgr.alpha_parameters[0, 0, :] = np.array([10, 1]) # 智能体0认为任务1(索引0)很可能是A型
    main_test_belief_mgr.alpha_parameters[0, 1, :] = np.array([1, 10]) # 智能体0认为任务2(索引1)很可能是B型
    main_test_belief_mgr.alpha_parameters[1, :, :] = np.array([[2, 2],[2, 2]]) # 智能体1对两个任务都是均匀信念
    main_test_belief_mgr.alpha_parameters[2, 0, :] = np.array([1, 10]) # 智能体2认为任务1(索引0)很可能是B型
    main_test_belief_mgr.alpha_parameters[2, 1, :] = np.array([10, 1]) # 智能体2认为任务2(索引1)很可能是A型


    main_test_utility_calc = MockUtilityCalculator(
        main_test_task_props_by_type, mock_fuel_calculator_main, main_test_num_k_types
    )
    main_test_comm_graph = {0:[1], 1:[0,2], 2:[1]} # 简单的链式通信
    main_test_k_lengths = {aid: 1 for aid in main_test_agent_ids} # K=1 意味着寻找Nash稳定解

    main_ksc_utility_hist = [] # 用于存储测试中的效用历史

    ksc_main_test_instance = KSCCoalitionFormation(
        main_test_agent_ids, main_test_task_ids_actual, main_test_utility_calc, main_test_belief_mgr,
        main_test_k_lengths, main_test_comm_graph, # 使用已定义的通信图变量名
        max_distinct_agents_in_chain=1,
        ksc_utility_history_for_plot=main_ksc_utility_hist
    )
    print("独立测试: 初始联盟结构 (智能体0视角):", ksc_main_test_instance.local_coalition_structures[0])
    initial_utility_main_test = ksc_main_test_instance._calculate_overall_system_utility(
        ksc_main_test_instance.local_coalition_structures[0], main_test_agent_states, 0 # 使用智能体0的信念评估
    )
    print(f"独立测试: 初始总效用 (智能体0视角): {initial_utility_main_test:.2f}")

    final_structure_main_test = ksc_main_test_instance.form_coalitions(main_test_agent_states, max_iterations=15, convergence_patience=2)
    print("\n--- 独立测试最终结果 (智能体0视角) ---")
    print("最终联盟结构:", final_structure_main_test)  # Changed from final_structure_ksc_main
    final_utility_main_test = ksc_main_test_instance._calculate_overall_system_utility(final_structure_main_test, main_test_agent_states, 0)
    print(f"最终总效用 (智能体0视角): {final_utility_main_test:.2f}")

    print("\nKSC 效用演化历史 (智能体0视角):")
    for i, util_val in enumerate(main_ksc_utility_hist):
        print(f" KSC迭代 {i}: 总效用 = {util_val:.2f}")

    if main_ksc_utility_hist:
        plt.figure(figsize=(8,5))
        # 绘制从第二次记录开始的效用（第一次是迭代前的初始效用）
        plt.plot(range(len(main_ksc_utility_hist)-1), main_ksc_utility_hist[1:], marker='o', linestyle='-', linewidth=2.0, label="系统总期望效用 (S0视角)")
        plt.xlabel("KSC 主迭代轮次 (KSC Main Iteration)", fontsize=12)
        plt.ylabel("系统总期望效用", fontsize=12)
        plt.title("KS-COAL 算法效用演化 (独立测试)", fontsize=14, weight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, linestyle=':')
        plt.tight_layout()
        if not os.path.exists("results"): os.makedirs("results")
        plt.savefig("results/ksc_standalone_test_utility_evolution.png")
        print("KSC效用演化图已保存至 results/ksc_standalone_test_utility_evolution.png")
        plt.close()