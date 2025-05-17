# strong_communication/belief_manager.py
# 文件路径: strong_communication/belief_manager.py

import numpy as np

class BeliefManager:
    """
    使用狄利克雷分布 (Dirichlet distributions) 管理航天器对任务类型的信念。
    每个信念由对应每种任务类型的伪计数 (alpha 参数) 表示。
    该设计参考了用户初稿中关于狄利克雷分布和聚合观测更新的描述 (source 8135, 8136)，
    以及 Huo et al. 论文中关于信息融合导致共享信念更新的概念 (source 4953, 5047, 5057, 5058)。
    """

    def __init__(self, num_spacecraft: int, num_actual_tasks: int, num_task_types: int,
                 initial_pseudo_counts: float = 1.0):
        """
        初始化 BeliefManager。

        Args:
            num_spacecraft (int): 系统中航天器的总数。
            num_actual_tasks (int): 实际任务的总数 (例如，如果任务ID为1, 2, 则此值为2)。
                                    信念将针对任务索引 0 到 num_actual_tasks-1 进行存储。
            num_task_types (int): 每种任务可能的类型数量 (K)。
            initial_pseudo_counts (float or np.ndarray): 每种任务类型的初始alpha值。
                                           如果是浮点数，则统一应用于所有情况。
                                           如果是 np.ndarray，其形状必须是
                                           (num_spacecraft, num_actual_tasks, num_task_types)。
                                           代表先验知识；1.0 通常表示均匀先验（即每个类型最初被认为是等可能的）。
        """
        if not isinstance(num_spacecraft, int) or num_spacecraft <= 0:
            raise ValueError("航天器数量必须是正整数。")
        if not isinstance(num_actual_tasks, int) or num_actual_tasks < 0: # 允许初始时没有任务
            raise ValueError("实际任务数量必须是非负整数。")
        if not isinstance(num_task_types, int) or num_task_types <= 0:
            raise ValueError("任务类型数量必须是正整数。")

        self.num_spacecraft = num_spacecraft
        self.num_tasks = num_actual_tasks # 内部任务索引从0开始，数量为 num_actual_tasks
        self.num_task_types = num_task_types

        # alpha_parameters 存储信念参数 (伪计数)
        # 维度: (航天器数量, 实际任务数量, 任务类型数量)
        expected_shape = (self.num_spacecraft, self.num_tasks, self.num_task_types)

        if isinstance(initial_pseudo_counts, (float, int)):
            if initial_pseudo_counts <= 0:
                raise ValueError("如果 initial_pseudo_counts 是标量，则必须为正。")
            # 仅当有任务时才初始化具有值的数组
            # 如果 num_tasks 为 0, expected_shape 的第二维将是0, np.full 会创建正确的空数组 (如果num_spacecraft或num_task_types也非0)
            # 或者如果 num_task_types也是0，那会是 (num_spacecraft,0,0)
            self.alpha_parameters = np.full(expected_shape, float(initial_pseudo_counts))

        elif isinstance(initial_pseudo_counts, np.ndarray):
            if initial_pseudo_counts.shape != expected_shape:
                raise ValueError(f"initial_pseudo_counts 数组形状不匹配。期望形状 {expected_shape}，实际为 {initial_pseudo_counts.shape}")
            if initial_pseudo_counts.size > 0 and np.any(initial_pseudo_counts <= 0): # 仅当数组非空时检查元素值
                raise ValueError("initial_pseudo_counts 数组中的所有元素必须为正。")
            self.alpha_parameters = initial_pseudo_counts.copy()
        else:
            raise TypeError("initial_pseudo_counts 必须是浮点数、整数或numpy数组。")

    def get_agent_task_belief_parameters(self, agent_id: int, task_idx_0_based: int) -> np.ndarray:
        """
        检索特定智能体和任务的alpha参数 (伪计数)。

        Args:
            agent_id (int): 航天器ID (0 到 num_spacecraft-1)。
            task_idx_0_based (int): 任务的0索引ID (0 到 num_tasks-1)。
        Returns:
            np.ndarray: 对应任务类型的alpha参数数组 [alpha_1, ..., alpha_K]。
        """
        if not (0 <= agent_id < self.num_spacecraft):
            raise IndexError(f"智能体ID {agent_id} 超出范围 (0 至 {self.num_spacecraft-1})。")
        if self.num_tasks == 0: # 如果环境中没有任务
             if task_idx_0_based == 0 : # 允许查询索引0如果num_tasks也是0，但应返回空
                 return np.array([])
             else: # 如果查询非0索引但没任务，则明确错误
                 raise IndexError(f"任务索引 {task_idx_0_based} 超出范围，因为当前任务数量为0。")

        if not (0 <= task_idx_0_based < self.num_tasks):
            raise IndexError(f"任务索引 {task_idx_0_based} 超出范围 (0 至 {self.num_tasks-1})。")
        return self.alpha_parameters[agent_id, task_idx_0_based, :].copy()

    def get_expected_belief_distribution(self, agent_id: int, task_idx_0_based: int) -> np.ndarray:
        """
        计算期望的信念分布 (即每种类型的概率)。
        公式: beta_i,k^j = alpha_i,k^j / sum(alpha_i,k'^j for k' in K)。
        这对应用户初稿 (source 8135) 和标准的狄利克雷均值。
        """
        # 如果系统中没有定义任务类型，则返回空数组或根据情况处理
        if self.num_task_types == 0:
            return np.array([])
        # 如果系统中没有任务，则返回关于类型的均匀先验（因为没有特定任务的alpha计数）
        if self.num_tasks == 0:
             return np.full(self.num_task_types, 1.0 / self.num_task_types)

        alphas = self.get_agent_task_belief_parameters(agent_id, task_idx_0_based)
        if alphas.size == 0: # 从 get_agent_task_belief_parameters 返回，如果 num_task_types 为0
            return np.array([])

        sum_alphas = np.sum(alphas)
        if sum_alphas == 0: # 通常不应发生，因为伪计数初始化为正
            # print(f"警告: 智能体 {agent_id} 对任务索引 {task_idx_0_based} 的Alpha参数总和为零。返回均匀分布。")
            return np.full(self.num_task_types, 1.0 / self.num_task_types)
        return alphas / sum_alphas

    def update_beliefs_from_aggregated_observations(self,
                                                    aggregated_observation_counts_for_tasks: np.ndarray):
        """
        基于所有任务的全局聚合观测计数矩阵，更新所有智能体的信念 (alpha参数)。
        这反映了信息融合，所有智能体从集体观测中受益。

        Args:
            aggregated_observation_counts_for_tasks (np.ndarray):
                形状为 (num_actual_tasks, num_task_types) 的矩阵，表示在本轮中，
                每个任务 (0索引) 被所有航天器观测为各种类型的总次数。
                这对应于 Huo et al. 中的 sum(eta_ijk) (Eq. 12, source 7602) 或用户初稿中的 Delta_A (source 8136)。
        """
        if self.num_tasks == 0: # 如果没有任务，则无法更新信念
            # print("警告 (BeliefManager): 尝试更新信念，但系统中没有任务。")
            return

        expected_shape = (self.num_tasks, self.num_task_types)
        if aggregated_observation_counts_for_tasks.shape != expected_shape:
            raise ValueError(f"聚合观测计数形状不匹配。期望形状 {expected_shape}，"
                             f"实际为 {aggregated_observation_counts_for_tasks.shape}")
        if np.any(aggregated_observation_counts_for_tasks < 0):
            raise ValueError("观测计数不能为负。")

        # 将新的观测计数加到所有航天器的alpha参数上。
        # 这确保了所有智能体共享相同的更新知识库，
        # 从而如Huo et al.所述，信念趋于收敛。 (source 7608, 7613)
        # 使用广播机制：alpha_parameters 的形状是 (num_spacecraft, num_tasks, num_task_types)
        # aggregated_observation_counts_for_tasks 的形状是 (num_tasks, num_task_types)
        # NumPy的广播会将后者加到前者的每个 "num_spacecraft" 切片上。
        self.alpha_parameters += aggregated_observation_counts_for_tasks # NumPy broadcasting

    def update_specific_agent_task_belief(self, agent_id: int, task_idx_0_based: int,
                                          observed_type_idx: int, count: int = 1):
        """
        基于特定智能体的直接观测，更新其对特定任务的信念。
        这更多用于没有全局信息融合的场景，或者用于单个智能体的独立观测处理。

        Args:
            agent_id (int): 进行观测的航天器ID。
            task_idx_0_based (int): 被观测任务的0索引ID。
            observed_type_idx (int): 被观测到的任务类型的索引。
            count (int): 该观测发生的次数 (通常为1)。
        """
        if not (0 <= agent_id < self.num_spacecraft):
            raise IndexError(f"智能体ID {agent_id} 超出范围。")
        if self.num_tasks == 0: return # 没有任务可更新
        if not (0 <= task_idx_0_based < self.num_tasks):
            raise IndexError(f"任务索引 {task_idx_0_based} 超出范围。")
        if not (0 <= observed_type_idx < self.num_task_types):
            raise IndexError(f"观测到的任务类型索引 {observed_type_idx} 超出范围。")
        if count < 0:
            raise ValueError("观测计数不能为负。")

        self.alpha_parameters[agent_id, task_idx_0_based, observed_type_idx] += count

    def set_agent_task_belief_parameters(self, agent_id: int, task_idx_0_based: int, alphas: np.ndarray):
        """
        直接设置特定智能体和任务的alpha参数。
        可用于初始化或特定的信念注入场景。

        Args:
            agent_id (int): 航天器ID。
            task_idx_0_based (int): 任务的0索引ID。
            alphas (np.ndarray): 新的alpha参数数组，形状为 (num_task_types,)。
        """
        if not (0 <= agent_id < self.num_spacecraft):
            raise IndexError("智能体ID 超出范围。")
        if self.num_tasks == 0: # 如果没有任务
            if alphas.size == 0 and alphas.shape == (self.num_task_types,): # 允许设置为空的alpha（如果类型也为0）
                return
            else:
                raise ValueError("当系统中没有任务时，不能设置非空的alphas，除非alphas的类型维度也为0。")

        if not (0 <= task_idx_0_based < self.num_tasks):
            raise IndexError(f"任务索引 {task_idx_0_based} 超出范围。")
        if not isinstance(alphas, np.ndarray) or alphas.shape != (self.num_task_types,):
            raise ValueError(f"Alphas 必须是形状为 ({self.num_task_types},) 的numpy数组。")
        if np.any(alphas <= 0): # 狄利克雷参数必须为正
            raise ValueError("Alpha 参数必须为正。")

        self.alpha_parameters[agent_id, task_idx_0_based, :] = alphas.copy()


if __name__ == '__main__':
    # --- 示例用法 ---
    num_s_example = 2
    num_t_example = 2 # 两个实际任务，将被索引为 0 和 1
    num_k_types_example = 3 # 例如：简单，中等，困难三种类型

    print(f"测试 BeliefManager: {num_s_example}个航天器, {num_t_example}个任务, {num_k_types_example}个任务类型")
    belief_manager_example = BeliefManager(num_s_example, num_t_example, num_k_types_example, initial_pseudo_counts=1.0)

    print("\n初始状态:")
    for ag_idx_ex in range(num_s_example):
        for tk_idx_ex in range(num_t_example):
            task_actual_id = tk_idx_ex + 1 # 假设实际任务ID从1开始
            print(f" 航天器 {ag_idx_ex}, 任务 {task_actual_id} (内部索引 {tk_idx_ex}) - Alpha参数: "
                  f"{belief_manager_example.get_agent_task_belief_parameters(ag_idx_ex, tk_idx_ex)}, "
                  f"期望信念: {np.round(belief_manager_example.get_expected_belief_distribution(ag_idx_ex, tk_idx_ex), 3)}")

    # 模拟一轮聚合观测
    # aggregated_obs_round1 的形状: (num_actual_tasks, num_task_types)
    # 任务0 (即实际任务ID 1) 被观测为类型0两次，类型1一次。
    # 任务1 (即实际任务ID 2) 被观测为类型2三次。
    aggregated_obs_example_r1 = np.array([
        [2, 1, 0],  # 任务索引0的观测
        [0, 0, 3]   # 任务索引1的观测
    ])
    
    belief_manager_example.update_beliefs_from_aggregated_observations(aggregated_obs_example_r1)
    print("\n第一轮聚合更新后:")
    for ag_idx_ex in range(num_s_example):
        for tk_idx_ex in range(num_t_example):
            task_actual_id = tk_idx_ex + 1
            print(f" 航天器 {ag_idx_ex}, 任务 {task_actual_id} (内部索引 {tk_idx_ex}) - Alpha参数: "
                  f"{belief_manager_example.get_agent_task_belief_parameters(ag_idx_ex, tk_idx_ex)}, "
                  f"期望信念: {np.round(belief_manager_example.get_expected_belief_distribution(ag_idx_ex, tk_idx_ex), 3)}")
    # 预期结果 (Agent 0, Task 1 (idx 0)): 初始 [1,1,1] + 聚合 [2,1,0] = [3,2,1] -> 信念 [0.5, 0.333, 0.167]
    # 预期结果 (Agent 0, Task 2 (idx 1)): 初始 [1,1,1] + 聚合 [0,0,3] = [1,1,4] -> 信念 [0.167, 0.167, 0.667]
    # Agent 1 的结果应该相同，因为是全局更新。

    # 模拟第二轮聚合观测
    aggregated_obs_example_r2 = np.array([
        [3, 0, 0],  # 任务索引0主要被观测为类型0
        [0, 1, 2]   # 任务索引1的观测分散在类型1和类型2
    ])
    belief_manager_example.update_beliefs_from_aggregated_observations(aggregated_obs_example_r2)
    print("\n第二轮聚合更新后:")
    for ag_idx_ex in range(num_s_example):
        for tk_idx_ex in range(num_t_example):
            task_actual_id = tk_idx_ex + 1
            print(f" 航天器 {ag_idx_ex}, 任务 {task_actual_id} (内部索引 {tk_idx_ex}) - Alpha参数: "
                  f"{belief_manager_example.get_agent_task_belief_parameters(ag_idx_ex, tk_idx_ex)}, "
                  f"期望信念: {np.round(belief_manager_example.get_expected_belief_distribution(ag_idx_ex, tk_idx_ex), 3)}")
    # 预期结果 (Agent 0, Task 1 (idx 0)): 上一轮 [3,2,1] + 聚合 [3,0,0] = [6,2,1] -> 信念 [0.667, 0.222, 0.111]
    # 预期结果 (Agent 0, Task 2 (idx 1)): 上一轮 [1,1,4] + 聚合 [0,1,2] = [1,2,6] -> 信念 [0.111, 0.222, 0.667]