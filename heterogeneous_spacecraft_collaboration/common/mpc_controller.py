# common/mpc_controller.py
# 文件路径: common/mpc_controller.py

import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import time # 用于潜在的计时/调试
import matplotlib.pyplot as plt 
class MPCController:
    """
    针对线性系统（例如，离散化的CW动力学）的模型预测控制器。
    在有限视界内最小化二次代价函数。
    参考用户初稿中的MPC问题表述 (source 3353, 3354)。
    """

    def __init__(self,
                 Ad: np.ndarray,         # 离散时间状态矩阵 (n_states x n_states)
                 Bd: np.ndarray,         # 离散时间输入矩阵 (n_states x n_inputs)
                 Q: np.ndarray,          # 状态权重矩阵 (n_states x n_states)
                 R: np.ndarray,          # 输入权重矩阵 (n_inputs x n_inputs)
                 Q_terminal: np.ndarray, # 终端状态权重矩阵 (n_states x n_states)
                 N: int,                 # 预测视界 (步数)
                 u_min: np.ndarray,      # 控制输入下限 (m_inputs,)
                 u_max: np.ndarray,      # 控制输入上限 (m_inputs,)
                 s_min: np.ndarray = None, # 可选的状态下限 (n_states,)
                 s_max: np.ndarray = None, # 可选的状态上限 (n_states,)
                 solver_options: dict = None # scipy.optimize.minimize 求解器选项
                ):
        """
        初始化MPC控制器。
        """
        self.Ad, self.Bd = Ad, Bd
        self.Q, self.R = Q, R
        self.Q_terminal = Q_terminal if Q_terminal is not None else Q # 如果未指定终端权重，则默认为Q
        self.N = N # 预测视界
        self.n_states, self.m_inputs = Ad.shape[0], Bd.shape[1] # 状态和控制维度

        if u_min.shape != (self.m_inputs,) or u_max.shape != (self.m_inputs,):
            raise ValueError(f"控制输入边界 u_min 和 u_max 的维度必须为 ({self.m_inputs},)")
        # 将单步的输入边界扩展到整个视界，用于优化器
        self.u_min_flat = np.tile(u_min, N)
        self.u_max_flat = np.tile(u_max, N)

        self.s_min, self.s_max = s_min, s_max
        if s_min is not None and (s_min.shape != (self.n_states,) or s_max.shape != (self.n_states,)):
            raise ValueError(f"如果提供状态边界 s_min 和 s_max，其维度必须为 ({self.n_states},)")

        # 默认求解器选项
        self.solver_options = solver_options if solver_options is not None else \
                              {'maxiter': 100, 'disp': False, 'ftol': 1e-6, 'iprint': -1}


    def _objective_function(self, u_flat: np.ndarray, current_state: np.ndarray,
                            s_ref_trajectory: np.ndarray) -> float:
        """
        计算给定控制输入序列的MPC代价。
        代价函数形式参考用户初稿 (source 3353)。

        Args:
            u_flat (np.ndarray): 展平的控制输入序列 [u0_0,...,u0_m-1, u1_0,..., uN-1_m-1]。维度 (N * m_inputs,).
            current_state (np.ndarray): 预测的初始状态。维度 (n_states,).
            s_ref_trajectory (np.ndarray): 参考状态轨迹 [s_ref_0, ..., s_ref_N]。维度 (N+1, n_states).
                                         s_ref_trajectory[k] 是状态 s_k 的参考。
                                         注意：代价通常从 s_1 (由 u_0 产生) 开始计算。
        Returns:
            float: 总代价。
        """
        cost = 0.0
        s_k = current_state.copy() # 当前预测序列的起始状态 (即 s_0)
        u_sequence = u_flat.reshape(self.N, self.m_inputs) # 将扁平化的u转换为 (N, m_inputs)

        # 循环 N 次，对应控制输入 u_0, u_1, ..., u_{N-1}
        # 这些控制输入会产生状态 s_1, s_2, ..., s_N
        for k in range(self.N):
            u_k = u_sequence[k, :] # 当前周期的控制输入 u_k
            
            # 预测下一状态 s_{k+1}
            s_k_plus_1 = self.Ad @ s_k + self.Bd @ u_k
            
            # 获取 s_{k+1} 的参考状态 s_ref_{k+1}
            s_ref_k_plus_1 = s_ref_trajectory[k+1, :] 
            
            state_error_k_plus_1 = s_k_plus_1 - s_ref_k_plus_1 # s_{k+1} 的状态误差
            
            # 根据是否为终端状态，选择不同的状态权重矩阵
            if k < self.N - 1: # 对于中间状态 s_1, ..., s_{N-1}
                cost += state_error_k_plus_1.T @ self.Q @ state_error_k_plus_1
            else: # 对于终端状态 s_N (即 k = N-1 时产生的 s_k_plus_1)
                cost += state_error_k_plus_1.T @ self.Q_terminal @ state_error_k_plus_1
            
            cost += u_k.T @ self.R @ u_k # 控制代价
            
            s_k = s_k_plus_1 # 更新当前状态为下一状态，用于下一次迭代
            
        return cost

    def _predict_trajectory(self, u_flat: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """辅助函数：根据给定的初始状态和控制输入序列预测状态轨迹。"""
        s_trajectory_predicted = np.zeros((self.N + 1, self.n_states)) # 存储 s_0, s_1, ..., s_N
        s_trajectory_predicted[0, :] = current_state
        u_sequence = u_flat.reshape(self.N, self.m_inputs)
        for k in range(self.N):
            s_trajectory_predicted[k+1, :] = self.Ad @ s_trajectory_predicted[k, :] + self.Bd @ u_sequence[k, :]
        return s_trajectory_predicted

    def _state_constraint_function(self, u_flat: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """
        用于 scipy.optimize.NonlinearConstraint 的函数。
        返回 g(u) 的值，其中约束为 g_lower <= g(u) <= g_upper。
        这里，g(u) 是预测的状态轨迹 S = [s_1^T, s_2^T, ..., s_N^T]^T (展平)。
        """
        predicted_states_over_horizon = self._predict_trajectory(u_flat, current_state)
        # 我们约束的是由 u_0, ..., u_{N-1} 产生的状态 s_1, ..., s_N
        return predicted_states_over_horizon[1:, :].flatten() # 展平 s_1 到 s_N


    def compute_control_sequence(self,
                                 current_state: np.ndarray,
                                 target_state_final_abs: np.ndarray, # 视界末端的绝对目标状态
                                 initial_guess_u_sequence: np.ndarray = None
                                ) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
        """
        计算最优控制输入序列，目标是使系统在视界末端达到 target_state_final_abs。
        参考轨迹的构建方式会影响MPC的行为。

        Args:
            current_state (np.ndarray): 当前状态 [x, y, vx, vy]。
            target_state_final_abs (np.ndarray): 视界末端 N 的期望绝对状态。
                                                 这个值通常由更高级别的规划器（例如结合谱分析的主动推理）
                                                 通过 current_state + delta_s_target 计算得出。
            initial_guess_u_sequence (np.ndarray, optional): 控制序列的初始猜测值。
                                                            形状 (N, m_inputs)。默认为零。

        Returns:
            tuple: (第一个最优控制输入, 完整的最优控制序列, 优化后的代价值)
                   如果优化失败，则返回 (None, None, np.inf)。
        """
        if target_state_final_abs.shape != (self.n_states,):
            raise ValueError(f"绝对目标状态 target_state_final_abs 的维度错误。")

        # 构建参考轨迹 s_ref_trajectory (维度 N+1, n_states)
        # s_ref_trajectory[k] 是状态 s_k 的参考。代价函数评估 s_1, ..., s_N 的误差。
        s_ref_trajectory = np.zeros((self.N + 1, self.n_states))
        # s_ref_trajectory[0] 是 s_0 (即 current_state) 的参考。通常，我们不惩罚初始状态的误差。
        s_ref_trajectory[0, :] = current_state
        # 对于视界内的所有未来步骤 (s_1 到 s_N)，将参考设置为最终的绝对目标状态。
        # 这会驱使系统尽快朝向 target_state_final_abs 移动。
        for k_ref in range(1, self.N + 1):
            s_ref_trajectory[k_ref, :] = target_state_final_abs

        if initial_guess_u_sequence is None:
            u_initial_guess_flat = np.zeros(self.N * self.m_inputs)
        else:
            if initial_guess_u_sequence.shape != (self.N, self.m_inputs):
                raise ValueError("控制序列初始猜测的维度错误。")
            u_initial_guess_flat = initial_guess_u_sequence.flatten()

        # 控制输入的边界
        bounds = Bounds(self.u_min_flat, self.u_max_flat)
        
        # 状态约束 (如果定义了)
        active_constraints = []
        if self.s_min is not None and self.s_max is not None:
            # 为状态轨迹 s_1, ..., s_N 定义上下界
            s_lower_bounds_flat = np.tile(self.s_min, self.N)
            s_upper_bounds_flat = np.tile(self.s_max, self.N)
            state_nlc = NonlinearConstraint(
                fun=lambda u_var: self._state_constraint_function(u_var, current_state),
                lb=s_lower_bounds_flat,
                ub=s_upper_bounds_flat
            )
            active_constraints.append(state_nlc)

        # 执行优化
        # start_opt_time_debug = time.time() # 用于调试优化时间
        solution = minimize(self._objective_function,
                              u_initial_guess_flat,
                              args=(current_state, s_ref_trajectory),
                              method='SLSQP', # 序列最小二乘规划，适用于带约束的非线性优化
                              bounds=bounds,
                              constraints=active_constraints if active_constraints else (), # 如果列表为空则传递空元组
                              options=self.solver_options)
        # optimization_duration_debug = time.time() - start_opt_time_debug
        # print(f"MPC求解耗时: {optimization_duration_debug:.4f}s, 成功: {solution.success}, 迭代次数: {solution.nit}")

        if not solution.success:
            # print(f"MPC 优化失败: {solution.message}。成本: {solution.fun}")
            # 如果优化失败，可以返回一个安全的控制输入，例如初始猜测或零控制
            optimal_u_sequence_fallback = u_initial_guess_flat.reshape(self.N, self.m_inputs)
            return optimal_u_sequence_fallback[0, :], optimal_u_sequence_fallback, np.inf
        
        optimal_u_sequence_res = solution.x.reshape(self.N, self.m_inputs)
        first_optimal_input_res = optimal_u_sequence_res[0, :]
        optimized_cost_res = solution.fun

        return first_optimal_input_res, optimal_u_sequence_res, optimized_cost_res

if __name__ == '__main__':
    try: from dynamics import CWDynamics # 假设 dynamics.py 在同一目录或可访问
    except ImportError:
        class CWDynamics: # 用于MPC测试的最小化模拟
            def __init__(self, n, ts): self.Ad = np.eye(4); self.Bd = np.zeros((4,2)); self.state_dim=4;self.control_dim=2
            def get_discrete_matrices(self): return self.Ad, self.Bd
            def step(self,s,u): return s # 模拟的step，实际应为 Ad@s + Bd@u
        print("警告: MPC测试使用的是模拟的CWDynamics。")

    # 系统参数
    n_orbital_test = 0.00113  # rad/s
    sampling_time_test = 1.0  # s
    horizon_N_test = 20       # MPC预测视界，适当增加以获得更平滑的控制

    # CW动力学模型
    cw_dyn_test = CWDynamics(n=n_orbital_test, ts=sampling_time_test)
    Ad_test, Bd_test = cw_dyn_test.get_discrete_matrices()

    # MPC代价矩阵
    Q_mat_test = np.diag([10.0, 10.0, 0.5, 0.5])  # 更大程度地惩罚位置误差，适度惩罚速度误差
    R_mat_test = np.diag([0.01, 0.01])           # 对控制输入的惩罚较小，允许更大的控制量
    Q_terminal_mat_test = Q_mat_test * 20       # 终端状态权重显著增大，强调最终到达目标

    # 控制输入边界
    u_max_val_test = 0.2 # m/s^2 ; 增大了最大加速度以允许更快移动
    u_min_arr_test = np.array([-u_max_val_test] * cw_dyn_test.control_dim)
    u_max_arr_test = np.array([u_max_val_test] * cw_dyn_test.control_dim)

    # 可选的状态约束 (例如，保持在某个区域内，速度不超过某个值)
    s_min_bounds_test = np.array([-500, -500, -5.0, -5.0]) # x, y, vx, vy 下限
    s_max_bounds_test = np.array([500,  500,  5.0,  5.0]) # x, y, vx, vy 上限

    mpc_test_instance = MPCController(Ad_test, Bd_test, Q_mat_test, R_mat_test, Q_terminal_mat_test,
                                      horizon_N_test, u_min_arr_test, u_max_arr_test,
                                      s_min=s_min_bounds_test, s_max=s_max_bounds_test)

    # 仿真设置
    s_current_test = np.array([150.0, 80.0, 0.0, 0.0])  # 初始状态 [x, y, vx, vy]
    
    # 场景1: 绝对目标状态 (例如，回到原点)
    # s_target_final_abs_test = np.array([0.0, 0.0, 0.0, 0.0])

    # 场景2: 目标状态是相对于当前状态的一个期望偏差 (更符合您的需求)
    # 假设高级决策模块（如谱分析引导的主动推理）给出了一个期望的状态变化量 delta_s
    delta_s_target_test = np.array([-150.0, -80.0, 0.0, 0.0]) # 期望从当前位置移动到原点，速度为零
    s_target_final_abs_test = s_current_test + delta_s_target_test
    print(f"初始状态: {s_current_test}")
    print(f"期望的状态偏差 (delta_s): {delta_s_target_test}")
    print(f"MPC的视界末端绝对目标状态: {s_target_final_abs_test}")


    num_sim_steps_test = 60
    actual_trajectory_test = [s_current_test.copy()]
    controls_applied_list_test = []
    initial_guess_u_mpc = np.zeros((horizon_N_test, cw_dyn_test.control_dim)) # MPC控制序列的初始猜测

    for i in range(num_sim_steps_test):
        # print(f"\n--- MPC仿真第 {i+1} 步 ---")
        # print(f"当前状态: {np.round(s_current_test,3)}")

        # 如果 delta_s_target 是动态变化的 (例如，每隔几个MPC步由高级规划器更新一次)
        # 则 s_target_final_abs_test 需要在每次调用MPC前重新计算：
        # current_delta_s = get_delta_s_from_high_level_planner(s_current_test)
        # s_target_final_abs_test = s_current_test + current_delta_s
        # 对于这个独立测试，我们假设 s_target_final_abs_test (即原点) 是固定的。

        u_first_applied, u_sequence_optimal, cost_val = mpc_test_instance.compute_control_sequence(
            s_current_test, s_target_final_abs_test, initial_guess_u_mpc
        )

        if u_first_applied is None: # 优化失败
            print(f"步骤 {i+1}: MPC优化失败，使用零控制。")
            u_first_applied = np.zeros(cw_dyn_test.control_dim)
            # 重置初始猜测，或者使用上一个成功的序列（如果适用）
            initial_guess_u_mpc = np.zeros((horizon_N_test, cw_dyn_test.control_dim))
        else:
            # 使用上一个优化序列进行温启动 (移除第一个，末尾重复最后一个)
            initial_guess_u_mpc = np.vstack([u_sequence_optimal[1:,:], u_sequence_optimal[-1,:]]) 
        
        # print(f"施加的最优控制: {np.round(u_first_applied,5)}, 成本: {cost_val:.4f}")
        
        # 应用控制并更新状态 (在真实仿真中，这是由 env.step 完成的)
        s_next_test = cw_dyn_test.step(s_current_test, u_first_applied)
        s_current_test = s_next_test
        
        actual_trajectory_test.append(s_current_test.copy())
        controls_applied_list_test.append(u_first_applied.copy())

        if np.linalg.norm(s_current_test - s_target_final_abs_test) < 1.0: # 收敛判断阈值
            print(f"\n在第 {i+1} 步到达目标!")
            break
    else: # 如果循环正常结束（未break）
        print(f"\n仿真结束。最终状态: {np.round(s_current_test,3)}")

    # 绘图 (确保 matplotlib 可用)
    try:
        traj_array_test = np.array(actual_trajectory_test)
        ctrl_array_test = np.array(controls_applied_list_test)
        time_axis_test = np.arange(traj_array_test.shape[0]) * sampling_time_test

        fig_test, axs_test = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        fig_test.suptitle("MPC 控制器独立测试", fontsize=16)
        
        axs_test[0].plot(traj_array_test[:, 0], traj_array_test[:, 1], 'b.-', label='轨迹', linewidth=1.5)
        axs_test[0].plot(s_target_final_abs_test[0], s_target_final_abs_test[1], 'rx', markersize=12, mew=2, label='目标点')
        axs_test[0].plot(actual_trajectory_test[0][0], actual_trajectory_test[0][1], 'go', markersize=10, label='起始点')
        axs_test[0].set_xlabel("X 位置 (m)"); axs_test[0].set_ylabel("Y 位置 (m)"); axs_test[0].set_title("航天器轨迹")
        axs_test[0].legend(); axs_test[0].grid(True); axs_test[0].axis('equal')

        if ctrl_array_test.size > 0:
            axs_test[1].plot(time_axis_test[:-1], ctrl_array_test[:, 0], 'r.-', label='$a_x$', linewidth=1.5)
            axs_test[1].plot(time_axis_test[:-1], ctrl_array_test[:, 1], 'g.--', label='$a_y$', linewidth=1.5)
        axs_test[1].set_ylabel("加速度 (m/s²)"); axs_test[1].set_title("控制输入 (加速度)")
        axs_test[1].legend(); axs_test[1].grid(True)

        axs_test[2].plot(time_axis_test, traj_array_test[:, 2], 'r.-', label='$v_x$', linewidth=1.5)
        axs_test[2].plot(time_axis_test, traj_array_test[:, 3], 'g.--', label='$v_y$', linewidth=1.5)
        axs_test[2].set_xlabel("时间 (s)"); axs_test[2].set_ylabel("速度 (m/s)"); axs_test[2].set_title("速度分量")
        axs_test[2].legend(); axs_test[2].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    except ImportError:
        print("Matplotlib 未找到，跳过MPC测试的绘图。")
    except Exception as e_plot_test:
        print(f"绘图过程中发生错误: {e_plot_test}")