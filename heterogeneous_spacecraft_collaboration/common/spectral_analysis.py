# common/spectral_analysis.py
# 文件路径: common/spectral_analysis.py

import numpy as np
from scipy.linalg import LinAlgError # 用于捕获 eigendecomposition 中的错误
import matplotlib.pyplot as plt
import seaborn as sns # 用于绘制热力图
import os # 用于保存绘图文件

class SpectralAnalyzer:
    """
    对离散时间线性时不变 (LTI) 系统的可控性格拉姆矩阵进行谱分析，
    以找到主要的控制方向（模式）。
    参考 SETS 论文 [cite: 1, 1553] 中的概念，例如输入归一化和谱的可视化。
    """

    def __init__(self, Ad: np.ndarray, Bd: np.ndarray, H: int,
                 u_limits_min: np.ndarray = None, # 用于输入归一化矩阵 S
                 u_limits_max: np.ndarray = None  # 用于输入归一化矩阵 S
                ):
        """
        初始化 SpectralAnalyzer。

        Args:
            Ad (np.ndarray): 离散时间状态矩阵 (n_states x n_states)。
            Bd (np.ndarray): 离散时间输入矩阵 (n_states x n_inputs)。
            H (int): 可控性分析的时域长度 (步数)。
            u_limits_min (np.ndarray, optional): 控制输入的下限 (m_inputs,)。
                                                 用于计算输入归一化因子 S。
            u_limits_max (np.ndarray, optional): 控制输入的上限 (m_inputs,)。
                                                 用于计算输入归一化因子 S。
        """
        if not isinstance(Ad, np.ndarray) or Ad.ndim != 2 or Ad.shape[0] != Ad.shape[1]:
            raise ValueError("Ad 必须是一个二维方阵 numpy 数组。")
        if not isinstance(Bd, np.ndarray) or Bd.ndim != 2 or Bd.shape[0] != Ad.shape[0]:
            raise ValueError("Bd 必须是一个二维 numpy 数组，并且行数与 Ad 的维度匹配。")
        if not isinstance(H, int) or H <= 0:
            raise ValueError("时域长度 H 必须是一个正整数。")

        self.Ad = Ad
        self.Bd = Bd
        self.H = H # 可控性分析的时域
        self.n_states = Ad.shape[0] # 状态维度
        self.m_inputs = Bd.shape[1] # 输入维度

        # 输入归一化矩阵 S (参考 SETS Algorithm 1, line 19 [cite: 317, 1869])
        self.S_normalization = np.eye(self.m_inputs) # 默认为单位矩阵 (无归一化)
        if u_limits_min is not None and u_limits_max is not None:
            if u_limits_min.shape == (self.m_inputs,) and u_limits_max.shape == (self.m_inputs,):
                input_range = u_limits_max - u_limits_min
                if np.any(input_range <= 1e-9): # 避免除以零或非常小的范围
                    print("警告 (SpectralAnalyzer): 部分控制输入的范围为零或过小。归一化矩阵 S 将使用单位矩阵。")
                else:
                    # SETS 论文 Alg1, line 19: S = diag_r{2 / (u_bar_j - u_lower_j)}
                    self.S_normalization = np.diag(2.0 / input_range)
                    # print(f"调试 (SpectralAnalyzer): 使用输入归一化矩阵 S:\n{self.S_normalization}")
            else:
                print("警告 (SpectralAnalyzer): u_limits_min/max 的形状不正确，无法用于S归一化。归一化矩阵 S 将使用单位矩阵。")

        # 计算归一化后的H步可控性矩阵
        self.controllability_matrix_H_normalized = self._compute_controllability_matrix_H_normalized()
        
        # 计算可控性格拉姆矩阵 W_H = C_H_norm * C_H_norm^T
        self.controllability_gramian_H = self.controllability_matrix_H_normalized @ self.controllability_matrix_H_normalized.T

        # 对格拉姆矩阵进行谱分解 (获取特征值和特征向量)
        self.eigenvalues, self.eigenvectors = self._compute_spectral_decomposition()

    def _compute_controllability_matrix_H_normalized(self) -> np.ndarray:
        """
        计算带有输入归一化的H步可控性矩阵:
        CH_norm = [Bd*S, Ad*Bd*S, ..., Ad^(H-1)*Bd*S]。
        SETS 论文 (Algorithm 1, line 20 [cite: 317, 1869]) 构建 C 的方式略有不同，用于时变系统。
        对于LTI系统，每个块 Ad^k * Bd * S 是合适的。
        """
        CH_norm = np.zeros((self.n_states, self.m_inputs * self.H))
        
        # Ad^0 * Bd * S
        current_Ad_power_Bd_S_term = self.Bd @ self.S_normalization
        CH_norm[:, 0 * self.m_inputs:(0 + 1) * self.m_inputs] = current_Ad_power_Bd_S_term

        for i in range(1, self.H): # 从 Ad^1 * Bd * S 开始
            current_Ad_power_Bd_S_term = self.Ad @ current_Ad_power_Bd_S_term # (Ad^i * Bd * S)
            CH_norm[:, i * self.m_inputs:(i + 1) * self.m_inputs] = current_Ad_power_Bd_S_term
        return CH_norm

    def _compute_spectral_decomposition(self) -> tuple[np.ndarray, np.ndarray]:
        """
        计算H步可控性格拉姆矩阵的谱分解 (特征值和特征向量)。
        特征向量代表主要的控制模式/方向。
        特征值代表在这些方向上的控制“能量”或有效性。
        """
        try:
            # np.linalg.eigh 用于对称/厄米矩阵，返回排序的特征值和对应的特征向量
            eigenvalues, eigenvectors = np.linalg.eigh(self.controllability_gramian_H)
            
            # 将特征值和对应的特征向量按特征值大小降序排列
            sorted_indices = np.argsort(eigenvalues)[::-1]
            sorted_eigenvalues = eigenvalues[sorted_indices]
            sorted_eigenvectors = eigenvectors[:, sorted_indices]
            
            return sorted_eigenvalues, sorted_eigenvectors
        except LinAlgError as e:
            print(f"错误 (SpectralAnalyzer): 谱分解失败: {e}")
            return np.array([]), np.array([]) # 返回空数组表示失败

    def get_controllability_gramian(self) -> np.ndarray:
        """返回H步可控性格拉姆矩阵。"""
        return self.controllability_gramian_H.copy()

    def get_control_modes(self) -> tuple[np.ndarray, np.ndarray]:
        """
        返回控制模式 (特征向量) 及其对应的有效性 (特征值)。

        Returns:
            tuple: (eigenvalues, eigenvectors)
                   特征值已按降序排列。
                   特征向量是列向量，与排序后的特征值对应。
        """
        return self.eigenvalues.copy(), self.eigenvectors.copy()

    def get_principal_control_directions(self, num_modes: int = None) -> np.ndarray:
        """
        返回指定数量的主要控制方向 (对应最大特征值的特征向量)。

        Args:
            num_modes (int, optional): 要返回的主要模式数量。
                                       如果为 None 或超出范围，则返回所有计算出的模式。
                                       默认为 None。
        Returns:
            np.ndarray: 列向量为主要控制方向的数组。
        """
        if self.eigenvectors.size == 0: return np.array([]) # 如果没有特征向量
        
        if num_modes is None or not (0 < num_modes <= self.eigenvectors.shape[1]):
            return self.eigenvectors.copy()
        return self.eigenvectors[:, :num_modes].copy()

    def calculate_target_deviation(self, alpha_weights: np.ndarray) -> np.ndarray | None:
        """
        根据给定的模态权重计算期望的状态偏差 delta_s_target。
        delta_s_target = sum alpha_j * sqrt(lambda_j) * v_j
        (参考用户初稿 source 4058 和 SETS 论文 Fig. 6A [cite: 1912, 2685])
        这个偏差是相对于当前状态的。

        Args:
            alpha_weights (np.ndarray): 对应于主要控制模式的权重。
                                        其长度应等于希望使用的模式数量。

        Returns:
            np.ndarray | None: 计算得到的状态偏差向量，如果无法计算则返回 None。
        """
        if self.eigenvalues.size == 0 or self.eigenvectors.size == 0:
            print("警告 (SpectralAnalyzer): 无法计算目标偏差，因为谱信息不可用。")
            return None
        
        num_modes_to_use = len(alpha_weights)
        if num_modes_to_use == 0:
            return np.zeros(self.n_states)
        if num_modes_to_use > len(self.eigenvalues):
            print(f"警告 (SpectralAnalyzer): 请求的alpha_weights数量 ({num_modes_to_use}) "
                  f"超过可用模式数量 ({len(self.eigenvalues)})。将使用所有可用模式。")
            num_modes_to_use = len(self.eigenvalues)
        
        delta_s_target = np.zeros(self.n_states)
        for i in range(num_modes_to_use):
            lambda_i = self.eigenvalues[i]
            v_i = self.eigenvectors[:, i]
            alpha_i = alpha_weights[i]
            
            # 确保 lambda_i 为正以进行 sqrt 运算，非常小的负数可能是数值误差
            if lambda_i < -1e-9: # 容忍非常小的负值
                print(f"警告 (SpectralAnalyzer): 特征值 lambda_{i} = {lambda_i:.2e} 为负。跳过此模式的 sqrt(lambda) 项。")
                term_scale = 0.0
            elif lambda_i < 1e-9: # 如果特征值非常小，sqrt 可能导致数值问题或影响不大
                term_scale = 0.0 # 或者 alpha_i * 0.01 * v_i 之类的小扰动
            else:
                term_scale = np.sqrt(lambda_i)
            
            delta_s_target += alpha_i * term_scale * v_i
            
        return delta_s_target


    def plot_spectrum_heatmap(self, num_modes_to_plot: int = None, state_labels: list = None,
                              title: str = "可控性格拉姆矩阵谱的热力图",
                              filename: str = None, results_dir="results"):
        """
        绘制控制模式 (特征向量) 的热力图。
        灵感来源于 SETS 论文 Fig 2B[cite: 1618, 2391], Fig 6B [cite: 1914, 2687]。

        Args:
            num_modes_to_plot (int, optional): 要显示的主要模式数量。默认为所有计算出的模式中最多8个。
            state_labels (list, optional): 状态向量各维度的标签 (对应热力图的行)。
            title (str, optional): 图表标题。
            filename (str, optional): 如果提供，则将图表保存到此文件名 (不含扩展名)。
            results_dir (str, optional): 保存图表的目录。
        """
        if self.eigenvectors.size == 0:
            print("无特征向量可绘制。谱分解可能失败。")
            return

        if num_modes_to_plot is None:
            num_modes_to_plot = min(self.eigenvectors.shape[1], 8) # 默认最多显示8个模式
        elif num_modes_to_plot <= 0 or num_modes_to_plot > self.eigenvectors.shape[1]:
            num_modes_to_plot = self.eigenvectors.shape[1] # 显示所有可用模式

        modes_to_display = self.get_principal_control_directions(num_modes_to_plot)
        if modes_to_display.size == 0:
            print("没有选择用于显示的模式。")
            return

        if state_labels is None:
            state_labels = [f"状态 {i+1}" for i in range(self.n_states)]
        elif len(state_labels) != self.n_states:
            print(f"警告: state_labels 长度 ({len(state_labels)}) 与状态数量 ({self.n_states}) 不匹配。将使用默认标签。")
            state_labels = [f"状态 {i+1}" for i in range(self.n_states)]

        mode_indices_labels = [f"模式 {i+1}\n(λ={self.eigenvalues[i]:.2e})" for i in range(modes_to_display.shape[1])]

        plt.figure(figsize=(max(8, modes_to_display.shape[1] * 1.2), max(6, self.n_states * 0.8))) # 调整图像大小
        sns.heatmap(modes_to_display, annot=True, fmt=".2f", cmap="viridis_r", # 使用反转的viridis，或 "coolwarm", "icefire"
                    yticklabels=state_labels, xticklabels=mode_indices_labels,
                    linewidths=.5, linecolor='gray', cbar_kws={'label': '模式分量大小 (Mode Component Magnitude)'})
        plt.title(title, fontsize=16, weight='bold')
        plt.ylabel("状态维度 (State Dimension)", fontsize=14)
        plt.xlabel("控制模式 (按特征值降序排列)", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()

        if filename:
            if not os.path.exists(results_dir):
                os.makedirs(results_dir, exist_ok=True)
            full_path = os.path.join(results_dir, f"{filename}.png")
            try:
                plt.savefig(full_path, dpi=300, bbox_inches='tight')
                print(f"谱热力图已保存至: {full_path}")
            except Exception as e:
                print(f"保存谱热力图 '{full_path}' 失败: {e}")
        else:
            plt.show()
        plt.close() # 关闭图像，防止在循环中重复显示


if __name__ == '__main__':
    # --- 示例用法 ---
    try:
        from dynamics import CWDynamics # 假设 dynamics.py 在可访问路径中
    except ImportError:
        # 如果直接运行此文件且 dynamics.py 不在标准路径中，使用一个模拟类
        class CWDynamics:
            def __init__(self, n, ts):
                self.n, self.ts = n, ts
                # 简化版 Euler 离散化 Ad, Bd 用于测试
                self.A_cont = np.array([[0,0,1,0], [0,0,0,1], [3*n*n,0,0,2*n], [0,0,-2*n,0]])
                self.B_cont = np.array([[0,0],[0,0],[1,0],[0,1]])
                self.Ad = np.eye(4) + self.A_cont * ts
                self.Bd = self.B_cont * ts
                self.state_dim = 4; self.control_dim = 2
            def get_discrete_matrices(self): return self.Ad, self.Bd
        print("警告: SpectralAnalyzer 测试使用的是简化的 CWDynamics。")

    n_orbital_sa_test = 0.00113 # 近地轨道近似角速度 (rad/s)
    sampling_time_sa_test = 1.0  # 采样时间
    horizon_H_sa_test = 10       # 可控性分析时域长度

    # 控制输入限制示例 (用于归一化)
    u_max_sa_test = 0.1 # m/s^2
    u_min_limits_sa_test = np.array([-u_max_sa_test, -u_max_sa_test])
    u_max_limits_sa_test = np.array([u_max_sa_test, u_max_sa_test])

    # 创建动力学模型实例
    cw_dyn_sa_test = CWDynamics(n=n_orbital_sa_test, ts=sampling_time_sa_test)
    Ad_matrix_sa_test, Bd_matrix_sa_test = cw_dyn_sa_test.get_discrete_matrices()

    print("用于谱分析的 Ad 矩阵:\n", np.round(Ad_matrix_sa_test, 4))
    print("\n用于谱分析的 Bd 矩阵:\n", np.round(Bd_matrix_sa_test, 4))

    # 创建 SpectralAnalyzer 实例，包含输入归一化
    analyzer_instance_test = SpectralAnalyzer(Ad=Ad_matrix_sa_test, Bd=Bd_matrix_sa_test, H=horizon_H_sa_test,
                                              u_limits_min=u_min_limits_sa_test, u_limits_max=u_max_limits_sa_test)
    
    print("\n输入归一化矩阵 S:\n", np.round(analyzer_instance_test.S_normalization, 4))

    eigenvalues_sorted_test, eigenvectors_sorted_test = analyzer_instance_test.get_control_modes()
    print("\n排序后的特征值 (有效性):\n", np.round(eigenvalues_sorted_test, 4))
    # print("\n对应的特征向量 (控制模式 - 列向量):\n", np.round(eigenvectors_sorted_test, 4)) # 可能较长，选择性打印

    num_principal_to_show = min(analyzer_instance_test.n_states, 4) # 最多显示4个或状态数个模式
    principal_directions_test = analyzer_instance_test.get_principal_control_directions(num_modes=num_principal_to_show)
    print(f"\n前 {num_principal_to_show} 个主要控制方向:\n", np.round(principal_directions_test, 4))

    # 绘制热力图并保存
    cw_state_labels_test = ["x 位置 (m)", "y 位置 (m)", "vx (m/s)", "vy (m/s)"]
    analyzer_instance_test.plot_spectrum_heatmap(num_modes_to_plot=num_principal_to_show,
                                                 state_labels=cw_state_labels_test,
                                                 title="CW系统可控性谱热力图 (归一化输入)",
                                                 filename="spectral_heatmap_cw_normalized_input_example",
                                                 results_dir="results_test_spectral") # 指定保存目录

    # 概念性使用：计算期望状态偏差 delta_s_target
    # 假设高级决策模块给出了前2个主要模式的权重
    if len(eigenvalues_sorted_test) >= 2:
        alpha_weights_test = np.array([0.7, -0.3]) # 例如：主要沿模式1正向，次要沿模式2反向
        
        delta_s_target_conceptual_test = analyzer_instance_test.calculate_target_deviation(alpha_weights_test[:2])
        
        if delta_s_target_conceptual_test is not None:
            print(f"\n基于权重 {alpha_weights_test[:2]} 的概念性目标状态偏差 (相对于当前状态):\n",
                  np.round(delta_s_target_conceptual_test, 4))
            # 这个 delta_s_target_conceptual_test 可以被加到当前状态，形成MPC的绝对目标状态
            # mpc_target_abs = current_state + delta_s_target_conceptual_test