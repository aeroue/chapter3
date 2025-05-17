# simulation/visualizer.py

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import numpy as np
import seaborn as sns
import os

try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'Arial Unicode MS', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"Warning: Could not set Chinese font properties for Matplotlib: {e}")

class SimulationVisualizer:
    def __init__(self, env, num_task_types: int, task_properties_by_type: dict,
                 fixed_xlim=None, fixed_ylim=None, main_title_prefix=""):
        self.env = env
        self.num_task_types = num_task_types
        self.task_properties_by_type = task_properties_by_type if isinstance(task_properties_by_type, dict) else {}

        self.fig_main, self.ax_main = plt.subplots(figsize=(12, 9))

        self.fixed_xlim = fixed_xlim
        self.fixed_ylim = fixed_ylim
        if self.fixed_xlim: self.ax_main.set_xlim(self.fixed_xlim)
        if self.fixed_ylim: self.ax_main.set_ylim(self.fixed_ylim)
        self.main_title_prefix = main_title_prefix

        self.agent_trails = {}
        self.agent_markers = {}
        self.agent_id_texts = {}
        self.task_markers = {}
        self.task_id_texts = {}
        self.coalition_lines = {}
        self.belief_bar_ax = None
        self.belief_bars = None
        self._belief_plot_config = {}

        self.history = {
            "time": [],
            "agent_states": {ag_id: [] for ag_id in self.env.agents},
            "agent_assignments": {ag_id: [] for ag_id in self.env.agents},
            "agent_controls": {ag_id: [] for ag_id in self.env.agents},
            "agent_task_beliefs_strong": {
                ag_id: {task_obj.id: [] for task_obj in self.env.tasks.values()}
                for ag_id in self.env.agents
            },
            "aif_system_beliefs_weak": {ag_id: [] for ag_id in self.env.agents},
            "belief_for_dynamic_plot": []
        }

        num_unique_agents = len(self.env.agents) if self.env.agents else 1
        self.agent_colors = sns.color_palette("husl", n_colors=num_unique_agents)
        # Define distinct linestyles for agents if needed, e.g., for velocity/accel plots
        self.agent_linestyles = ['-', '--', ':', '-.'] * ( (num_unique_agents // 4) + 1)


        all_revenues = []
        for props_dict in self.task_properties_by_type.values():
            if isinstance(props_dict, dict) and "revenue" in props_dict:
                all_revenues.append(props_dict["revenue"])
        if all_revenues:
            self.min_rev_val = min(all_revenues); self.max_rev_val = max(all_revenues)
        else:
            self.min_rev_val = 0; self.max_rev_val = 1

        if abs(self.max_rev_val - self.min_rev_val) < 1e-6:
            self.norm_for_task_color = mcolors.Normalize(vmin=self.min_rev_val - 0.5, vmax=self.max_rev_val + 0.5)
        else:
            padding = (self.max_rev_val - self.min_rev_val) * 0.1
            self.norm_for_task_color = mcolors.Normalize(
                vmin=self.min_rev_val - padding, vmax=self.max_rev_val + padding
            )
        self.task_type_colormap = plt.cm.get_cmap('YlOrRd')
        self._setup_main_plot_elements()

    def _setup_main_plot_elements(self):
        self.ax_main.set_xlabel("X 位置 (m)", fontsize=14, weight='bold')
        self.ax_main.set_ylabel("Y 位置 (m)", fontsize=14, weight='bold')
        self.ax_main.grid(True, linestyle='--', alpha=0.6)
        self.ax_main.axis('equal')
        self.ax_main.tick_params(axis='both', which='major', labelsize=12)

        for task_id, task_obj in self.env.tasks.items():
            task_color = 'grey'
            props_for_this_type = self.task_properties_by_type.get(task_obj.true_type, {})
            type_name_display = f"类型 {task_obj.true_type}"
            if isinstance(task_obj.properties, dict):
                type_name_display = task_obj.properties.get("name", type_name_display)
            elif isinstance(props_for_this_type, dict):
                type_name_display = props_for_this_type.get("name", type_name_display)
            task_label = f"任务 {task_id} ({type_name_display})"
            revenue = props_for_this_type.get("revenue") if isinstance(props_for_this_type,dict) else None
            if revenue is not None:
                task_color = self.task_type_colormap(self.norm_for_task_color(revenue))

            self.task_markers[task_id] = self.ax_main.plot(
                task_obj.position[0], task_obj.position[1],
                marker='P', markersize=14, linestyle='None',
                markeredgecolor='black', markerfacecolor=task_color, mew=1.5, # Thicker edge
                label=task_label
            )[0]
            self.task_id_texts[task_id] = self.ax_main.text(
                task_obj.position[0], task_obj.position[1] + 1.0, f"T{task_id}", # Increased offset
                fontsize=11, ha='center', weight='bold'
            )

        agent_keys = list(self.env.agents.keys())
        for i, agent_id in enumerate(agent_keys):
            color_idx = i % len(self.agent_colors)
            color = self.agent_colors[color_idx]
            self.agent_trails[agent_id], = self.ax_main.plot([], [], '-', color=color, alpha=0.5, linewidth=2.5) # Thicker trail
            self.agent_markers[agent_id] = self.ax_main.plot(
                [], [], marker='o', markersize=10, linestyle='None',
                color=color, markeredgecolor='black', mew=1.0, label=f"航天器 {agent_id}"
            )[0]
            self.agent_id_texts[agent_id] = self.ax_main.text(0, 0, f"S{agent_id}", fontsize=10, ha='center', va='bottom', weight='bold', color=color)
            self.coalition_lines[agent_id], = self.ax_main.plot([], [], '--', color=color, alpha=0.6, linewidth=1.5) # Thicker dash

        if self.env.tasks or self.env.agents:
            handles, labels = self.ax_main.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            self.ax_main.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=10, frameon=True, facecolor='white', framealpha=0.8, borderaxespad=0.)


    def setup_dynamic_belief_plot(self, agent_id_to_show: int, task_id_to_show: int):
        if not (self.env.agents and agent_id_to_show in self.env.agents and \
                self.env.tasks and task_id_to_show in self.env.tasks and \
                self.num_task_types > 0):
            self._belief_plot_config = {}
            if self.belief_bar_ax: self.belief_bar_ax.set_visible(False)
            return

        self._belief_plot_config = {"agent_id": agent_id_to_show, "task_id": task_id_to_show}
        if self.belief_bar_ax is None:
            self.belief_bar_ax = self.fig_main.add_axes([0.02, 0.02, 0.25, 0.2]) # Repositioned to bottom-left
        
        self.belief_bar_ax.clear()
        self.belief_bar_ax.set_visible(True)
        self.belief_bar_ax.set_ylim(0, 1.05)
        self.belief_bar_ax.set_xticks(np.arange(self.num_task_types))
        
        type_labels = []
        for i in range(self.num_task_types):
            type_master_props = self.task_properties_by_type.get(i, {})
            type_labels.append(type_master_props.get("name", f"类型{i}"))

        self.belief_bar_ax.set_xticklabels(type_labels, rotation=30, ha="right", fontsize=9)
        self.belief_bar_ax.set_ylabel("信念概率", fontsize=10)
        self.belief_bar_ax.set_title(f"S{agent_id_to_show} 对 T{task_id_to_show} 的信念", fontsize=10)
        self.belief_bar_ax.tick_params(axis='both', which='major', labelsize=9)
        bar_colors = sns.color_palette("viridis", self.num_task_types) # Changed palette
        self.belief_bars = self.belief_bar_ax.bar(
            np.arange(self.num_task_types),
            [1.0/self.num_task_types if self.num_task_types > 0 else 0] * self.num_task_types,
            color=bar_colors, edgecolor='black', linewidth=0.7
        )
        self.belief_bar_ax.grid(True, linestyle=':', alpha=0.5, axis='y')


    def record_state(self, agent_controls: dict,
                     strong_belief_manager=None,
                     aif_agents_weak=None):
        self.history["time"].append(self.env.current_time)
        for agent_id, agent_obj in self.env.agents.items():
            self.history["agent_states"][agent_id].append(agent_obj.state.copy())
            self.history["agent_assignments"][agent_id].append(agent_obj.assigned_task_id)
            self.history["agent_controls"][agent_id].append(agent_controls.get(agent_id, np.zeros(agent_obj.dynamics_model.control_dim)).copy())

            if strong_belief_manager:
                if agent_id not in self.history["agent_task_beliefs_strong"]: # Initialize if new agent
                    self.history["agent_task_beliefs_strong"][agent_id] = {task_obj.id: [] for task_obj in self.env.tasks.values()}
                for task_obj in self.env.tasks.values():
                    task_idx_for_bm = task_obj.id - 1
                    if 0 <= task_idx_for_bm < strong_belief_manager.num_tasks:
                        belief_dist = strong_belief_manager.get_expected_belief_distribution(agent_id, task_idx_for_bm)
                        if task_obj.id not in self.history["agent_task_beliefs_strong"][agent_id]:
                             self.history["agent_task_beliefs_strong"][agent_id][task_obj.id] = []
                        self.history["agent_task_beliefs_strong"][agent_id][task_obj.id].append(belief_dist)
            
            if aif_agents_weak and agent_id in aif_agents_weak:
                 self.history["aif_system_beliefs_weak"][agent_id].append(aif_agents_weak[agent_id].get_current_belief())

        belief_dist_for_dyn_plot = None
        if hasattr(self, '_belief_plot_config') and self._belief_plot_config: # Check if dict is not empty
            cfg = self._belief_plot_config
            target_agent_id = cfg.get("agent_id")
            target_task_id_actual = cfg.get("task_id")

            if strong_belief_manager and target_agent_id in self.env.agents and \
               target_task_id_actual is not None:
                task_idx_for_bm_dyn = target_task_id_actual - 1
                if 0 <= task_idx_for_bm_dyn < strong_belief_manager.num_tasks:
                    belief_dist_for_dyn_plot = strong_belief_manager.get_expected_belief_distribution(target_agent_id, task_idx_for_bm_dyn)
        
        if belief_dist_for_dyn_plot is None and self.num_task_types > 0:
            belief_dist_for_dyn_plot = np.full(self.num_task_types, 1.0/self.num_task_types if self.num_task_types > 0 else 0)
        self.history["belief_for_dynamic_plot"].append(belief_dist_for_dyn_plot)


    def _update_frame(self, frame_idx: int):
        # (Code from previous version, with check for self.agent_colors emptiness)
        current_sim_time = self.history['time'][frame_idx]
        agent_states_at_frame = {ag_id: self.history["agent_states"][ag_id][frame_idx] for ag_id in self.env.agents if frame_idx < len(self.history["agent_states"].get(ag_id,[]))}
        assignments_at_frame = {ag_id: self.history["agent_assignments"][ag_id][frame_idx] for ag_id in self.env.agents if frame_idx < len(self.history["agent_assignments"].get(ag_id,[]))}

        all_x_coords_frame, all_y_coords_frame = [], [] # These are for potential auto-scaling, less critical if create_animation sets global bounds
        agent_keys = list(self.env.agents.keys())
        for i, agent_id in enumerate(agent_keys):
            if agent_id not in agent_states_at_frame: continue
            agent_full_state = agent_states_at_frame[agent_id]
            current_pos_xy = agent_full_state[:2]

            if frame_idx < len(self.history["agent_states"][agent_id]):
                trail_data_x = [s[0] for s in self.history["agent_states"][agent_id][:frame_idx + 1]]
                trail_data_y = [s[1] for s in self.history["agent_states"][agent_id][:frame_idx + 1]]
                self.agent_trails[agent_id].set_data(trail_data_x, trail_data_y)
            
            self.agent_markers[agent_id].set_data([current_pos_xy[0]], [current_pos_xy[1]])
            
            self.agent_id_texts[agent_id].set_position((current_pos_xy[0], current_pos_xy[1] + 0.8)) # Adjusted offset
            self.agent_id_texts[agent_id].set_text(f"S{agent_id}")
            # all_x_coords_frame.append(current_pos_xy[0]); all_y_coords_frame.append(current_pos_xy[1]) # Less critical now

            assigned_task_id = assignments_at_frame.get(agent_id, 0)
            if assigned_task_id != 0 and assigned_task_id in self.env.tasks:
                task_pos = self.env.tasks[assigned_task_id].position
                self.coalition_lines[agent_id].set_data([current_pos_xy[0], task_pos[0]], [current_pos_xy[1], task_pos[1]])
            else:
                self.coalition_lines[agent_id].set_data([], [])

        for task_id, task_obj_env in self.env.tasks.items():
            task_marker = self.task_markers.get(task_id)
            if task_marker:
                if task_obj_env.is_completed: # Check current status from env object
                    task_marker.set_markerfacecolor('lightcoral'); task_marker.set_markeredgecolor('darkred') # More distinct completed color
                    task_marker.set_alpha(0.7)
                else:
                    task_color_update = 'grey'
                    props_for_type = self.task_properties_by_type.get(task_obj_env.true_type,{})
                    if isinstance(props_for_type, dict):
                        revenue = props_for_type.get("revenue")
                        if revenue is not None:
                            task_color_update = self.task_type_colormap(self.norm_for_task_color(revenue))
                    task_marker.set_markerfacecolor(task_color_update); task_marker.set_markeredgecolor('black'); task_marker.set_alpha(1.0)


        if self.belief_bars and frame_idx < len(self.history["belief_for_dynamic_plot"]):
            belief_dist_frame = self.history["belief_for_dynamic_plot"][frame_idx]
            if belief_dist_frame is not None and len(belief_dist_frame) == len(self.belief_bars):
                for bar, h_val in zip(self.belief_bars, belief_dist_frame): bar.set_height(h_val)

        self.ax_main.set_title(f"{self.main_title_prefix}仿真 (时间: {current_sim_time:.1f}s)", fontsize=16, weight='bold')
        
        artists = list(self.agent_trails.values()) + list(self.agent_markers.values()) + \
                    list(self.agent_id_texts.values()) + list(self.task_markers.values()) + \
                    list(self.task_id_texts.values()) + list(self.coalition_lines.values())
        if self.belief_bars: artists.extend(self.belief_bars)
        return artists


    def create_animation(self, interval=200, output_filename=None):
        # (Identical to previous create_animation method, with robust axis setting)
        if not self.history["time"]:
            print("可视化历史记录为空，无法创建动画。")
            return None
        num_frames = len(self.history["time"])
        print(f"为 {num_frames} 帧创建动画...")

        if (not self.fixed_xlim or not self.fixed_ylim) and num_frames > 0 : # Set bounds if not fixed and history exists
            overall_min_x, overall_max_x, overall_min_y, overall_max_y = np.inf, -np.inf, np.inf, -np.inf
            has_pos_data = False
            for agent_id_hist_key in self.history["agent_states"]: 
                agent_hist_data = self.history["agent_states"][agent_id_hist_key]
                if agent_hist_data: 
                    agent_trail_np = np.array(agent_hist_data)
                    if agent_trail_np.size > 0:
                        has_pos_data = True
                        overall_min_x = min(overall_min_x, np.min(agent_trail_np[:,0])); overall_max_x = max(overall_max_x, np.max(agent_trail_np[:,0]))
                        overall_min_y = min(overall_min_y, np.min(agent_trail_np[:,1])); overall_max_y = max(overall_max_y, np.max(agent_trail_np[:,1]))
            for task_obj in self.env.tasks.values():
                has_pos_data = True 
                overall_min_x = min(overall_min_x, task_obj.position[0]); overall_max_x = max(overall_max_x, task_obj.position[0])
                overall_min_y = min(overall_min_y, task_obj.position[1]); overall_max_y = max(overall_max_y, task_obj.position[1])
            
            if has_pos_data and np.isfinite(overall_min_x): 
                padding_factor = 0.15
                range_x = overall_max_x - overall_min_x
                range_y = overall_max_y - overall_min_y
                padding_x = max(5, range_x * padding_factor if range_x > 1e-6 else 5)
                padding_y = max(5, range_y * padding_factor if range_y > 1e-6 else 5)

                if not self.fixed_xlim: self.ax_main.set_xlim(overall_min_x - padding_x, overall_max_x + padding_x)
                if not self.fixed_ylim: self.ax_main.set_ylim(overall_min_y - padding_y, overall_max_y + padding_y)
            else: 
                if not self.fixed_xlim : self.ax_main.set_xlim(-15,15) # Default if no data for auto-scale
                if not self.fixed_ylim : self.ax_main.set_ylim(-15,15)


        ani = FuncAnimation(self.fig_main, self._update_frame, frames=num_frames,
                            interval=interval, blit=True, repeat=False)
        if output_filename:
            try:
                output_dir = os.path.dirname(output_filename)
                if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
                final_filename = f"{output_filename}.mp4"
                writer_kwargs = {'fps': max(5, int(1000 / interval)), 'codec': 'libx264', 'bitrate': 3000, 'extra_args': ['-vcodec', 'libx264', '-pix_fmt', 'yuv420p']}
                ani.save(final_filename, writer='ffmpeg', dpi=250, **writer_kwargs) # Increased DPI
                print(f"动画已保存至: {final_filename}")
            except Exception as e:
                print(f"保存动画时出错: {e}。请确保 FFmpeg 已安装并配置在系统路径中。")
        return ani

    def plot_final_static_charts(self, results_dir="results", scenario_name_prefix="sim"):
        if not os.path.exists(results_dir): os.makedirs(results_dir, exist_ok=True)
        time_array = np.array(self.history["time"])
        if len(time_array) == 0:
            print("无历史数据可用于绘制静态图表。")
            return

        # --- 1. Belief Evolution Plot for all agents and tasks ---
        plot_belief_flag = False
        if self.env.agents and self.env.tasks and self.history["agent_task_beliefs_strong"]:
            # Check if there's any actual belief data recorded
            for ag_id_chk in self.env.agents.keys():
                if self.history["agent_task_beliefs_strong"].get(ag_id_chk):
                    for task_id_chk in self.env.tasks.keys():
                        if self.history["agent_task_beliefs_strong"][ag_id_chk].get(task_id_chk):
                            plot_belief_flag = True; break
                    if plot_belief_flag: break
        
        if plot_belief_flag:
            num_agents_total = len(self.env.agents)
            num_tasks_total = len(self.env.tasks)
            if num_agents_total > 0 and num_tasks_total > 0:
                ncols_belief = min(num_tasks_total, 3) 
                nrows_belief = num_agents_total * ((num_tasks_total + ncols_belief - 1) // ncols_belief)
                
                fig_belief_width = max(12, 6 * ncols_belief) # Increased width per subplot
                fig_belief_height = max(8, 4.5 * nrows_belief) # Increased height per subplot
                
                fig_belief, axes_belief_flat = plt.subplots(nrows_belief, ncols_belief,
                                                            figsize=(fig_belief_width, fig_belief_height), squeeze=False)
                axes_belief_flat = axes_belief_flat.flatten()
                fig_belief.suptitle(f"{self.main_title_prefix}各航天器对各任务的信念演化", fontsize=18, weight='bold')
                plot_idx = 0

                for agent_id in self.env.agents.keys():
                    for task_obj in self.env.tasks.values():
                        if plot_idx >= len(axes_belief_flat): break
                        ax = axes_belief_flat[plot_idx]
                        belief_history_for_task = np.array(self.history["agent_task_beliefs_strong"][agent_id].get(task_obj.id, []))
                        
                        true_type_name = self.task_properties_by_type.get(task_obj.true_type, {}).get("name", f"类型 {task_obj.true_type}")
                        plot_title = f"S{agent_id} 对 T{task_obj.id} (真: {true_type_name})"

                        if belief_history_for_task.ndim == 2 and \
                           belief_history_for_task.shape[0] > 0 and \
                           belief_history_for_task.shape[0] <= len(time_array) and \
                           belief_history_for_task.shape[1] == self.num_task_types:
                            
                            current_time_array_belief = time_array[:belief_history_for_task.shape[0]]
                            for k_type in range(self.num_task_types):
                                type_props = self.task_properties_by_type.get(k_type, {})
                                type_label = type_props.get("name", f"类型 {k_type}")
                                line, = ax.plot(current_time_array_belief, belief_history_for_task[:, k_type], 
                                                label=type_label, linewidth=2.0, linestyle=self.agent_linestyles[k_type % len(self.agent_linestyles)])
                                if k_type == task_obj.true_type: # Highlight true type
                                    line.set_linewidth(3.5)
                                    line.set_alpha(0.9)
                        else:
                            ax.text(0.5, 0.5, "无有效信念数据", ha='center', va='center', transform=ax.transAxes)
                        
                        ax.set_title(plot_title, fontsize=11)
                        if plot_idx // ncols_belief == (nrows_belief // num_agents_total) * num_agents_total -1 : # Simplified bottom row check per agent block
                             ax.set_xlabel("时间 (s)", fontsize=10)
                        if plot_idx % ncols_belief == 0:
                             ax.set_ylabel("信念概率", fontsize=10)
                        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize='small', loc='best'); ax.grid(True, linestyle=':', alpha=0.6)
                        ax.tick_params(axis='both', which='major', labelsize=9)
                        plot_idx += 1
                
                for k_ax_extra in range(plot_idx, len(axes_belief_flat)): axes_belief_flat[k_ax_extra].set_visible(False)
                plt.tight_layout(rect=[0, 0.03, 1, 0.96])
                fig_belief_path = os.path.join(results_dir, f"{scenario_name_prefix}_all_beliefs_evolution.png")
                fig_belief.savefig(fig_belief_path, dpi=300); plt.close(fig_belief)
                print(f"所有信念演化图已保存至: {fig_belief_path}")
        else:
            print("没有足够的信念数据来绘制所有航天器的信念演化图。")

        # --- 2. Velocity Plot (X and Y components) ---
        fig_vel_comp, axes_vel_comp = plt.subplots(2, 1, figsize=(12, 8), sharex=True) # Wider
        fig_vel_comp.suptitle(f"{self.main_title_prefix}航天器速度分量", fontsize=18, weight='bold')
        has_vel_data = False; agent_keys_vel = list(self.env.agents.keys())
        for i, agent_id in enumerate(agent_keys_vel):
            agent_states_hist = np.array(self.history["agent_states"].get(agent_id, []))
            if agent_states_hist.size > 0 and agent_states_hist.shape[0] == len(time_array) and agent_states_hist.shape[1] >= 4:
                color = self.agent_colors[i % len(self.agent_colors)]
                linestyle = self.agent_linestyles[i % len(self.agent_linestyles)]
                axes_vel_comp[0].plot(time_array, agent_states_hist[:, 2], label=f"S{agent_id} $v_x$", color=color, linestyle=linestyle, linewidth=2.0)
                axes_vel_comp[1].plot(time_array, agent_states_hist[:, 3], label=f"S{agent_id} $v_y$", color=color, linestyle=linestyle, linewidth=2.0)
                has_vel_data = True
        axes_vel_comp[0].set_ylabel("$v_x$ (m/s)", fontsize=12); axes_vel_comp[0].grid(True, linestyle=':', alpha=0.6); axes_vel_comp[0].tick_params(labelsize=10)
        if has_vel_data: axes_vel_comp[0].legend(fontsize='medium', loc='best')
        axes_vel_comp[1].set_xlabel("时间 (s)", fontsize=14, weight='bold'); axes_vel_comp[1].set_ylabel("$v_y$ (m/s)", fontsize=12)
        axes_vel_comp[1].grid(True, linestyle=':', alpha=0.6); axes_vel_comp[1].tick_params(labelsize=10)
        if has_vel_data: axes_vel_comp[1].legend(fontsize='medium', loc='best')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig_vel_path = os.path.join(results_dir, f"{scenario_name_prefix}_velocity_components.png")
        fig_vel_comp.savefig(fig_vel_path, dpi=300); plt.close(fig_vel_comp)
        print(f"速度分量图已保存至: {fig_vel_path}")

        # --- 3. Acceleration Plot (X and Y components) ---
        fig_acc_comp, axes_acc_comp = plt.subplots(2, 1, figsize=(12, 8), sharex=True) # Wider
        fig_acc_comp.suptitle(f"{self.main_title_prefix}航天器加速度分量", fontsize=18, weight='bold')
        has_acc_data = False; agent_keys_acc = list(self.env.agents.keys())
        for i, agent_id in enumerate(agent_keys_acc):
            agent_controls_hist = np.array(self.history["agent_controls"].get(agent_id, []))
            if agent_controls_hist.size > 0 and agent_controls_hist.shape[0] == len(time_array) and agent_controls_hist.shape[1] >= 2:
                color = self.agent_colors[i % len(self.agent_colors)]
                linestyle = self.agent_linestyles[i % len(self.agent_linestyles)]
                axes_acc_comp[0].plot(time_array, agent_controls_hist[:, 0], label=f"S{agent_id} $a_x$", color=color, linestyle=linestyle, linewidth=2.0)
                axes_acc_comp[1].plot(time_array, agent_controls_hist[:, 1], label=f"S{agent_id} $a_y$", color=color, linestyle=linestyle, linewidth=2.0)
                has_acc_data = True
        axes_acc_comp[0].set_ylabel("$a_x$ (m/s²)", fontsize=12); axes_acc_comp[0].grid(True, linestyle=':', alpha=0.6); axes_acc_comp[0].tick_params(labelsize=10)
        if has_acc_data: axes_acc_comp[0].legend(fontsize='medium', loc='best')
        axes_acc_comp[1].set_xlabel("时间 (s)", fontsize=14, weight='bold'); axes_acc_comp[1].set_ylabel("$a_y$ (m/s²)", fontsize=12)
        axes_acc_comp[1].grid(True, linestyle=':', alpha=0.6); axes_acc_comp[1].tick_params(labelsize=10)
        if has_acc_data: axes_acc_comp[1].legend(fontsize='medium', loc='best')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig_acc_path = os.path.join(results_dir, f"{scenario_name_prefix}_acceleration_components.png")
        fig_acc_comp.savefig(fig_acc_path, dpi=300); plt.close(fig_acc_comp)
        print(f"加速度分量图已保存至: {fig_acc_path}")

        if self.fig_main:
            if len(time_array) > 0 :
                 self._update_frame(len(time_array) -1) # Update main animation plot to last frame
                 final_traj_path = os.path.join(results_dir, f"{scenario_name_prefix}_final_trajectory.png")
                 self.fig_main.savefig(final_traj_path, dpi=300)
                 print(f"最终轨迹图已保存至: {final_traj_path}")
            plt.close(self.fig_main) # Close main animation figure

if __name__ == '__main__':
    print("运行可视化器独立示例...")
    # (The __main__ block from the previous version of visualizer.py can be used here for testing)
    pass