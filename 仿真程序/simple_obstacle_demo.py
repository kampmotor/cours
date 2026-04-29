"""
多智能体系统避障实验演示（简化版）

此脚本演示了多智能体系统中的避障功能，使用较少的机器人和时间步以加快运行
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

# 添加源代码路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python_mult_agent_sim'))

from src.simulator import MultiAgentSimulator
from src.utils import calculate_acceleration, calculate_polarization, clamp_vector
from src.utils import get_neighbor_knn as original_get_neighbor_knn
from src.utils import get_neighbor_spotlights as original_get_neighbor_spotlights


class Obstacle:
    """
    障碍物类
    """
    def __init__(self, center, radius, repulsion_strength=5.0):
        """
        初始化圆形障碍物
        
        参数:
        - center: 障碍物中心坐标 [x, y]
        - radius: 障碍物半径
        - repulsion_strength: 排斥力强度
        """
        self.center = np.array(center)
        self.radius = radius
        self.repulsion_strength = repulsion_strength


class ObstacleAvoidanceSimulator(MultiAgentSimulator):
    """
    支持避障的多智能体仿真器
    """
    def __init__(self, n_robots=10, field_size=20, umax=4, vmax_robot=1, dt=0.02, T=50,
                 lambda_param=1.6, r_alpha=2.0, beta_gain=0.004, cos_alpha=0.866, k_neighbors=4):
        # 直接调用父类构造函数的参数
        super().__init__(
            n_robots=n_robots,
            field_size=field_size,
            umax=umax,
            vmax_robot=vmax_robot,
            dt=dt,
            T=T,
            lambda_param=lambda_param,
            r_alpha=r_alpha,
            beta_gain=beta_gain,
            cos_alpha=cos_alpha,
            k_neighbors=k_neighbors
        )
        
        # 障碍物列表
        self.obstacles = []
        
        # 避障参数
        self.obstacle_repulsion_radius = 3.0  # 障碍物影响半径
        self.obstacle_repulsion_strength = 2.0  # 障碍物排斥强度
    
    def get_neighbor_knn(self, current_robot_idx, robot_positions, k):
        """
        K近邻邻居选择
        """
        return original_get_neighbor_knn(current_robot_idx, robot_positions, k)
    
    def get_neighbor_spotlights(self, i, robot_positions, cos_alpha):
        """
        Spotlight邻居选择
        """
        return original_get_neighbor_spotlights(i, robot_positions, cos_alpha)
    
    def add_obstacle(self, center, radius, repulsion_strength=None):
        """
        添加障碍物
        
        参数:
        - center: 障碍物中心坐标 [x, y]
        - radius: 障碍物半径
        - repulsion_strength: 排斥力强度（可选）
        """
        if repulsion_strength is None:
            repulsion_strength = self.obstacle_repulsion_strength
        obstacle = Obstacle(center, radius, repulsion_strength)
        self.obstacles.append(obstacle)
    
    def calculate_obstacle_force(self, position):
        """
        计算障碍物对智能体的排斥力
        
        参数:
        - position: 智能体位置 [x, y]
        
        返回:
        - obstacle_force: 障碍物排斥力 [fx, fy]
        """
        total_force = np.zeros(2)
        
        for obstacle in self.obstacles:
            # 计算智能体到障碍物中心的距离
            vec_to_center = obstacle.center - position
            distance = np.linalg.norm(vec_to_center)
            
            # 如果智能体在障碍物影响范围内
            if distance < (obstacle.radius + self.obstacle_repulsion_radius):
                if distance > 0.1:  # 避免除零
                    # 计算排斥力方向（从障碍物中心指向智能体）
                    repulsion_direction = vec_to_center / distance
                    
                    # 计算排斥力大小（距离越近，力越大）
                    if distance <= obstacle.radius:
                        # 如果已经在障碍物内部，使用最大排斥力
                        force_magnitude = obstacle.repulsion_strength * 10.0
                    else:
                        # 使用递减函数
                        normalized_distance = (distance - obstacle.radius) / self.obstacle_repulsion_radius
                        force_magnitude = obstacle.repulsion_strength * (1.0 - normalized_distance)
                    
                    # 累积排斥力
                    total_force += repulsion_direction * force_magnitude
        
        return total_force
    
    def mainloop(self, neighbor_method='knn', enable_obstacle_avoidance=True):
        """
        主模拟循环（重写以包含避障功能）
        
        参数:
            neighbor_method: 'knn' 或 'spotlight'
            enable_obstacle_avoidance: 是否启用避障功能
        
        返回:
            polarization_history: 极化度历史
            robot_trajectory: 智能体轨迹
            robot_velocities_history: 速度历史
            neighbors_history: 邻居历史
            algebraic_connectivity_history: 代数连通性历史
        """
        if neighbor_method not in ['knn', 'spotlight']:
            raise ValueError(f"neighbor_method must be 'knn' or 'spotlight', got '{neighbor_method}'")

        if self.robot_positions is None or self.robot_velocities is None:
            raise RuntimeError("Agents not initialized. Call initialize_agents() first.")

        # 初始化结果数组
        self.polarization_history = np.zeros(self.T)
        self.robot_trajectory = np.zeros((self.n_robots, 2, self.T))
        self.robot_velocities_history = np.zeros((self.n_robots, 2, self.T))
        self.neighbors_history = [[[] for _ in range(self.T)] for _ in range(self.n_robots)]
        self.algebraic_connectivity_history = np.zeros(self.T)

        # 主模拟循环
        for t in range(self.T):
            # 内循环更新每个智能体
            for i in range(self.n_robots):
                # 根据输入选择邻居查找算法
                if neighbor_method == 'knn':
                    selected_neighbors_i = original_get_neighbor_knn(i, self.robot_positions, self.k_neighbors)
                elif neighbor_method == 'spotlight':
                    selected_neighbors_i = original_get_neighbor_spotlights(i, self.robot_positions, self.cos_alpha)
                
                # 存储此智能体和时间步的邻居历史
                self.neighbors_history[i][t] = selected_neighbors_i
                
                # 标准模拟步骤
                if selected_neighbors_i:
                    nearby_positions_globals = self.robot_positions[selected_neighbors_i]
                    nearby_velocities = self.robot_velocities[selected_neighbors_i]

                    relative_positions = nearby_positions_globals - self.robot_positions[i]

                    robot_acceleration = calculate_acceleration(
                        relative_positions,
                        self.robot_velocities[i],
                        nearby_velocities,
                        self.lambda_param,
                        self.r_alpha,
                        self.beta_gain,
                        self.umax
                    )

                    # 添加避障力
                    if enable_obstacle_avoidance and self.obstacles:
                        obstacle_force = self.calculate_obstacle_force(self.robot_positions[i])
                        robot_acceleration = robot_acceleration + obstacle_force

                    self.robot_velocities[i] = clamp_vector(
                        self.robot_velocities[i] + robot_acceleration * self.dt,
                        self.vmax_robot
                    )
            
            # 在计算完所有新速度后更新所有机器人的位置
            self.robot_positions += self.robot_velocities * self.dt
            
            # --- 计算当前图的代数连通性 (lambda_2) ---
            # 性能优化：每10步计算一次代数连通性
            if t % 10 == 0 or t == 0:
                A = np.zeros((self.n_robots, self.n_robots))
                for j in range(self.n_robots):
                    if self.neighbors_history[j][t]:
                        A[j, self.neighbors_history[j][t]] = 1
                
                D_out = np.diag(np.sum(A, axis=1))
                L = D_out - A
                eigenvalues = np.linalg.eigvals(L)
                sorted_eigenvalues = np.sort(np.real(eigenvalues))
                if len(sorted_eigenvalues) > 1:
                    current_connectivity = sorted_eigenvalues[1]
                else:
                    current_connectivity = 0  # 处理单个机器人的情况
            self.algebraic_connectivity_history[t] = current_connectivity
            
            # 记录此时间步的结果
            self.robot_trajectory[:, :, t] = self.robot_positions.copy()
            self.robot_velocities_history[:, :, t] = self.robot_velocities.copy()
            self.polarization_history[t] = calculate_polarization(self.robot_velocities)
        
        return (
            self.polarization_history,
            self.robot_trajectory,
            self.robot_velocities_history,
            self.neighbors_history,
            self.algebraic_connectivity_history
        )


def visualize_simple_obstacle_avoidance_results(results_with_obstacles, results_without_obstacles, obstacles, 
                                       field_size=30, output_dir='results'):
    """
    简化版可视化避障实验结果
    
    参数:
    - results_with_obstacles: 启用避障的仿真结果
    - results_without_obstacles: 未启用避障的仿真结果
    - obstacles: 障碍物列表
    - field_size: 场景大小
    - output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 解包结果
    pol_with, traj_with, vel_with, nbrs_with, conn_with = results_with_obstacles
    pol_without, traj_without, vel_without, nbrs_without, conn_without = results_without_obstacles

    # 获取时间步数和机器人数量
    T = len(pol_with)
    n_robots = traj_with.shape[0]

    # 创建时间向量
    time_steps = np.arange(T)

    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 最终时刻的轨迹对比（有避障）
    ax1 = axes[0, 0]
    ax1.set_xlim(0, field_size)
    ax1.set_ylim(0, field_size)
    ax1.set_title('With Obstacle Avoidance (Final State)')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.grid(True)
    ax1.axis('equal')

    # 绘制障碍物
    for obs in obstacles:
        circle = patches.Circle(obs.center, obs.radius, color='red', alpha=0.3, label='Obstacle')
        ax1.add_patch(circle)

    # 绘制最终位置
    final_positions_with = traj_with[:, :, -1]
    ax1.scatter(final_positions_with[:, 0], final_positions_with[:, 1], c='blue', s=50, alpha=0.7, label='Robots')

    # 绘制最终速度向量
    final_velocities_with = vel_with[:, :, -1]
    ax1.quiver(
        final_positions_with[:, 0],
        final_positions_with[:, 1],
        final_velocities_with[:, 0],
        final_velocities_with[:, 1],
        angles='xy', scale_units='xy', scale=1, color='green', width=0.003, alpha=0.7
    )

    # 2. 最终时刻的轨迹对比（无避障）
    ax2 = axes[0, 1]
    ax2.set_xlim(0, field_size)
    ax2.set_ylim(0, field_size)
    ax2.set_title('Without Obstacle Avoidance (Final State)')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.grid(True)
    ax2.axis('equal')

    # 绘制障碍物
    for obs in obstacles:
        circle = patches.Circle(obs.center, obs.radius, color='red', alpha=0.3, label='Obstacle')
        ax2.add_patch(circle)

    # 绘制最终位置
    final_positions_without = traj_without[:, :, -1]
    ax2.scatter(final_positions_without[:, 0], final_positions_without[:, 1], c='blue', s=50, alpha=0.7, label='Robots')

    # 绘制最终速度向量
    final_velocities_without = vel_without[:, :, -1]
    ax2.quiver(
        final_positions_without[:, 0],
        final_positions_without[:, 1],
        final_velocities_without[:, 0],
        final_velocities_without[:, 1],
        angles='xy', scale_units='xy', scale=1, color='green', width=0.003, alpha=0.7
    )

    # 3. 极化度对比
    ax3 = axes[1, 0]
    ax3.plot(time_steps, pol_with, 'b-', label='With Obstacle Avoidance', linewidth=2)
    ax3.plot(time_steps, pol_without, 'r--', label='Without Obstacle Avoidance', linewidth=2)
    ax3.set_title('Polarization Comparison')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Polarization')
    ax3.legend()
    ax3.grid(True)
    ax3.set_ylim(0, 1)

    # 4. 代数连通性对比
    ax4 = axes[1, 1]
    ax4.plot(time_steps, conn_with, 'b-', label='With Obstacle Avoidance', linewidth=2)
    ax4.plot(time_steps, conn_without, 'r--', label='Without Obstacle Avoidance', linewidth=2)
    ax4.set_title('Algebraic Connectivity (λ₂) Comparison')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('λ₂ (Connectivity)')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'simple_obstacle_avoidance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()


def run_simple_obstacle_avoidance_demo():
    """
    运行简化版避障演示实验
    """
    print("开始简化版避障演示实验...")
    
    # 创建仿真器（使用较少的机器人和时间步以加快运行）
    simulator = ObstacleAvoidanceSimulator(
        n_robots=8,
        field_size=20,
        T=50,
        dt=0.02
    )
    
    # 添加障碍物
    simulator.add_obstacle(center=[10, 10], radius=2.0, repulsion_strength=3.0)  # 中心障碍物
    
    # 初始化智能体（使用双集群场景以更好地展示避障效果）
    simulator.initialize_agents(scenario_type='two_clusters')
    
    # 运行仿真（带避障）
    print("运行带避障的仿真...")
    results_with_obstacles = simulator.mainloop(neighbor_method='knn', enable_obstacle_avoidance=True)
    
    # 重新初始化并运行仿真（无避障）
    print("运行无避障的仿真...")
    simulator.initialize_agents(scenario_type='two_clusters')
    results_without_obstacles = simulator.mainloop(neighbor_method='knn', enable_obstacle_avoidance=False)
    
    # 可视化结果
    print("生成可视化结果...")
    visualize_simple_obstacle_avoidance_results(
        results_with_obstacles,
        results_without_obstacles,
        simulator.obstacles,
        field_size=simulator.field_size,
        output_dir='../results'
    )
    
    print("简化版避障演示实验完成！")
    
    return {
        'with_obstacles': results_with_obstacles,
        'without_obstacles': results_without_obstacles,
        'obstacles': simulator.obstacles
    }


if __name__ == "__main__":
    # 运行演示
    demo_results = run_simple_obstacle_avoidance_demo()
    print("简化版避障演示已完成")