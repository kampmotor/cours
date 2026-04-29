"""
多智能体系统避障演示（极简版）

此脚本演示了多智能体系统中的避障功能，使用最少的代码实现核心功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

# 添加源代码路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python_mult_agent_sim'))

from src.utils import calculate_acceleration, calculate_polarization, clamp_vector


class SimpleObstacle:
    """
    简单障碍物类
    """
    def __init__(self, center, radius, repulsion_strength=5.0):
        self.center = np.array(center)
        self.radius = radius
        self.repulsion_strength = repulsion_strength


class SimpleObstacleAvoidanceSimulator:
    """
    简化的避障仿真器
    """
    def __init__(self, n_robots=8, field_size=20, umax=4, vmax_robot=1, dt=0.02, T=50,
                 lambda_param=1.6, r_alpha=2.0, beta_gain=0.004, cos_alpha=0.866, k_neighbors=4):
        self.n_robots = n_robots
        self.field_size = field_size
        self.umax = umax
        self.vmax_robot = vmax_robot
        self.dt = dt
        self.T = T
        self.lambda_param = lambda_param
        self.r_alpha = r_alpha
        self.beta_gain = beta_gain
        self.cos_alpha = cos_alpha
        self.k_neighbors = k_neighbors
        
        # 初始化位置和速度
        self.robot_positions = None
        self.robot_velocities = None
        
        # 障碍物列表
        self.obstacles = []
        
        # 避障参数
        self.obstacle_repulsion_radius = 3.0
        self.obstacle_repulsion_strength = 2.0
    
    def add_obstacle(self, center, radius, repulsion_strength=None):
        """
        添加障碍物
        """
        if repulsion_strength is None:
            repulsion_strength = self.obstacle_repulsion_strength
        obstacle = SimpleObstacle(center, radius, repulsion_strength)
        self.obstacles.append(obstacle)
    
    def initialize_agents(self, scenario_type='uniform'):
        """
        初始化智能体
        """
        if scenario_type == 'two_clusters':
            n_half = self.n_robots // 2
            n_A = n_half
            n_B = self.n_robots - n_half
            
            # 集群参数
            cluster_center_A = np.array([self.field_size/2 - 8, self.field_size/2])
            cluster_center_B = np.array([self.field_size/2 + 8, self.field_size/2])
            cluster_size = 5
            
            # 集群A的位置
            pos_A = np.random.rand(n_A, 2) * cluster_size + (cluster_center_A - cluster_size/2)
            
            # 集群B的位置
            pos_B = np.random.rand(n_B, 2) * cluster_size + (cluster_center_B - cluster_size/2)
            
            self.robot_positions = np.vstack([pos_A, pos_B])
        else:
            # 均匀随机场景
            self.robot_positions = np.random.rand(self.n_robots, 2) * 10 + (self.field_size/2 - 5)
        
        # 初始速度
        self.robot_velocities = (np.random.rand(self.n_robots, 2) * 2 - 1) * 0.1
    
    def calculate_obstacle_force(self, position):
        """
        计算障碍物对智能体的排斥力
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
    
    def get_knn_neighbors(self, current_robot_idx, k):
        """
        获取K近邻邻居
        """
        if self.robot_positions is None:
            return []
        
        # 计算与其他机器人的距离
        relative_positions = self.robot_positions - self.robot_positions[current_robot_idx]
        distances = np.linalg.norm(relative_positions, axis=1)
        
        # 排除自己
        distances[current_robot_idx] = float('inf')
        
        # 获取最近的k个邻居
        sorted_indices = np.argsort(distances)
        num_neighbors = min(k, self.n_robots - 1)
        selected_neighbors = sorted_indices[:num_neighbors].tolist()
        
        return selected_neighbors
    
    def mainloop(self, enable_obstacle_avoidance=True):
        """
        主模拟循环
        """
        if self.robot_positions is None or self.robot_velocities is None:
            raise RuntimeError("Agents not initialized. Call initialize_agents() first.")

        # 初始化结果数组
        polarization_history = np.zeros(self.T)
        robot_trajectory = np.zeros((self.n_robots, 2, self.T))
        robot_velocities_history = np.zeros((self.n_robots, 2, self.T))

        # 主模拟循环
        for t in range(self.T):
            # 内循环更新每个智能体
            for i in range(self.n_robots):
                # 获取邻居
                selected_neighbors_i = self.get_knn_neighbors(i, self.k_neighbors)
                
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
            
            # 更新所有机器人的位置
            self.robot_positions += self.robot_velocities * self.dt
            
            # 记录此时间步的结果
            robot_trajectory[:, :, t] = self.robot_positions.copy()
            robot_velocities_history[:, :, t] = self.robot_velocities.copy()
            polarization_history[t] = calculate_polarization(self.robot_velocities)
        
        return (
            polarization_history,
            robot_trajectory,
            robot_velocities_history
        )


def visualize_simple_results(traj_with, traj_without, vel_with, vel_without, obstacles, field_size=20):
    """
    简单可视化结果
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 有避障的结果
    ax1 = axes[0]
    ax1.set_xlim(0, field_size)
    ax1.set_ylim(0, field_size)
    ax1.set_title('With Obstacle Avoidance')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.grid(True)
    ax1.axis('equal')
    
    # 绘制障碍物
    for obs in obstacles:
        circle = patches.Circle(obs.center, obs.radius, color='red', alpha=0.3, label='Obstacle')
        ax1.add_patch(circle)
    
    # 绘制最终位置和速度向量
    final_positions_with = traj_with[:, :, -1]
    final_velocities_with = vel_with[:, :, -1]
    ax1.scatter(final_positions_with[:, 0], final_positions_with[:, 1], c='blue', s=50, alpha=0.7, label='Robots')
    ax1.quiver(
        final_positions_with[:, 0], 
        final_positions_with[:, 1],
        final_velocities_with[:, 0], 
        final_velocities_with[:, 1],
        angles='xy', scale_units='xy', scale=1, color='green', width=0.003, alpha=0.7
    )
    
    # 无避障的结果
    ax2 = axes[1]
    ax2.set_xlim(0, field_size)
    ax2.set_ylim(0, field_size)
    ax2.set_title('Without Obstacle Avoidance')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.grid(True)
    ax2.axis('equal')
    
    # 绘制障碍物
    for obs in obstacles:
        circle = patches.Circle(obs.center, obs.radius, color='red', alpha=0.3, label='Obstacle')
        ax2.add_patch(circle)
    
    # 绘制最终位置和速度向量
    final_positions_without = traj_without[:, :, -1]
    final_velocities_without = vel_without[:, :, -1]
    ax2.scatter(final_positions_without[:, 0], final_positions_without[:, 1], c='blue', s=50, alpha=0.7, label='Robots')
    ax2.quiver(
        final_positions_without[:, 0], 
        final_positions_without[:, 1],
        final_velocities_without[:, 0], 
        final_velocities_without[:, 1],
        angles='xy', scale_units='xy', scale=1, color='green', width=0.003, alpha=0.7
    )
    
    plt.tight_layout()
    plt.savefig('simple_obstacle_demo_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def run_simple_demo():
    """
    运行简化版演示
    """
    print("开始简化版避障演示...")
    
    # 创建仿真器
    simulator = SimpleObstacleAvoidanceSimulator(
        n_robots=8,
        field_size=20,
        T=50,
        dt=0.02
    )
    
    # 添加障碍物
    simulator.add_obstacle(center=[10, 10], radius=2.0, repulsion_strength=3.0)
    
    # 运行带避障的仿真
    print("运行带避障的仿真...")
    simulator.initialize_agents(scenario_type='two_clusters')
    results_with = simulator.mainloop(enable_obstacle_avoidance=True)
    pol_with, traj_with, vel_with = results_with
    
    # 运行无避障的仿真
    print("运行无避障的仿真...")
    simulator.initialize_agents(scenario_type='two_clusters')
    results_without = simulator.mainloop(enable_obstacle_avoidance=False)
    pol_without, traj_without, vel_without = results_without
    
    # 可视化结果
    print("生成可视化结果...")
    visualize_simple_results(traj_with, traj_without, vel_with, vel_without, simulator.obstacles, simulator.field_size)
    
    print("简化版避障演示完成！")
    
    return {
        'with_obstacles': (pol_with, traj_with, vel_with),
        'without_obstacles': (pol_without, traj_without, vel_without),
        'obstacles': simulator.obstacles
    }


if __name__ == "__main__":
    results = run_simple_demo()
    print("演示已成功运行")