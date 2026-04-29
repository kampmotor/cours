"""
多智能体系统避障演示（核心版）

此脚本演示了多智能体系统中的避障功能，使用最简化的实现
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

# 添加源代码路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python_mult_agent_sim', 'src'))

from utils import calculate_acceleration, calculate_polarization, clamp_vector


class SimpleObstacle:
    """
    简单障碍物类
    """
    def __init__(self, center, radius, repulsion_strength=5.0):
        self.center = np.array(center)
        self.radius = radius
        self.repulsion_strength = repulsion_strength


def calculate_obstacle_force(position, obstacles, repulsion_radius=3.0, repulsion_strength=2.0):
    """
    计算障碍物对智能体的排斥力
    """
    total_force = np.zeros(2)
    
    for obstacle in obstacles:
        # 计算智能体到障碍物中心的距离
        vec_to_center = obstacle.center - position
        distance = np.linalg.norm(vec_to_center)
        
        # 如果智能体在障碍物影响范围内
        if distance < (obstacle.radius + repulsion_radius):
            if distance > 0.1:  # 避免除零
                # 计算排斥力方向（从障碍物中心指向智能体）
                repulsion_direction = vec_to_center / distance
                
                # 计算排斥力大小（距离越近，力越大）
                if distance <= obstacle.radius:
                    # 如果已经在障碍物内部，使用最大排斥力
                    force_magnitude = obstacle.repulsion_strength * 10.0
                else:
                    # 使用递减函数
                    normalized_distance = (distance - obstacle.radius) / repulsion_radius
                    force_magnitude = obstacle.repulsion_strength * (1.0 - normalized_distance)
                
                # 累积排斥力
                total_force += repulsion_direction * force_magnitude
    
    return total_force


def get_knn_neighbors(robot_positions, current_robot_idx, k):
    """
    获取K近邻邻居
    """
    if robot_positions is None:
        return []
    
    # 计算与其他机器人的距离
    relative_positions = robot_positions - robot_positions[current_robot_idx]
    distances = np.linalg.norm(relative_positions, axis=1)
    
    # 排除自己
    distances[current_robot_idx] = float('inf')
    
    # 获取最近的k个邻居
    sorted_indices = np.argsort(distances)
    num_neighbors = min(k, len(robot_positions) - 1)
    selected_neighbors = sorted_indices[:num_neighbors].tolist()
    
    return selected_neighbors


def simple_avoidance_simulation(n_robots=8, field_size=20, umax=4, vmax_robot=1, dt=0.02, T=50,
                               lambda_param=1.6, r_alpha=2.0, beta_gain=0.004, k_neighbors=4,
                               enable_obstacle_avoidance=True, obstacles=None):
    """
    简化的避障仿真函数
    """
    if obstacles is None:
        obstacles = []
    
    # 初始化位置和速度
    robot_positions = np.random.rand(n_robots, 2) * 10 + (field_size/2 - 5)
    robot_velocities = (np.random.rand(n_robots, 2) * 2 - 1) * 0.1
    
    # 初始化结果数组
    polarization_history = np.zeros(T)
    robot_trajectory = np.zeros((n_robots, 2, T))
    robot_velocities_history = np.zeros((n_robots, 2, T))

    # 主模拟循环
    for t in range(T):
        # 内循环更新每个智能体
        for i in range(n_robots):
            # 获取邻居
            selected_neighbors_i = get_knn_neighbors(robot_positions, i, k_neighbors)
            
            # 标准模拟步骤
            if selected_neighbors_i:
                nearby_positions_globals = robot_positions[selected_neighbors_i]
                nearby_velocities = robot_velocities[selected_neighbors_i]

                relative_positions = nearby_positions_globals - robot_positions[i]

                robot_acceleration = calculate_acceleration(
                    relative_positions, 
                    robot_velocities[i], 
                    nearby_velocities, 
                    lambda_param, 
                    r_alpha, 
                    beta_gain, 
                    umax
                )
                
                # 添加避障力
                if enable_obstacle_avoidance and obstacles:
                    obstacle_force = calculate_obstacle_force(robot_positions[i], obstacles)
                    robot_acceleration = robot_acceleration + obstacle_force
                
                robot_velocities[i] = clamp_vector(
                    robot_velocities[i] + robot_acceleration * dt, 
                    vmax_robot
                )
        
        # 更新所有机器人的位置
        robot_positions += robot_velocities * dt
        
        # 记录此时间步的结果
        robot_trajectory[:, :, t] = robot_positions.copy()
        robot_velocities_history[:, :, t] = robot_velocities.copy()
        polarization_history[t] = calculate_polarization(robot_velocities)
    
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
    
    # 创建障碍物
    obstacles = [SimpleObstacle(center=[10, 10], radius=2.0, repulsion_strength=3.0)]
    
    # 运行带避障的仿真
    print("运行带避障的仿真...")
    results_with = simple_avoidance_simulation(
        n_robots=8,
        field_size=20,
        umax=4,
        vmax_robot=1,
        T=50,
        dt=0.02,
        lambda_param=1.6,
        r_alpha=2.0,
        beta_gain=0.004,
        k_neighbors=4,
        enable_obstacle_avoidance=True,
        obstacles=obstacles
    )
    pol_with, traj_with, vel_with = results_with

    # 运行无避障的仿真
    print("运行无避障的仿真...")
    results_without = simple_avoidance_simulation(
        n_robots=8,
        field_size=20,
        umax=4,
        vmax_robot=1,
        T=50,
        dt=0.02,
        lambda_param=1.6,
        r_alpha=2.0,
        beta_gain=0.004,
        k_neighbors=4,
        enable_obstacle_avoidance=False,
        obstacles=obstacles
    )
    pol_without, traj_without, vel_without = results_without
    
    # 可视化结果
    print("生成可视化结果...")
    visualize_simple_results(traj_with, traj_without, vel_with, vel_without, obstacles, field_size=20)
    
    print("简化版避障演示完成！")
    
    return {
        'with_obstacles': (pol_with, traj_with, vel_with),
        'without_obstacles': (pol_without, traj_without, vel_without),
        'obstacles': obstacles
    }


if __name__ == "__main__":
    results = run_simple_demo()
    print("演示已成功运行")