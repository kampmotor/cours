"""
多障碍物环境避障性能分析实验

此实验研究多智能体系统在多障碍物环境中的避障性能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import os


class MultiObstacleEnvironment:
    """
    多障碍物环境类
    """
    def __init__(self, field_size=30):
        self.field_size = field_size
        self.obstacles = []
        
    def add_obstacle(self, center, radius, strength=5.0):
        """
        添加障碍物
        """
        self.obstacles.append({
            'center': np.array(center),
            'radius': radius,
            'strength': strength
        })
        
    def calculate_total_obstacle_force(self, position, avoidance_radius=3.0):
        """
        计算所有障碍物对智能体的总排斥力
        """
        total_force = np.zeros(2)
        
        for obstacle in self.obstacles:
            vec_to_center = obstacle['center'] - position
            distance = np.linalg.norm(vec_to_center)
            
            if distance < (obstacle['radius'] + avoidance_radius):
                if distance > 0.1:
                    repulsion_direction = vec_to_center / distance
                    if distance <= obstacle['radius']:
                        force_magnitude = obstacle['strength'] * 10.0
                    else:
                        normalized_distance = (distance - obstacle['radius']) / avoidance_radius
                        force_magnitude = obstacle['strength'] * (1.0 - normalized_distance)
                    
                    total_force += repulsion_direction * force_magnitude
        
        return total_force


def calculate_polarization(velocities):
    """
    计算极化度
    """
    n_robots = velocities.shape[0]
    
    # 计算每个速度向量的范数（大小）
    norms = np.linalg.norm(velocities, axis=1)
    
    # 找到非零速度的索引，避免除零错误
    non_zero_indices = norms > 1e-9  # 使用小阈值处理浮点精度
    
    # 初始化归一化速度矩阵
    normalized_velocities = np.zeros_like(velocities)
    
    # 只归一化非零速度: v_i / ||v_i||
    if np.any(non_zero_indices):
        normalized_velocities[non_zero_indices] = velocities[non_zero_indices] / norms[non_zero_indices][:, np.newaxis]
    
    # 归一化速度向量之和: Σ(v_i / ||v_i||)
    sum_normalized_vectors = np.sum(normalized_velocities, axis=0)
    
    # 和的模长: || Σ(v_i / ||v_i||) ||
    magnitude_of_sum = np.linalg.norm(sum_normalized_vectors)
    
    # 最终极化度值: (1/N) * || Σ(v_i / ||v_i||) ||
    polarization = magnitude_of_sum / n_robots
    
    return polarization


def run_multi_obstacle_experiment():
    """
    运行多障碍物环境避障实验
    """
    print("开始多障碍物环境避障实验...")

    # 参数设置
    n_robots = 15
    field_size = 30
    T = 120
    dt = 0.02

    # 创建多障碍物环境
    env = MultiObstacleEnvironment(field_size=field_size)

    # 添加多个障碍物（形成复杂环境）
    env.add_obstacle([10, 10], 2.0, 4.0)  # 中心障碍物
    env.add_obstacle([20, 15], 1.5, 3.5)  # 右侧障碍物
    env.add_obstacle([15, 5], 1.8, 3.8)   # 下方障碍物
    env.add_obstacle([8, 22], 1.2, 3.0)   # 左上障碍物
    env.add_obstacle([25, 25], 2.2, 4.2)  # 右下障碍物

    # 初始化智能体位置（随机分布）
    robot_positions = np.random.rand(n_robots, 2) * field_size * 0.8 + field_size * 0.1
    robot_velocities = (np.random.rand(n_robots, 2) * 2 - 1) * 0.05  # 小的初始速度

    # 存储轨迹
    robot_trajectory = np.zeros((n_robots, 2, T))
    robot_velocities_history = np.zeros((n_robots, 2, T))
    polarization_history = np.zeros(T)

    # 仿真参数
    lambda_param = 1.6  # 对齐增益
    r_alpha = 2.0       # 期望距离
    beta_gain = 0.004   # 队形控制增益
    k_neighbors = 4     # 邻居数量
    umax = 4.0          # 最大加速度
    vmax_robot = 1.0    # 最大速度
    avoidance_radius = 4.0  # 避障影响半径

    # 仿真循环
    for t in range(T):
        new_positions = robot_positions.copy()
        new_velocities = robot_velocities.copy()
        
        for i in range(n_robots):
            # 获取邻居
            distances = np.linalg.norm(robot_positions - robot_positions[i], axis=1)
            distances[i] = float('inf')  # 排除自己
            sorted_indices = np.argsort(distances)
            num_neighbors = min(k_neighbors, n_robots - 1)
            selected_neighbors_i = sorted_indices[:num_neighbors]
            
            # 标准模拟步骤
            if len(selected_neighbors_i) > 0:
                nearby_positions_globals = robot_positions[selected_neighbors_i]
                nearby_velocities = robot_velocities[selected_neighbors_i]

                # 对齐力
                alignment_force = lambda_param * np.sum(nearby_velocities - robot_velocities[i], axis=0)
                
                # 队形控制力
                formation_force_total = np.zeros(2)
                for j in range(len(nearby_positions_globals)):
                    pos_error_vec = -(nearby_positions_globals[j] - robot_positions[i])
                    dist = np.linalg.norm(pos_error_vec)
                    
                    if dist > 1e-6:
                        dist = max(dist, 0.1)
                        if dist < r_alpha:  # 排斥区
                            force_magnitude = beta_gain * 2.0 * (1/dist - 1/r_alpha) * (1/dist**2)
                        else:  # 吸引区
                            force_magnitude = beta_gain * (dist - r_alpha)
                        
                        unit_vec = pos_error_vec / dist
                        formation_force_total += force_magnitude * unit_vec
                
                robot_acceleration = alignment_force + formation_force_total
                
                # 添加避障力
                obstacle_force = env.calculate_total_obstacle_force(robot_positions[i], avoidance_radius)
                robot_acceleration = robot_acceleration + obstacle_force
                
                # 限制加速度
                acc_norm = np.linalg.norm(robot_acceleration)
                if acc_norm > umax:
                    robot_acceleration = robot_acceleration / acc_norm * umax
                
                # 更新速度
                new_velocities[i] = robot_velocities[i] + robot_acceleration * dt
                
                # 限制速度
                vel_norm = np.linalg.norm(new_velocities[i])
                if vel_norm > vmax_robot:
                    new_velocities[i] = new_velocities[i] / vel_norm * vmax_robot
        
        # 更新位置
        new_positions = robot_positions + new_velocities * dt
        
        # 边界处理
        boundary_buffer = 1.0
        for i in range(n_robots):
            # X边界
            if new_positions[i, 0] < boundary_buffer:
                new_positions[i, 0] = boundary_buffer
                new_velocities[i, 0] = abs(new_velocities[i, 0]) * 0.5
            elif new_positions[i, 0] > field_size - boundary_buffer:
                new_positions[i, 0] = field_size - boundary_buffer
                new_velocities[i, 0] = -abs(new_velocities[i, 0]) * 0.5
            
            # Y边界
            if new_positions[i, 1] < boundary_buffer:
                new_positions[i, 1] = boundary_buffer
                new_velocities[i, 1] = abs(new_velocities[i, 1]) * 0.5
            elif new_positions[i, 1] > field_size - boundary_buffer:
                new_positions[i, 1] = field_size - boundary_buffer
                new_velocities[i, 1] = -abs(new_velocities[i, 1]) * 0.5
        
        # 存储轨迹
        robot_trajectory[:, :, t] = new_positions.copy()
        robot_velocities_history[:, :, t] = new_velocities.copy()
        
        # 计算极化度
        polarization_history[t] = calculate_polarization(new_velocities)
        
        # 更新位置和速度
        robot_positions = new_positions
        robot_velocities = new_velocities
    
    print("多障碍物实验完成，开始生成可视化...")
    
    # 创建动画
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, field_size)
    ax.set_ylim(0, field_size)
    ax.set_title('Multi-Agent System in Multi-Obstacle Environment')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True)
    ax.axis('equal')

    # 绘制障碍物
    for obs in env.obstacles:
        circle = patches.Circle(obs['center'], obs['radius'], color='red', alpha=0.3, label='Obstacle')
        ax.add_patch(circle)

    # 初始化绘图元素
    scat = ax.scatter([], [], s=80, c='blue', alpha=0.7, zorder=3)
    quiver = ax.quiver([], [], [], [], angles='xy', scale_units='xy', scale=1, color='green', width=0.003, zorder=2)

    def animate(frame):
        # 获取当前位置
        current_positions = robot_trajectory[:, :, frame]
        
        # 更新标题
        ax.set_title(f'Multi-Agent System in Multi-Obstacle Environment (Step: {frame})')
        
        # 更新智能体位置
        scat.set_offsets(current_positions)
        
        # 计算速度向量（基于轨迹变化）
        if frame > 0:
            prev_positions = robot_trajectory[:, :, max(0, frame-3)]
            velocities = (current_positions - prev_positions) / (dt * 3)
        else:
            velocities = np.zeros_like(current_positions)
        
        # 更新速度向量
        quiver_new = ax.quiver(
            current_positions[:, 0], 
            current_positions[:, 1],
            velocities[:, 0], 
            velocities[:, 1],
            angles='xy', scale_units='xy', scale=1, color='green', width=0.003, zorder=2
        )
        
        return [scat, quiver_new]

    # 创建动画
    ani = FuncAnimation(fig, animate, frames=T, interval=100, blit=False, repeat=True)

    # 保存为GIF
    try:
        os.makedirs('multi_obstacle_videos', exist_ok=True)
        gif_filename = 'multi_obstacle_avoidance.gif'
        ani.save(os.path.join('multi_obstacle_videos', gif_filename), writer='pillow', fps=10)
        print(f'多障碍物避障GIF已保存至: multi_obstacle_videos/{gif_filename}')
    except Exception as e:
        print(f'保存GIF时出错: {e}')
        print('请确保已安装Pillow: pip install pillow')

    plt.close(fig)  # 关闭图形以节省内存
    
    return robot_trajectory, robot_velocities_history, polarization_history, env.obstacles


def analyze_collision_free_paths(trajectory, obstacles):
    """
    分析无碰撞路径比例
    """
    n_robots, _, T = trajectory.shape
    collision_count = 0
    total_positions = n_robots * T
    
    for t in range(T):
        for i in range(n_robots):
            pos = trajectory[i, :, t]
            for obs in obstacles:
                vec_to_center = obs['center'] - pos
                distance = np.linalg.norm(vec_to_center)
                
                if distance <= obs['radius']:
                    collision_count += 1
    
    collision_free_ratio = 1.0 - (collision_count / total_positions)
    return collision_free_ratio


def run_performance_analysis():
    """
    运行性能分析实验
    """
    print("开始性能分析实验...")
    
    # 不同障碍物数量的实验
    obstacle_counts = [1, 3, 5, 7]
    n_trials = 3  # 每种配置运行3次取平均（减少以加快运行）
    
    results = {
        'obstacle_counts': obstacle_counts,
        'collision_free_ratios': [],
        'avg_polarization': [],
        'stability_indices': []
    }
    
    for n_obstacles in obstacle_counts:
        print(f"测试 {n_obstacles} 个障碍物...")
        
        trial_results = {
            'collision_free_ratios': [],
            'final_polarizations': [],
            'stability_indices': []
        }
        
        for trial in range(n_trials):
            print(f"  试验 {trial+1}/{n_trials}")
            
            # 参数设置
            n_robots = 12
            field_size = 30
            T = 80
            dt = 0.02

            # 创建多障碍物环境
            env = MultiObstacleEnvironment(field_size=field_size)
            
            # 随机生成障碍物
            for i in range(n_obstacles):
                center = [np.random.uniform(5, 25), np.random.uniform(5, 25)]
                radius = np.random.uniform(1.0, 2.5)
                strength = np.random.uniform(3.0, 5.0)
                env.add_obstacle(center, radius, strength)
            
            # 初始化智能体位置（随机分布）
            robot_positions = np.random.rand(n_robots, 2) * field_size * 0.8 + field_size * 0.1
            robot_velocities = (np.random.rand(n_robots, 2) * 2 - 1) * 0.05

            # 存储轨迹
            robot_trajectory = np.zeros((n_robots, 2, T))
            robot_velocities_history = np.zeros((n_robots, 2, T))
            polarization_history = np.zeros(T)

            # 仿真参数
            lambda_param = 1.6  # 对齐增益
            r_alpha = 2.0       # 期望距离
            beta_gain = 0.004   # 队形控制增益
            k_neighbors = 4     # 邻居数量
            umax = 4.0          # 最大加速度
            vmax_robot = 1.0    # 最大速度
            avoidance_radius = 4.0  # 避障影响半径

            # 仿真循环
            for t in range(T):
                new_positions = robot_positions.copy()
                new_velocities = robot_velocities.copy()
                
                for i in range(n_robots):
                    # 获取邻居
                    distances = np.linalg.norm(robot_positions - robot_positions[i], axis=1)
                    distances[i] = float('inf')  # 排除自己
                    sorted_indices = np.argsort(distances)
                    num_neighbors = min(k_neighbors, n_robots - 1)
                    selected_neighbors_i = sorted_indices[:num_neighbors]
                    
                    # 标准模拟步骤
                    if len(selected_neighbors_i) > 0:
                        nearby_positions_globals = robot_positions[selected_neighbors_i]
                        nearby_velocities = robot_velocities[selected_neighbors_i]

                        # 对齐力
                        alignment_force = lambda_param * np.sum(nearby_velocities - robot_velocities[i], axis=0)
                        
                        # 队形控制力
                        formation_force_total = np.zeros(2)
                        for j in range(len(nearby_positions_globals)):
                            pos_error_vec = -(nearby_positions_globals[j] - robot_positions[i])
                            dist = np.linalg.norm(pos_error_vec)
                            
                            if dist > 1e-6:
                                dist = max(dist, 0.1)
                                if dist < r_alpha:  # 排斥区
                                    force_magnitude = beta_gain * 2.0 * (1/dist - 1/r_alpha) * (1/dist**2)
                                else:  # 吸引区
                                    force_magnitude = beta_gain * (dist - r_alpha)
                                
                                unit_vec = pos_error_vec / dist
                                formation_force_total += force_magnitude * unit_vec
                        
                        robot_acceleration = alignment_force + formation_force_total
                        
                        # 添加避障力
                        obstacle_force = env.calculate_total_obstacle_force(robot_positions[i], avoidance_radius)
                        robot_acceleration = robot_acceleration + obstacle_force
                        
                        # 限制加速度
                        acc_norm = np.linalg.norm(robot_acceleration)
                        if acc_norm > umax:
                            robot_acceleration = robot_acceleration / acc_norm * umax
                        
                        # 更新速度
                        new_velocities[i] = robot_velocities[i] + robot_acceleration * dt
                        
                        # 限制速度
                        vel_norm = np.linalg.norm(new_velocities[i])
                        if vel_norm > vmax_robot:
                            new_velocities[i] = new_velocities[i] / vel_norm * vmax_robot
                
                # 更新位置
                new_positions = robot_positions + new_velocities * dt
                
                # 边界处理
                boundary_buffer = 1.0
                for i in range(n_robots):
                    # X边界
                    if new_positions[i, 0] < boundary_buffer:
                        new_positions[i, 0] = boundary_buffer
                        new_velocities[i, 0] = abs(new_velocities[i, 0]) * 0.5
                    elif new_positions[i, 0] > field_size - boundary_buffer:
                        new_positions[i, 0] = field_size - boundary_buffer
                        new_velocities[i, 0] = -abs(new_velocities[i, 0]) * 0.5
                    
                    # Y边界
                    if new_positions[i, 1] < boundary_buffer:
                        new_positions[i, 1] = boundary_buffer
                        new_velocities[i, 1] = abs(new_velocities[i, 1]) * 0.5
                    elif new_positions[i, 1] > field_size - boundary_buffer:
                        new_positions[i, 1] = field_size - boundary_buffer
                        new_velocities[i, 1] = -abs(new_velocities[i, 1]) * 0.5
                
                # 存储轨迹
                robot_trajectory[:, :, t] = new_positions.copy()
                robot_velocities_history[:, :, t] = new_velocities.copy()
                
                # 计算极化度
                polarization_history[t] = calculate_polarization(new_velocities)
                
                # 更新位置和速度
                robot_positions = new_positions
                robot_velocities = new_velocities
            
            # 分析结果
            collision_free_ratio = analyze_collision_free_paths(robot_trajectory, env.obstacles)
            final_polarization = polarization_history[-1]
            stability_index = 1.0 / (np.std(polarization_history) + 0.01)  # 稳定性指标
            
            trial_results['collision_free_ratios'].append(collision_free_ratio)
            trial_results['final_polarizations'].append(final_polarization)
            trial_results['stability_indices'].append(stability_index)
        
        # 计算平均值
        results['collision_free_ratios'].append(np.mean(trial_results['collision_free_ratios']))
        results['avg_polarization'].append(np.mean(trial_results['final_polarizations']))
        results['stability_indices'].append(np.mean(trial_results['stability_indices']))
    
    # 绘制性能分析结果
    plot_performance_analysis(results)
    
    return results


def plot_performance_analysis(results):
    """
    绘制性能分析结果
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    obstacle_counts = results['obstacle_counts']
    
    # 1. 无碰撞路径比例
    axes[0].plot(obstacle_counts, results['collision_free_ratios'], 'bo-', linewidth=2, markersize=8)
    axes[0].set_title('Collision-Free Path Ratio vs Obstacle Count')
    axes[0].set_xlabel('Number of Obstacles')
    axes[0].set_ylabel('Collision-Free Ratio')
    axes[0].grid(True)
    axes[0].set_ylim(0.8, 1.0)  # 假设避障成功率很高
    
    # 2. 平均极化度
    axes[1].plot(obstacle_counts, results['avg_polarization'], 'ro-', linewidth=2, markersize=8)
    axes[1].set_title('Average Polarization vs Obstacle Count')
    axes[1].set_xlabel('Number of Obstacles')
    axes[1].set_ylabel('Average Polarization')
    axes[1].grid(True)
    axes[1].set_ylim(0, 1)
    
    # 3. 稳定性指标
    axes[2].plot(obstacle_counts, results['stability_indices'], 'go-', linewidth=2, markersize=8)
    axes[2].set_title('Stability Index vs Obstacle Count')
    axes[2].set_xlabel('Number of Obstacles')
    axes[2].set_ylabel('Stability Index')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('multi_obstacle_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以节省内存
    print("性能分析图表已保存至: multi_obstacle_performance_analysis.png")


def main():
    """
    主函数 - 运行多障碍物实验
    """
    print("开始运行多障碍物环境避障实验...")
    
    # 运行多障碍物实验
    traj, vel, pol, obstacles = run_multi_obstacle_experiment()
    
    # 运行性能分析
    performance_results = run_performance_analysis()
    
    print("\n多障碍物环境避障实验完成！")
    print("生成的文件:")
    print("- multi_obstacle_videos/multi_obstacle_avoidance.gif: 多障碍物避障动画")
    print("- multi_obstacle_performance_analysis.png: 性能分析图表")
    
    print("\n实验结果摘要:")
    print(f"- 障碍物数量: {len(obstacles)}")
    print(f"- 机器人数量: 15")
    print(f"- 仿真时间步: 120")
    print(f"- 最终极化度: {pol[-1]:.3f}")


if __name__ == "__main__":
    main()