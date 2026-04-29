"""
三维多智能体系统避障仿真平台

此模块实现了三维空间中的多智能体避障仿真，扩展了二维版本的功能
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import os


class Obstacle3D:
    """
    三维障碍物类
    """
    def __init__(self, center, radius, repulsion_strength=5.0):
        """
        初始化球形障碍物
        
        参数:
        - center: 障碍物中心坐标 [x, y, z]
        - radius: 障碍物半径
        - repulsion_strength: 排斥力强度
        """
        self.center = np.array(center)
        self.radius = radius
        self.repulsion_strength = repulsion_strength


class MultiAgentSimulator3D:
    """
    三维多智能体仿真器
    """
    def __init__(self, n_robots=20, field_size=30, umax=4, vmax_robot=1, dt=0.02, T=100,
                 lambda_param=1.6, r_alpha=2.0, beta_gain=0.004, cos_alpha=np.cos(np.pi/6), k_neighbors=6):
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
        添加三维障碍物
        """
        if repulsion_strength is None:
            repulsion_strength = self.obstacle_repulsion_strength
        obstacle = Obstacle3D(center, radius, repulsion_strength)
        self.obstacles.append(obstacle)
    
    def initialize_agents(self, scenario_type='uniform'):
        """
        初始化三维智能体状态
        """
        if scenario_type == 'two_clusters':
            print('Initializing agents in Two-Cluster scenario (3D)...')
            n_half = self.n_robots // 2
            n_A = n_half
            n_B = self.n_robots - n_half
            
            # 集群参数
            cluster_center_A = np.array([self.field_size/2 - 10, self.field_size/2, self.field_size/2])  # 集群A中心
            cluster_center_B = np.array([self.field_size/2 + 10, self.field_size/2, self.field_size/2])  # 集群B中心
            cluster_size = 8  # 立方体边长
            
            # 集群A的位置
            pos_A = np.random.rand(n_A, 3) * cluster_size + (cluster_center_A - cluster_size/2)
            
            # 集群B的位置
            pos_B = np.random.rand(n_B, 3) * cluster_size + (cluster_center_B - cluster_size/2)
            
            self.robot_positions = np.vstack([pos_A, pos_B])
        else:
            # 均匀随机场景（默认）
            print('Initializing agents in Uniform Random scenario (3D)...')
            self.robot_positions = np.random.rand(self.n_robots, 3) * 10 + (self.field_size/2 - 5)
        
        # 初始速度（三维）
        self.robot_velocities = (np.random.rand(self.n_robots, 3) * 2 - 1) * 0.1
    
    def calculate_obstacle_force(self, position):
        """
        计算三维障碍物对智能体的排斥力
        """
        total_force = np.zeros(3)
        
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
    
    def get_knn_neighbors(self, current_robot_idx):
        """
        获取三维K近邻邻居
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
        num_neighbors = min(self.k_neighbors, self.n_robots - 1)
        selected_neighbors = sorted_indices[:num_neighbors].tolist()

        return selected_neighbors
    
    def get_spotlight_neighbors(self, i, cos_alpha):
        """
        获取三维Spotlight邻居
        """
        if self.robot_positions is None or self.robot_velocities is None:
            return []
        
        # 获取智能体i的速度方向
        agent_direction = self.robot_velocities[i]
        agent_direction_norm = np.linalg.norm(agent_direction)
        if agent_direction_norm > 1e-6:
            agent_direction = agent_direction / agent_direction_norm
        else:
            # 如果速度为零，使用随机方向
            agent_direction = np.random.rand(3) - 0.5
            agent_direction = agent_direction / np.linalg.norm(agent_direction)
        
        # 计算与其他智能体的相对位置
        relative_positions = self.robot_positions - self.robot_positions[i]
        distances = np.linalg.norm(relative_positions, axis=1)
        
        # 排除自己
        distances[i] = float('inf')
        
        # 计算方向向量
        direction_vectors = relative_positions / distances[:, np.newaxis]
        
        # 计算夹角余弦值
        cos_angles = np.dot(direction_vectors, agent_direction)
        
        # 选择在视场锥内的邻居
        in_cone = cos_angles >= cos_alpha
        in_range = distances <= self.field_size / 3  # 设置感知范围
        
        # 同时满足角度和距离条件的邻居
        valid_neighbors = in_cone & in_range
        
        # 获取邻居索引
        neighbor_indices = np.where(valid_neighbors)[0].tolist()
        
        return neighbor_indices
    
    def calculate_acceleration(self, relative_positions, self_velocity, nearby_velocities, 
                              lambda_param, r_alpha, beta_gain, umax):
        """
        计算三维加速度
        """
        n_neighbors = len(relative_positions)
        
        if n_neighbors == 0:
            return np.array([0.0, 0.0, 0.0])
        
        # 1. 速度一致性项 (Alignment)
        sum_velocity_diff = np.sum(nearby_velocities - self_velocity, axis=0)
        alignment_force = lambda_param * sum_velocity_diff
        
        # 2. 修正后的非线性队形控制项 (Corrected Non-linear Formation Control)
        k_attract = beta_gain
        k_repel = beta_gain * 2.0
        
        formation_force_total = np.zeros(3)
        for j in range(n_neighbors):
            pos_error_vec = -relative_positions[j]  # 注意符号变化
            dist = np.linalg.norm(pos_error_vec)
            
            if dist > 1e-6:
                # 添加最小距离保护，防止数值爆炸
                dist = max(dist, 0.1)
                if dist < r_alpha:  # 排斥区 (Repulsion Zone)
                    force_magnitude = k_repel * (1/dist - 1/r_alpha) * (1/dist**2)
                else:  # 吸引区 (Attraction Zone)
                    force_magnitude = k_attract * (dist - r_alpha)
                
                unit_vec = pos_error_vec / dist
                formation_force_total += force_magnitude * unit_vec
        
        # 3. 计算理论上的总加速度
        theoretical_acceleration = alignment_force + formation_force_total
        
        # 4. 加速度饱和限制 (Saturation)
        acceleration_norm = np.linalg.norm(theoretical_acceleration)
        if acceleration_norm > umax:
            acceleration = theoretical_acceleration / acceleration_norm * umax
        else:
            acceleration = theoretical_acceleration
        
        return acceleration
    
    def clamp_vector(self, v, max_norm):
        """
        限制三维向量大小
        """
        v_norm = np.linalg.norm(v)
        if v_norm > max_norm:
            v_clamped = v / v_norm * max_norm
        else:
            v_clamped = v
        return v_clamped
    
    def mainloop(self, neighbor_method='knn', enable_obstacle_avoidance=True):
        """
        三维主模拟循环
        """
        if neighbor_method not in ['knn', 'spotlight']:
            raise ValueError(f"neighbor_method must be 'knn' or 'spotlight', got '{neighbor_method}'")

        if self.robot_positions is None or self.robot_velocities is None:
            raise RuntimeError("Agents not initialized. Call initialize_agents() first.")

        # 初始化结果数组
        polarization_history = np.zeros(self.T)
        robot_trajectory = np.zeros((self.n_robots, 3, self.T))
        robot_velocities_history = np.zeros((self.n_robots, 3, self.T))
        neighbors_history = [[[] for _ in range(self.T)] for _ in range(self.n_robots)]
        algebraic_connectivity_history = np.zeros(self.T)

        # 主模拟循环
        for t in range(self.T):
            # 内循环更新每个智能体
            for i in range(self.n_robots):
                # 根据输入选择邻居查找算法
                if neighbor_method == 'knn':
                    selected_neighbors_i = self.get_knn_neighbors(i)
                elif neighbor_method == 'spotlight':
                    selected_neighbors_i = self.get_spotlight_neighbors(i, self.cos_alpha)
                
                # 存储此智能体和时间步的邻居历史
                neighbors_history[i][t] = selected_neighbors_i
                
                # 标准模拟步骤
                if selected_neighbors_i:
                    nearby_positions_globals = self.robot_positions[selected_neighbors_i]
                    nearby_velocities = self.robot_velocities[selected_neighbors_i]

                    relative_positions = nearby_positions_globals - self.robot_positions[i]

                    robot_acceleration = self.calculate_acceleration(
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
                    
                    self.robot_velocities[i] = self.clamp_vector(
                        self.robot_velocities[i] + robot_acceleration * self.dt, 
                        self.vmax_robot
                    )
            
            # 更新所有机器人的位置
            self.robot_positions += self.robot_velocities * self.dt
            
            # 边界处理
            self.robot_positions = np.clip(self.robot_positions, 0.5, self.field_size - 0.5)
            
            # --- 计算当前图的代数连通性 (lambda_2) ---
            # 性能优化：每10步计算一次代数连通性
            if t % 10 == 0 or t == 0:
                A = np.zeros((self.n_robots, self.n_robots))
                for j in range(self.n_robots):
                    if neighbors_history[j][t]:
                        A[j, neighbors_history[j][t]] = 1
                
                D_out = np.diag(np.sum(A, axis=1))
                L = D_out - A
                eigenvalues = np.linalg.eigvals(L)
                sorted_eigenvalues = np.sort(np.real(eigenvalues))
                if len(sorted_eigenvalues) > 1:
                    current_connectivity = sorted_eigenvalues[1]
                else:
                    current_connectivity = 0  # 处理单个机器人的情况
            algebraic_connectivity_history[t] = current_connectivity
            
            # 记录此时间步的结果
            robot_trajectory[:, :, t] = self.robot_positions.copy()
            robot_velocities_history[:, :, t] = self.robot_velocities.copy()
            polarization_history[t] = self.calculate_polarization(self.robot_velocities)
        
        return (
            polarization_history,
            robot_trajectory,
            robot_velocities_history,
            neighbors_history,
            algebraic_connectivity_history
        )
    
    def calculate_polarization(self, velocities):
        """
        计算三维群体极化度
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


def create_3d_avoidance_video(robot_trajectory, robot_velocities_history, neighbors_history, params, 
                            model_to_run='KNN', output_filename='3d_avoidance_simulation.gif'):
    """
    创建三维避障仿真视频
    """
    T = params['T']
    n_robots = params['n_robots']
    field_size = params['field_size']
    
    # 创建3D图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标轴范围
    ax.set_xlim(0, field_size)
    ax.set_ylim(0, field_size)
    ax.set_zlim(0, field_size)
    ax.set_title(f'3D Multi-Agent Obstacle Avoidance - {model_to_run.upper()} Model')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    
    # 绘制障碍物（如果有的话）
    if 'obstacles' in params:
        for obs in params['obstacles']:
            # 绘制球形障碍物
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = obs.radius * np.outer(np.cos(u), np.sin(v)) + obs.center[0]
            y = obs.radius * np.outer(np.sin(u), np.sin(v)) + obs.center[1]
            z = obs.radius * np.outer(np.ones(np.size(u)), np.cos(v)) + obs.center[2]
            ax.plot_surface(x, y, z, color='red', alpha=0.3)
    
    # 初始化绘图元素
    scat = ax.scatter([], [], [], s=50, c='blue', alpha=0.7)
    
    def animate(frame):
        # 清除之前的绘图
        ax.clear()
        ax.set_xlim(0, field_size)
        ax.set_ylim(0, field_size)
        ax.set_zlim(0, field_size)
        ax.set_title(f'3D Multi-Agent Obstacle Avoidance - {model_to_run.upper()} Model (Step: {frame})')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        
        # 获取当前位置
        current_positions = robot_trajectory[:, :, frame]
        
        # 绘制障碍物
        if 'obstacles' in params:
            for obs in params['obstacles']:
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x = obs.radius * np.outer(np.cos(u), np.sin(v)) + obs.center[0]
                y = obs.radius * np.outer(np.sin(u), np.sin(v)) + obs.center[1]
                z = obs.radius * np.outer(np.ones(np.size(u)), np.cos(v)) + obs.center[2]
                ax.plot_surface(x, y, z, color='red', alpha=0.3)
        
        # 绘制智能体位置
        ax.scatter(
            current_positions[:, 0], 
            current_positions[:, 1], 
            current_positions[:, 2], 
            s=50, c='blue', alpha=0.7
        )
        
        # 绘制速度向量（每5个智能体绘制一个，避免过于拥挤）
        for i in range(0, n_robots, max(1, n_robots//8)):
            start_pos = current_positions[i]
            velocity = robot_velocities_history[i, :, frame] * 2  # 放大速度向量
            ax.quiver(
                start_pos[0], start_pos[1], start_pos[2],
                velocity[0], velocity[1], velocity[2],
                color='green', arrow_length_ratio=0.1
            )
        
        return [scat]
    
    # 创建动画
    ani = FuncAnimation(fig, animate, frames=T, interval=100, blit=False, repeat=True)
    
    # 保存为GIF
    try:
        ani.save(output_filename, writer='pillow', fps=10)
        print(f'3D避障仿真GIF已保存至: {output_filename}')
    except Exception as e:
        print(f'保存3D GIF时出错: {e}')
        print('请确保已安装Pillow: pip install pillow')
    
    plt.close(fig)  # 关闭图形以节省内存
    return ani


def run_3d_comparison_experiment():
    """
    运行三维对比实验
    """
    print("开始三维多智能体避障对比实验...")
    
    # 参数设置
    n_robots = 15
    field_size = 30
    T = 80
    dt = 0.02
    
    # 创建仿真器
    simulator = MultiAgentSimulator3D(
        n_robots=n_robots,
        field_size=field_size,
        T=T,
        dt=dt
    )
    
    # 添加三维障碍物
    obstacles = [
        Obstacle3D(center=[15, 15, 15], radius=3.0, repulsion_strength=4.0),  # 中心球形障碍物
        Obstacle3D(center=[8, 8, 8], radius=2.0, repulsion_strength=3.0),    # 角落障碍物
        Obstacle3D(center=[22, 22, 22], radius=2.5, repulsion_strength=3.5)  # 另对角落障碍物
    ]
    
    # 为仿真器添加障碍物
    for obs in obstacles:
        simulator.add_obstacle(obs.center, obs.radius, obs.repulsion_strength)
    
    # 测试KNN算法
    print("运行KNN算法仿真...")
    simulator.initialize_agents(scenario_type='uniform')
    knn_results = simulator.mainloop(neighbor_method='knn', enable_obstacle_avoidance=True)
    knn_pol, knn_traj, knn_vel, knn_nbrs, knn_conn = knn_results
    
    # 测试Spotlight算法
    print("运行Spotlight算法仿真...")
    simulator.initialize_agents(scenario_type='uniform')
    spotlight_results = simulator.mainloop(neighbor_method='spotlight', enable_obstacle_avoidance=True)
    spot_pol, spot_traj, spot_vel, spot_nbrs, spot_conn = spotlight_results
    
    # 准备参数用于视频生成
    params = {
        'T': T,
        'dt': dt,
        'field_size': field_size,
        'n_robots': n_robots,
        'obstacles': obstacles
    }
    
    # 生成KNN算法视频
    print("生成KNN算法3D视频...")
    create_3d_avoidance_video(
        knn_traj, knn_vel, knn_nbrs, params, 
        model_to_run='KNN', 
        output_filename='3d_knn_avoidance_simulation.gif'
    )
    
    # 生成Spotlight算法视频
    print("生成Spotlight算法3D视频...")
    create_3d_avoidance_video(
        spot_traj, spot_vel, spot_nbrs, params, 
        model_to_run='Spotlight', 
        output_filename='3d_spotlight_avoidance_simulation.gif'
    )
    
    # 绘制极化度对比
    plot_3d_polarization_comparison(knn_pol, spot_pol)
    
    print("三维对比实验完成！")
    
    return {
        'knn': knn_results,
        'spotlight': spotlight_results,
        'obstacles': obstacles
    }


def plot_3d_polarization_comparison(knn_pol, spotlight_pol):
    """
    绘制三维仿真极化度对比
    """
    T = len(knn_pol)
    time_steps = np.arange(T)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, knn_pol, 'b-', label='KNN Algorithm', linewidth=2)
    plt.plot(time_steps, spotlight_pol, 'r-', label='Spotlight Algorithm', linewidth=2)
    plt.title('3D Multi-Agent Polarization Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Polarization')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.savefig('3d_polarization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs('3d_videos', exist_ok=True)
    os.chdir('3d_videos')
    
    # 运行三维对比实验
    results = run_3d_comparison_experiment()
    
    print("\n生成的文件:")
    print("- 3d_knn_avoidance_simulation.gif: 3D KNN算法仿真")
    print("- 3d_spotlight_avoidance_simulation.gif: 3D Spotlight算法仿真")
    print("- 3d_polarization_comparison.png: 3D极化度对比图")