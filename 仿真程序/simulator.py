import numpy as np
from .utils import (
    calculate_acceleration,
    calculate_polarization,
    clamp_vector,
    get_neighbor_knn,
    get_neighbor_spotlights,
    calculate_obstacle_force,
    create_obstacles,
    CircularObstacle
)
import config


class MultiAgentSimulator:
    def __init__(self, n_robots=None, field_size=None, umax=None, vmax_robot=None, dt=None, T=None,
                 lambda_param=None, r_alpha=None, beta_gain=None, cos_alpha=None, k_neighbors=None):
        """
        初始化多智能体模拟器

        参数:
            n_robots: 智能体数量 (默认: config.N_ROBOTS)
            field_size: 场景大小 (默认: config.FIELD_SIZE)
            umax: 最大加速度 (默认: config.UMAX)
            vmax_robot: 智能体最大速度 (默认: config.VMAX_ROBOT)
            dt: 时间步长 (默认: config.DT)
            T: 模拟时间步数 (默认: config.T)
            lambda_param: 速度一致性项的增益 (默认: config.LAMBDA_PARAM)
            r_alpha: 期望的智能体间距离 (默认: config.R_ALPHA)
            beta_gain: 队形控制项的基础增益 (默认: config.BETA_GAIN)
            cos_alpha: 角度余弦阈值 (默认: config.COS_ALPHA)
            k_neighbors: KNN算法的邻居数量 (默认: config.K_NEIGHBORS)

        参数取值范围说明请参考 config.py 中的文档
        """
        # 使用配置文件中的默认值
        self.n_robots = n_robots if n_robots is not None else config.N_ROBOTS
        self.field_size = field_size if field_size is not None else config.FIELD_SIZE
        self.umax = umax if umax is not None else config.UMAX
        self.vmax_robot = vmax_robot if vmax_robot is not None else config.VMAX_ROBOT
        self.dt = dt if dt is not None else config.DT
        self.T = T if T is not None else config.T
        self.lambda_param = lambda_param if lambda_param is not None else config.LAMBDA_PARAM
        self.r_alpha = r_alpha if r_alpha is not None else config.R_ALPHA
        self.beta_gain = beta_gain if beta_gain is not None else config.BETA_GAIN
        self.cos_alpha = cos_alpha if cos_alpha is not None else config.COS_ALPHA
        self.k_neighbors = k_neighbors if k_neighbors is not None else config.K_NEIGHBORS

        # 避障参数
        self.safety_dist = config.OBSTACLE_SAFETY_DIST
        self.detection_range = config.OBSTACLE_DETECTION_RANGE
        self.force_gain = config.OBSTACLE_FORCE_GAIN
        self.max_obstacle_force = config.OBSTACLE_MAX_FORCE
        self.obstacle_weight = config.OBSTACLE_PRIORITY_WEIGHT

        # 初始化位置和速度
        self.robot_positions = None
        self.robot_velocities = None

        # 障碍物列表
        self.obstacles = []

        # 历史记录
        self.polarization_history = None
        self.robot_trajectory = None
        self.robot_velocities_history = None
        self.neighbors_history = None
        self.algebraic_connectivity_history = None
    
    def initialize_agents(self, scenario_type='uniform'):
        """
        初始化智能体状态

        参数:
            scenario_type: 'uniform' 或 'two_clusters'
        """
        if scenario_type not in ['uniform', 'two_clusters']:
            raise ValueError(f"scenario_type must be 'uniform' or 'two_clusters', got '{scenario_type}'")

        if scenario_type == 'two_clusters':
            print('Initializing agents in Two-Cluster scenario...')
            n_half = self.n_robots // 2
            n_A = n_half
            n_B = self.n_robots - n_half

            # 簇参数
            # 确保簇中心在场地内，留出足够的空间
            cluster_separation = self.field_size / 3  # 簇之间的距离
            cluster_center_A = np.array([self.field_size/2 - cluster_separation/2, self.field_size/2])  # 簇A中心
            cluster_center_B = np.array([self.field_size/2 + cluster_separation/2, self.field_size/2])  # 簇B中心
            cluster_size = min(8, self.field_size / 4)  # 每个簇的方形区域边长，确保不超出边界

            # 簇A的位置
            pos_A = np.random.rand(n_A, 2) * cluster_size + (cluster_center_A - cluster_size/2)

            # 簇B的位置
            pos_B = np.random.rand(n_B, 2) * cluster_size + (cluster_center_B - cluster_size/2)

            self.robot_positions = np.vstack([pos_A, pos_B])
        else:
            # 均匀随机场景（默认）
            print('Initializing agents in Uniform Random scenario...')
            # 智能体在整个场地上均匀分布，留出边界缓冲
            margin = 2.0  # 边界缓冲
            self.robot_positions = (np.random.rand(self.n_robots, 2) *
                                  (self.field_size - 2 * margin) + margin)

        # 使用较大的初始速度，使运动更加明显
        self.robot_velocities = (np.random.rand(self.n_robots, 2) * 2 - 1) * 1.0

    def initialize_obstacles(self, num_obstacles=None, obstacle_list=None, seed=None):
        """
        初始化障碍物

        参数:
            num_obstacles: 随机生成的障碍物数量（如果obstacle_list为None）
            obstacle_list: 自定义障碍物列表（CircularObstacle对象列表）
            seed: 随机种子（用于可重复的随机障碍物生成）
        """
        if obstacle_list is not None:
            # 使用自定义障碍物
            self.obstacles = obstacle_list
            print(f'Using {len(self.obstacles)} custom obstacles')
        else:
            # 随机生成障碍物
            n_obs = num_obstacles if num_obstacles is not None else config.NUM_OBSTACLES
            self.obstacles = create_obstacles(
                self.field_size,
                num_obstacles=n_obs,
                min_radius=config.OBSTACLE_MIN_RADIUS,
                max_radius=config.OBSTACLE_MAX_RADIUS,
                seed=seed
            )
            print(f'Initialized {len(self.obstacles)} obstacles')
    
    def mainloop(self, neighbor_method='knn'):
        """
        主模拟循环

        参数:
            neighbor_method: 'knn' 或 'spotlight'

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
                    selected_neighbors_i = get_neighbor_knn(i, self.robot_positions, self.k_neighbors)
                elif neighbor_method == 'spotlight':
                    selected_neighbors_i = get_neighbor_spotlights(i, self.robot_positions, self.cos_alpha)

                # 存储此智能体和时间步的邻居历史
                self.neighbors_history[i][t] = selected_neighbors_i

                # 标准模拟步骤
                if selected_neighbors_i:
                    nearby_positions_globals = self.robot_positions[selected_neighbors_i]
                    nearby_velocities = self.robot_velocities[selected_neighbors_i]

                    relative_positions = nearby_positions_globals - self.robot_positions[i]

                    # 计算避障力
                    obstacle_force = np.zeros(2)
                    if len(self.obstacles) > 0:
                        obstacle_force = calculate_obstacle_force(
                            self.robot_positions[i],
                            self.robot_velocities[i],
                            self.obstacles,
                            self.safety_dist,
                            self.detection_range,
                            self.force_gain,
                            self.max_obstacle_force
                        )

                    robot_acceleration = calculate_acceleration(
                        relative_positions,
                        self.robot_velocities[i],
                        nearby_velocities,
                        self.lambda_param,
                        self.r_alpha,
                        self.beta_gain,
                        self.umax,
                        obstacle_force=obstacle_force,
                        obstacle_weight=self.obstacle_weight
                    )

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


def run_single_simulation(model_to_run='spotlight'):
    """
    运行单次模拟
    
    参数:
        model_to_run: 'knn' 或 'spotlight'
    """
    print(f'Running single simulation with model: {model_to_run}')
    
    # 创建模拟器实例
    simulator = MultiAgentSimulator()
    
    # 初始化智能体状态
    simulator.initialize_agents(scenario_type='uniform')
    
    # 运行主模拟循环
    result = simulator.mainloop(neighbor_method=model_to_run)
    
    print('Simulation finished.')
    
    return result


if __name__ == "__main__":
    # 示例运行
    result = run_single_simulation(model_to_run='spotlight')
    print("Polarization history shape:", result[0].shape)
    print("Trajectory shape:", result[1].shape)
    print("Velocity history shape:", result[2].shape)
    print("Algebraic connectivity history shape:", result[4].shape)