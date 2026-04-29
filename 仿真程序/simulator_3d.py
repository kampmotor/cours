import numpy as np
from .utils import (
    # 2D函数
    calculate_acceleration,
    calculate_polarization,
    clamp_vector,
    get_neighbor_knn,
    get_neighbor_spotlights,
    calculate_obstacle_force,
    create_obstacles,
    CircularObstacle,
    # 3D函数
    calculate_acceleration_3d,
    calculate_polarization_3d,
    clamp_vector_3d,
    get_neighbor_knn_3d,
    get_neighbor_spotlights_3d,
    calculate_obstacle_force_3d,
    create_obstacles_3d,
    SphericalObstacle,
)
import config


class MultiAgentSimulator3D:
    def __init__(
        self,
        n_robots=None,
        field_size_3d=None,
        umax=None,
        vmax_robot=None,
        dt=None,
        T=None,
        lambda_param=None,
        r_alpha=None,
        beta_gain=None,
        cos_alpha_3d=None,
        k_neighbors=None,
        z_min=None,
        z_max=None,
    ):
        """
        初始化3D多智能体模拟器

        参数:
            n_robots: 智能体数量 (默认: config.N_ROBOTS)
            field_size_3d: 场景大小 (默认: config.FIELD_SIZE_3D)
            umax: 最大加速度 (默认: config.UMAX)
            vmax_robot: 智能体最大速度 (默认: config.VMAX_ROBOT)
            dt: 时间步长 (默认: config.DT)
            T: 模拟时间步数 (默认: config.T)
            lambda_param: 速度一致性项的增益 (默认: config.LAMBDA_PARAM)
            r_alpha: 期望的智能体间距离 (默认: config.R_ALPHA)
            beta_gain: 队形控制项的基础增益 (默认: config.BETA_GAIN)
            cos_alpha_3d: 3D角度余弦阈值 (默认: config.COS_ALPHA_3D)
            k_neighbors: KNN算法的邻居数量 (默认: config.K_NEIGHBORS)
            z_min: 最小飞行高度 (默认: config.Z_MIN)
            z_max: 最大飞行高度 (默认: config.Z_MAX)

        参数取值范围说明请参考 config.py 中的文档
        """
        # 使用配置文件中的默认值
        self.n_robots = n_robots if n_robots is not None else config.N_ROBOTS
        self.field_size_3d = (
            field_size_3d if field_size_3d is not None else config.FIELD_SIZE_3D
        )
        self.umax = umax if umax is not None else config.UMAX
        self.vmax_robot = vmax_robot if vmax_robot is not None else config.VMAX_ROBOT
        self.dt = dt if dt is not None else config.DT
        self.T = T if T is not None else config.T
        self.lambda_param = (
            lambda_param if lambda_param is not None else config.LAMBDA_PARAM
        )
        self.r_alpha = r_alpha if r_alpha is not None else config.R_ALPHA
        self.beta_gain = beta_gain if beta_gain is not None else config.BETA_GAIN
        self.cos_alpha_3d = (
            cos_alpha_3d if cos_alpha_3d is not None else config.COS_ALPHA_3D
        )
        self.k_neighbors = (
            k_neighbors if k_neighbors is not None else config.K_NEIGHBORS
        )
        self.z_min = z_min if z_min is not None else config.Z_MIN
        self.z_max = z_max if z_max is not None else config.Z_MAX

        # 避障参数
        self.safety_dist = config.OBSTACLE_SAFETY_DIST
        self.detection_range = config.OBSTACLE_DETECTION_RANGE
        self.force_gain = config.OBSTACLE_FORCE_GAIN
        self.max_obstacle_force = config.OBSTACLE_MAX_FORCE
        self.obstacle_weight = config.OBSTACLE_PRIORITY_WEIGHT

        # 初始化位置和速度 (3D)
        self.robot_positions = None
        self.robot_velocities = None

        # 3D障碍物列表
        self.obstacles = []

        # 历史记录 (3D)
        self.polarization_history = None
        self.robot_trajectory = None
        self.robot_velocities_history = None
        self.neighbors_history = None
        self.algebraic_connectivity_history = None

    def initialize_agents(self, scenario_type="uniform"):
        """
        初始化智能体状态 (3D版本)

        参数:
            scenario_type: 'uniform', 'two_clusters', 或 'sphere'
        """
        if scenario_type not in ["uniform", "two_clusters", "sphere"]:
            raise ValueError(
                f"scenario_type must be 'uniform', 'two_clusters', or 'sphere', got '{scenario_type}'"
            )

        if scenario_type == "two_clusters":
            print("Initializing agents in Two-Cluster 3D scenario...")
            n_half = self.n_robots // 2
            n_A = n_half
            n_B = self.n_robots - n_half

            # 簇参数 (3D)
            cluster_separation = self.field_size_3d / 3
            cluster_center_A = np.array(
                [
                    self.field_size_3d / 2 - cluster_separation / 2,
                    self.field_size_3d / 2,
                    (self.z_min + self.z_max) / 2,
                ]
            )
            cluster_center_B = np.array(
                [
                    self.field_size_3d / 2 + cluster_separation / 2,
                    self.field_size_3d / 2,
                    (self.z_min + self.z_max) / 2,
                ]
            )
            cluster_size = min(6, self.field_size_3d / 5)  # 簇的立方体区域边长

            # 簇A的位置
            pos_A = np.random.rand(n_A, 3) * cluster_size + (
                cluster_center_A - cluster_size / 2
            )

            # 簇B的位置
            pos_B = np.random.rand(n_B, 3) * cluster_size + (
                cluster_center_B - cluster_size / 2
            )

            self.robot_positions = np.vstack([pos_A, pos_B])

        elif scenario_type == "sphere":
            print("Initializing agents in Spherical 3D scenario...")
            # 在球体内均匀分布智能体
            center = np.array(
                [
                    self.field_size_3d / 2,
                    self.field_size_3d / 2,
                    (self.z_min + self.z_max) / 2,
                ]
            )
            radius = min(self.field_size_3d, self.z_max - self.z_min) / 3

            positions = []
            for i in range(self.n_robots):
                # 使用均匀分布的球坐标
                r = radius * (np.random.random() ** (1 / 3))  # 体积均匀分布
                theta = np.random.random() * 2 * np.pi  # 方位角
                phi = np.arccos(2 * np.random.random() - 1)  # 极角

                x = center[0] + r * np.sin(phi) * np.cos(theta)
                y = center[1] + r * np.sin(phi) * np.sin(theta)
                z = center[2] + r * np.cos(phi)

                # 确保在边界内
                x = np.clip(x, 2.0, self.field_size_3d - 2.0)
                y = np.clip(y, 2.0, self.field_size_3d - 2.0)
                z = np.clip(z, self.z_min + 1.0, self.z_max - 1.0)

                positions.append([x, y, z])

            self.robot_positions = np.array(positions)

        else:
            # 均匀随机场景（默认）
            print("Initializing agents in Uniform Random 3D scenario...")
            margin = 2.0
            self.robot_positions = np.zeros((self.n_robots, 3))

            # x, y坐标
            self.robot_positions[:, :2] = (
                np.random.rand(self.n_robots, 2) * (self.field_size_3d - 2 * margin)
                + margin
            )

            # z坐标（在高度范围内）
            z_range = self.z_max - self.z_min
            self.robot_positions[:, 2] = (
                np.random.rand(self.n_robots) * (z_range - 2 * margin)
                + self.z_min
                + margin
            )

        # 初始化3D速度
        self.robot_velocities = (np.random.rand(self.n_robots, 3) * 2 - 1) * 1.0

    def initialize_obstacles(self, num_obstacles=None, obstacle_list=None, seed=None):
        """
        初始化3D障碍物

        参数:
            num_obstacles: 随机生成的障碍物数量（如果obstacle_list为None）
            obstacle_list: 自定义障碍物列表（SphericalObstacle对象列表）
            seed: 随机种子（用于可重复的随机障碍物生成）
        """
        if obstacle_list is not None:
            # 使用自定义障碍物
            self.obstacles = obstacle_list
            print(f"Using {len(self.obstacles)} custom 3D obstacles")
        else:
            # 随机生成3D障碍物
            n_obs = (
                num_obstacles if num_obstacles is not None else config.NUM_OBSTACLES_3D
            )
            self.obstacles = create_obstacles_3d(
                self.field_size_3d,
                num_obstacles=n_obs,
                min_radius=config.OBSTACLE_MIN_RADIUS_3D,
                max_radius=config.OBSTACLE_MAX_RADIUS_3D,
                z_min=self.z_min,
                z_max=self.z_max,
                seed=seed,
            )
            print(f"Initialized {len(self.obstacles)} 3D obstacles")

    def mainloop(self, neighbor_method="knn"):
        """
        主模拟循环 (3D版本)

        参数:
            neighbor_method: 'knn' 或 'spotlight'

        返回:
            polarization_history: 极化度历史
            robot_trajectory: 智能体轨迹 (3D)
            robot_velocities_history: 速度历史 (3D)
            neighbors_history: 邻居历史
            algebraic_connectivity_history: 代数连通性历史
        """
        if neighbor_method not in ["knn", "spotlight"]:
            raise ValueError(
                f"neighbor_method must be 'knn' or 'spotlight', got '{neighbor_method}'"
            )

        if self.robot_positions is None or self.robot_velocities is None:
            raise RuntimeError(
                "Agents not initialized. Call initialize_agents() first."
            )

        # 初始化结果数组 (3D)
        self.polarization_history = np.zeros(self.T)
        self.robot_trajectory = np.zeros((self.n_robots, 3, self.T))
        self.robot_velocities_history = np.zeros((self.n_robots, 3, self.T))
        self.neighbors_history = [
            [[] for _ in range(self.T)] for _ in range(self.n_robots)
        ]
        self.algebraic_connectivity_history = np.zeros(self.T)

        # 主模拟循环
        for t in range(self.T):
            # 内循环更新每个智能体
            for i in range(self.n_robots):
                # 根据输入选择邻居查找算法 (3D版本)
                selected_neighbors_i = []
                if neighbor_method == "knn":
                    selected_neighbors_i = get_neighbor_knn_3d(
                        i, self.robot_positions, self.k_neighbors
                    )
                elif neighbor_method == "spotlight":
                    selected_neighbors_i = get_neighbor_spotlights_3d(
                        i,
                        self.robot_positions,
                        self.robot_velocities,
                        self.cos_alpha_3d,
                    )

                # 存储此智能体和时间步的邻居历史
                self.neighbors_history[i][t] = selected_neighbors_i

                # 标准模拟步骤 (3D版本)
                if selected_neighbors_i:
                    nearby_positions_globals = self.robot_positions[
                        selected_neighbors_i
                    ]
                    nearby_velocities = self.robot_velocities[selected_neighbors_i]

                    relative_positions = (
                        nearby_positions_globals - self.robot_positions[i]
                    )

                    # 计算3D避障力
                    obstacle_force = np.zeros(3)
                    if len(self.obstacles) > 0:
                        obstacle_force = calculate_obstacle_force_3d(
                            self.robot_positions[i],
                            self.robot_velocities[i],
                            self.obstacles,
                            self.safety_dist,
                            self.detection_range,
                            self.force_gain,
                            self.max_obstacle_force,
                        )

                    robot_acceleration = calculate_acceleration_3d(
                        relative_positions,
                        self.robot_velocities[i],
                        nearby_velocities,
                        self.lambda_param,
                        self.r_alpha,
                        self.beta_gain,
                        self.umax,
                        obstacle_force,
                        self.obstacle_weight,
                    )
                else:
                    # 没有邻居时只考虑避障
                    if len(self.obstacles) > 0:
                        obstacle_force = calculate_obstacle_force_3d(
                            self.robot_positions[i],
                            self.robot_velocities[i],
                            self.obstacles,
                            self.safety_dist,
                            self.detection_range,
                            self.force_gain,
                            self.max_obstacle_force,
                        )
                        robot_acceleration = clamp_vector_3d(obstacle_force, self.umax)
                    else:
                        robot_acceleration = np.zeros(3)

                # 更新速度和位置 (3D版本)
                self.robot_velocities[i] += robot_acceleration * self.dt
                self.robot_velocities[i] = clamp_vector_3d(
                    self.robot_velocities[i], self.vmax_robot
                )

                self.robot_positions[i] += self.robot_velocities[i] * self.dt

                # 3D边界处理
                self.robot_positions[i][0] = np.clip(
                    self.robot_positions[i][0], 0, self.field_size_3d
                )
                self.robot_positions[i][1] = np.clip(
                    self.robot_positions[i][1], 0, self.field_size_3d
                )
                self.robot_positions[i][2] = np.clip(
                    self.robot_positions[i][2], self.z_min, self.z_max
                )

            # 记录历史数据
            self.robot_trajectory[:, :, t] = self.robot_positions
            self.robot_velocities_history[:, :, t] = self.robot_velocities
            self.polarization_history[t] = calculate_polarization_3d(
                self.robot_velocities
            )

            # 计算代数连通性（简化版本）
            self.algebraic_connectivity_history[t] = (
                self._calculate_algebraic_connectivity_3d()
            )

        return (
            self.polarization_history,
            self.robot_trajectory,
            self.robot_velocities_history,
            self.neighbors_history,
            self.algebraic_connectivity_history,
        )

    def _calculate_algebraic_connectivity_3d(self):
        """
        计算3D场景的代数连通性（简化版本）
        """
        n = self.n_robots
        if n <= 1:
            return 0.0

        # 构建邻接矩阵（基于距离）
        adjacency_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                distance = np.linalg.norm(
                    self.robot_positions[i] - self.robot_positions[j]
                )
                # 如果距离小于某个阈值，认为有连接
                if distance < self.r_alpha * 2:  # 使用2倍期望距离作为连接阈值
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1

        # 计算度矩阵
        degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))

        # 计算拉普拉斯矩阵
        laplacian_matrix = degree_matrix - adjacency_matrix

        # 计算特征值
        eigenvalues = np.linalg.eigvals(laplacian_matrix)
        eigenvalues = np.real(eigenvalues)  # 取实部

        # 代数连通性是第二小的特征值
        eigenvalues_sorted = np.sort(eigenvalues)
        if len(eigenvalues_sorted) > 1:
            return max(0.0, eigenvalues_sorted[1])  # 确保非负
        else:
            return 0.0


def run_single_simulation_3d(
    model_to_run="knn", n_robots=None, scenario_type="uniform"
):
    """
    运行单次3D模拟的便捷函数

    参数:
        model_to_run: 'knn' 或 'spotlight'
        n_robots: 智能体数量（可选）
        scenario_type: 初始化场景类型

    返回:
        模拟结果元组
    """
    simulator = MultiAgentSimulator3D(n_robots=n_robots)
    simulator.initialize_agents(scenario_type=scenario_type)
    simulator.initialize_obstacles()  # 可选

    result = simulator.mainloop(neighbor_method=model_to_run)
    return result
