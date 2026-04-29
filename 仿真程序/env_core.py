__credits__ = ["lijianan@westlake.edu.cn"]

import gym
from gym.utils import seeding
from gym import error, spaces, utils
#from .putils import *
from gym.envs.pps.putils.prop import *
from gym.envs.pps.putils.putils import *
from gym.envs.pps.putils.param import *
import numpy as np
import os
import torch
import random
from gym.wrappers import NJP

class PredatorPreySwarmEnv(PredatorPreySwarmEnvProp):
    """
    Description:
        Multiple predators and prey interact with each other. If predators catch prey,
        predators receive positive rewards, while prey receive negative rewards.

    Source:
        This environment appeared first in the paper Li J, Li L, Zhao S.
        "Predator–prey survival pressure is sufficient to evolve swarming behaviors",
        New Journal of Physics, vol. 25, no. 9, pp. 092001, 2023.

    Observation:
        Type: Box(...)
        If in Cartesian mode:
            [ agent's own pos., vel.,
              relative pos. of observed pursuers,
              relative pos. of observed escapers  ]

        If in Polar mode:
            [ agent's own pos., vel., heading,
            relative pos. and headings of observed pursuers,
            relative pos. and headings of observed escapers ]

        Observation model is dependent on both metric and topological distance.
        Metric distance: an agent can only perceive others in its perception range which is assumed to be a disk with a pre-defined radius.
        Topological distance: how many at most an agent can perceive concurrently rather than how far away.

    Actions:
        Type: Box(2)
        If the dynamics mode for agents is Cartesian, then
        Num   Action
        0     acceleration in x-axis
        1     acceleration in y-axis

        If the dynamics mode for agents is Polar, then
        Num   Action
        0     angular velocity (or rotation angle in the given time step)
        1     acceleration in heading direction

        Note: The min and max values for the action values can be adjusted, but we strongly
        advise against doing so, as this adjustment is closely tied to the update time step.
        Incorrectly setting these values may result in a violation of physical laws and the
        environment dynamics may behave weirdly.

    Reward:
       The core reward is as follows: when a predator catches its prey, the predator receives
       a reward of +1, while the prey receives a reward of -1. For details on the other
       auxiliary rewards, please refer to the reward function.

    Starting State:
        All observations are assigned a uniform random value.

    """

    param_list = params

    def __init__(self, n_p=20):

        self._n_p = n_p
        self._n_o = 0
        self._n_po = self._n_p + self._n_o
        self.nearest_neighbor_directions = np.zeros(self._n_p)
        self._size = 0.01
        self._size_o = 0.05
        self._size_p = 0.01
        self.d_e = 0.05
        self._sensitivity = 0.5
        self._angle_p_max = 1.57

        #self._k_ball = 1
        self._m = get_mass(self._m_p, self._m_o, self._n_p, self._n_o)  ##质量 【1，1，2，2】
        self._I = self._m * (self._size_p * 2.5)**2
        self.w = 0
        self._size, self._sizes = get_sizes(self._size_p, self._size_o, self._n_p,
                                               self._n_o)  ##size 每个物体的大小【   】  sizes是二维矩阵存储第i与第j个之间的尺寸之和
        if self._dynamics_mode == 'Cartesian':
            self.max_energy_p = 1000.
        elif self._dynamics_mode == 'Polar':
            self.max_energy_p = 1000.
        self.viewer = None
        self.seed()

    def __reinit__(self):
        self._n_po = self._n_p + self._n_o
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self._m = get_mass(self._m_p, self._m_o, self._n_p, self._n_o)  ##质量 【1，1，2，2】
        self._size, self._sizes = get_sizes(self._size_p, self._size_o, self._n_p, self._n_o)##size 每个物体的大小【   】  sizes是二维矩阵存储第i与第j个之间的尺寸之和
        if self._billiards_mode:##碰撞的弹性
            self._c_wall = 0.1##与墙
            self._c_aero = 0.02##与空气

        if self._dynamics_mode == 'Cartesian':
            self._linAcc_p_min = -1 #线加速度下线-1
            if self._linAcc_p_max != 1 :##线加速度上线1
                raise ValueError('Currently in Cartesian mode, linAcc_p_max and linAcc_e_max have to be 1')
            assert (self._linAcc_p_min, self._linAcc_p_max) == (-1, 1)
        elif self._dynamics_mode == 'Polar':
            self._linAcc_p_min = 0 ##线加速度设置为0  极坐标下运动以角度和距离

        # Energy
        if self._dynamics_mode == 'Cartesian':
            self.max_energy_p = 1000.
        elif self._dynamics_mode == 'Polar':
            self.max_energy_p = 1000.

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        max_size = self._size
        max_respawn_times = 100
        for respawn_time in range(max_respawn_times):
            self._p = np.random.uniform(-1 + max_size, 1 - max_size, (2, self._n_po))  # 初始化智能体位置self._p
            if self._obstacles_is_constant:
                self._p[:, self._n_p:self._n_po] = self._p_o
            self._d_b2b_center, _, _is_collide_b2b = get_dist_b2b(self._p, self._L, self._is_periodic, self._sizes)
            if _is_collide_b2b.sum() == 0:
                break
            if respawn_time == max_respawn_times - 1:
                print('Some particles are overlapped at the initial time !')
        if self._render_traj == True:
            self._p_traj = np.zeros((self._traj_len, 2, self._n_po))
            self._p_traj[0, :, :] = self._p
        self._dp = np.zeros((2, self._n_po))##速度
        if self._billiards_mode:##壁球和极坐标不兼容
            self._dp = np.random.uniform(-1, 1, (2, self._n_po))  # ice mode
            if self._dynamics_mode == 'Polar':
                raise ValueError("Billiards_mode requires dynamics_mode be 'Cartesian' !")
        if self._obstacles_cannot_move:
            self._dp[:, self._n_p:self._n_po] = 0
        self._ddp = np.zeros((2, self._n_po))

        # self._energy = np.array(
        #     [self.max_energy_p for _ in range(self._n_p)]).reshape(1,self._n_p)  ##为每个智能体这是初始能量
        if self._dynamics_mode == 'Polar':
            self._theta = np.pi * np.random.uniform(-1, 1, (1, self._n_po))##生成智能体角度 随机（-1，1）然后乘以pi  （-p，p)  （1*n）
            # self._theta = np.pi * np.zeros((1, self._n_peo))
            self._heading = np.concatenate((np.cos(self._theta), np.sin(self._theta)), axis=0) #计算智能体朝向 x方向y方向  （2*n）
        return self._get_obs()

    def _get_obs(self):

        self.obs = np.zeros(self.observation_space.shape)
        # print("os",self.observation_space.shape)
        for i in range(self._n_p):
            ''' For pursuers

            If in Cartesian mode:
            [ agent's own pos., vel.,
              relative pos. of observed pursuers,
              relative pos. of observed escapers  ]

            If in Polar mode:
            [ agent's own pos., vel., heading,
              relative pos. and headings of observed pursuers,
              relative pos. and headings of observed escapers ]

            Observation model is dependent on both metric and topological distance.
            Metric distance means: an agent can only perceive others in its perception range which is assumed to be a disk with a pre-defined radius.
            Topological distancemeans: how many at most     an agent can perceive concurrently rather than how far away.
            '''
            relPos_p2p = self._p[:, :self._n_p] - self._p[:, [i]]  #2*n的数组 存储第i个智能体与其他智能体的x距离y距离0
            if self._is_periodic: relPos_p2p = make_periodic(relPos_p2p, self._L)  ## 计算位置差
            relVel_p2p = self._dp[:, :self._n_p] - self._dp[:,[i]] \
                if self._dynamics_mode == 'Cartesian' else self._heading[:,
                                                                                                   :self._n_p] - self._heading[
                                                                                                                 :, [i]]##笛卡尔计算速度差 极坐标计算朝向差
            relPos_p2p, relVel_p2p = get_focused(relPos_p2p, relVel_p2p, self._FoV_p, self._topo_n_p2p, True)



            obs_pursuer_pos = np.concatenate((self._p[:, [i]], relPos_p2p), axis=1)
            obs_pursuer_vel = np.concatenate((self._dp[:, [i]], relVel_p2p), axis=1)
            obs_pursuer = np.concatenate((obs_pursuer_pos, obs_pursuer_vel),
                                         axis=0)  # (4, n_peo+1) FIXME: only suitable for no obstacles  不包含障碍物的观测
            if self._dynamics_mode == 'Cartesian':
                self.obs[:self.obs_dim_pursuer, i] = obs_pursuer.T.reshape(-1)
            elif self._dynamics_mode == 'Polar':
                self.obs[:self.obs_dim_pursuer - 2, i] = obs_pursuer.T.reshape(-1)
                self.obs[self.obs_dim_pursuer - 2:self.obs_dim_pursuer, i] = self._heading[:, i]
        # print("o",self.obs)
        return self.obs

    def _get_reward(self, a):
        # 初始化捕食者奖励
        reward_p = -1.0 * self._is_collide_b2b[:self._n_p, :self._n_p].sum(axis=0, keepdims=True).astype(float)
        # 找到每个智能体的最近 10 个邻居索引
        nearest_idx = np.argsort(self.d_b2b_edge, axis=1)[:, 1:11]

        # 遍历每个智能体，计算速度一致性奖励
        for i in range(self._n_p):
            consistency_sum = 0
            for j in nearest_idx[i]:
                # 计算点积除以模的公式
                dot_product = np.dot(self._dp[:, i], self._dp[:, j])  # 点积
                norm_i = np.linalg.norm(self._dp[:, i])  # 智能体 i 的速度模
                norm_j = np.linalg.norm(self._dp[:, j])  # 智能体 j 的速度模
                # 避免除以 0 的情况
                if norm_i > 0 and norm_j > 0:
                    speed_consistency = dot_product / (norm_i * norm_j)
                else:
                    speed_consistency = 0  # 如果速度为零，则一致性为 0
                consistency_sum += speed_consistency

            # 平均速度一致性
            avg_consistency = consistency_sum / 10
            reward_p[0, i] += avg_consistency  # 一致性越高，奖励越大

        if self._penalize_distance:
            matrix = self._d_b2b_center[:self._n_p, :self._n_p]
            matrix[matrix < 0.1] = 0
            matrix[matrix > 0.1] = 1
            # print("a",a)
            reward_p += - matrix.sum(axis=0, keepdims=True)

        if self._penalize_control_effort:
            if self._dynamics_mode == 'Cartesian':
                reward_p -= 1 * np.sqrt(a[[0], :self._n_p] ** 2 + a[[1], :self._n_p] ** 2)
            elif self._dynamics_mode == 'Polar':
                reward_p -= 1 * np.abs(a[[0], :self._n_p]) + 0 * np.abs(a[[1], :self._n_p])

        if self._penalize_collide_obstacles:
            reward_p -= 5 * self._is_collide_b2b[self._n_p:self._n_po, :self._n_p].sum(axis=0, keepdims=True)

        # 惩罚与墙壁碰撞
        if self._penalize_collide_walls and not self._is_periodic:
            reward_p -= 5 * self.is_collide_b2w[:, :self._n_p].sum(axis=0, keepdims=True)

        # 奖励分享模式
        if self._reward_sharing_mode == 'sharing_mean':
            reward_p[:] = np.mean(reward_p)
        elif self._reward_sharing_mode == 'sharing_max':
            reward_p[:] = np.max(reward_p)
        elif self._reward_sharing_mode == 'individual':
            pass
        else:
            print('reward mode error !!')
        return reward_p

    def _get_done(self):
        # distances = np.linalg.norm(self._p, axis=0)
        # all_done =((distances <= 0.18)).astype(bool)
        # all_done = np.expand_dims(all_done, axis=0)

        all_done = np.zeros((1, self._n_p)).astype(bool)
        return all_done
        # return False

    def _get_info(self):
        # 计算捕食者之间的距离矩阵
        dist_matrix = self._d_b2b_center[:self._n_p, :self._n_p]
        dist_matrix += 10 * np.identity(self._n_p)  # 防止自距影响最小值计算
        # 计算每个捕食者到其最近邻的平均最小距离
        ave_min_dist = np.mean(np.min(dist_matrix, axis=0))
        DoC = 1 / ave_min_dist  # 密集度指标

        consistency_sum = 0
        for i in range(self._n_p):
            nearest_idx = np.argmin(dist_matrix[i], axis=0)
            dot_product = np.dot(self._dp[:, i], self._dp[:, nearest_idx])  # 点积
            norm_i = np.linalg.norm(self._dp[:, i])  # 智能体 i 的速度模
            norm_j = np.linalg.norm(self._dp[:, nearest_idx])  # 智能体 最近 的速度模
            if norm_i > 0 and norm_j > 0:
                speed_consistency = dot_product / (norm_i * norm_j)
            else:
                speed_consistency = 0  # 如果速度为零，则一致性为 0
            consistency_sum += speed_consistency
        DoA = 1 / consistency_sum

        DoA_global_sum = 0
        pair_count = 0
        for i in range(self._n_p):
            for j in range(i + 1, self._n_p):
                dot_product = np.dot(self._dp[:, i], self._dp[:, j])
                norm_i = np.linalg.norm(self._dp[:, i])
                norm_j = np.linalg.norm(self._dp[:, j])

                if norm_i > 0 and norm_j > 0:
                    speed_consistency = dot_product / (norm_i * norm_j)
                else:
                    speed_consistency = 0

                DoA_global_sum += speed_consistency
                pair_count += 1

        DoA_global = DoA_global_sum / pair_count


        # 可以添加全局密集度的计算
        ave_dist = self._d_b2b_center[:self._n_p, :self._n_p].sum() / (self._n_p * (self._n_p - 1))
        DoC_global = 1 / ave_dist
        # print("c",DoC)
        # print("A",DoA)
        # print("AG",DoA_global)
        # print("CG",DoC_global)
        return np.array( [DoC,DoA,DoA_global,DoC_global] ).reshape(4,1)


    def step(self, a):
        #print("a", a)
        for _ in range(self._n_frames):#self._n_frames=1
            if self._dynamics_mode == 'Polar':
                a[0, :self._n_p] *= 0.001 ##把a0转化为角速度  self._angle_p_max=0.5
                a[1, :self._n_p] = (self._linAcc_p_max - self._linAcc_p_min) / 2 * a[1, :self._n_p] + (
                            self._linAcc_p_max + self._linAcc_p_min) / 2##把a1【-1，1】到线加速度最大最小值 max=1 min=0
            self._d_b2b_center, self.d_b2b_edge, self._is_collide_b2b = get_dist_b2b(self._p, self._L,
                                                                                     self._is_periodic, self._sizes)
            #print("d",self._d_b2b_center)
            sf_b2b_out = np.zeros((2 * self._n_po, self._n_po))##存储凝聚力
            sf_b2b_in = np.zeros((2 * self._n_po, self._n_po))##存储斥力
            for i in range(self._n_po):
                for j in range(i):
                    delta = self._p[:, j] - self._p[:, i]#计算位置差
                    if self._is_periodic:
                        delta = make_periodic(delta, self._L)
                    dir = delta / self._d_b2b_center[i, j]
                    # print("dir",dir)##位置差除以两个智能体的中心距离 计算方向向量
                    #sf_b2b_all[2 * i:2 * (i + 1), j] = self._is_collide_b2b[i, j] * self.d_b2b_edge[
                    if self.d_b2b_edge[i,j] < 0.01 :
                        sf_b2b_out[2 * i:2 * (i + 1), j] = (0.01 - self.d_b2b_edge[i,j]) * 10 * (-dir)  ##20*10   2*i与2*i+1是代表该智能体的力
                        sf_b2b_out[2 * j:2 * (j + 1), i] = - sf_b2b_out[2 * i:2 * (i + 1), j]

                    # sf_b2b_in[2 * i:2 * (i + 1), j] = ((1 / (self.d_b2b_edge[i, j] ** 3))) * 0.0001 * (dir)  ##20*10   2*i与2*i+1是代表该智能体的力
                    # sf_b2b_in[2 * j:2 * (j + 1), i] = - sf_b2b_in[2 * i:2 * (i + 1), j]
            sf_b2b_o = np.sum(sf_b2b_out, axis=1, keepdims=True).reshape(self._n_po, 2).T #使得每个球的受力是一个列
            #print("b2b",sf_b2b_o)
            # print("so",sf_b2b_o)
            # sf_b2b_i = np.sum(sf_b2b_in, axis=1, keepdims=True).reshape(self._n_po, 2).T
            # print("si",sf_b2b_i)
            if self._is_periodic == False:
                self.d_b2w, self.is_collide_b2w = get_dist_b2w(self._p, self._size, self._L)
                sf_b2w = self.d_b2w
                sf_b2w = np.where(
                    sf_b2w > 0.1,  # 如果值大于 0.2
                    0,  # 置为 0
                    np.where(
                        sf_b2w > 0.01,  # 如果值小于等于 0.2 且大于 0.05
                        1 / sf_b2w,  # 置为倒数
                        1 / (sf_b2w ** 2)  # 如果值小于等于 0.05，置为平方的倒数
                    )
                    )
                #print("sd",sf_b2w)
                sf_b2w = np.array([[-1, 0, 1, 0], [0, 1, 0, -1]]).dot(0.01*sf_b2w)
                #print("sf",sf_b2w)
                # df_b2w = np.array([[-1, 0, -1, 0], [0, -1, 0, -1]]).dot(
                #     self.is_collide_b2w * np.concatenate((self._dp, self._dp), axis=0)) * self._c_wall


            if self.pursuer_strategy == 'input':##外部输出a
                pass
            elif self.pursuer_strategy == 'static':##静态
                a[:, :self._n_p] = np.zeros((self._act_dim_pursuer, self._n_p))
            elif self.pursuer_strategy == 'random':##随机
                a[:, :self._n_p] = np.random.uniform(-1, 1, (self._act_dim_pursuer, self._n_p))
                if self._dynamics_mode == 'Polar':
                    a[0, :self._n_p] *= self._angle_p_max
                    a[1, :self._n_p] = (self._linAcc_p_max - self._linAcc_p_min) / 2 * a[1, :self._n_p] + (
                                self._linAcc_p_max + self._linAcc_p_min) / 2

            else:
                print('Wrong in Step function')

            if self._dynamics_mode == 'Cartesian':
                u = a
            elif self._dynamics_mode == 'Polar':
                wa = a[[0], :] / self._I  ##角加速度等于力矩除以转动惯量
                self._theta += self.w * self._dt +0.5 * wa * self._dt **2  #角度进行叠加
                self.w += wa * self._dt
                self._theta = normalize_angle(self._theta)#归一化
                self._heading = np.concatenate((np.cos(self._theta), np.sin(self._theta)), axis=0) ##计算朝向

                u = a[[1], :] * self._heading#表示径向速度或力，与朝向向量相乘，得到二维空间中的实际速度向量或力向量 u。
            else:
                print('Wrong in updating dynamics')

            if self._is_periodic == True:
                F = self._sensitivity * u + sf_b2b_o - self._c_aero * self._dp#self=1
                # F = self._sensitivity * u  + sf_b2b + df_b2b - self._c_aero*dp
            elif self._is_periodic == False:
                F = self._sensitivity * u  + sf_b2b_o - self._c_aero * self._dp +sf_b2w

            else:
                print('Wrong in considering walls !!!')
            self._ddp = F / self._m
            self._dp += self._ddp * self._dt
            if self._obstacles_cannot_move:
                self._dp[:, self._n_p:self._n_po] = 0
            self._dp[:, :self._n_p] = np.clip(self._dp[:, :self._n_p], -self._linVel_p_max, self._linVel_p_max)
            self._p += self._dp * self._dt
            if self._obstacles_is_constant:
                self._p[:, self._n_p:self._n_po] = self._p_o
            if self._is_periodic:
                self._p = make_periodic(self._p, self._L)

            if self._render_traj == True:
                self._p_traj = np.concatenate((self._p_traj[1:, :, :], self._p.reshape(1, 2, self._n_po)), axis=0)
        return self._get_obs(), self._get_reward(a), self._get_done(), self._get_info()

        # TODO: obstacle or shelter

    # ============== ================= =====================

    def render(self, mode="human"):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-1, 1, -1, 1)

            agents = []
            self.tf = []
            if self._render_traj: self.trajrender = []
            for i in range(self._n_p):
                if self._render_traj: self.trajrender.append(
                    rendering.Traj(list(zip(self._p_traj[:, 0, i], self._p_traj[:, 1, i])), False))
                if i < self._n_p:
                    if self._dynamics_mode == 'Polar':
                        agents.append(rendering.make_unicycle(self._size_p))
                    elif self._dynamics_mode == 'Cartesian':
                        agents.append(rendering.make_circle(self._size_p))
                    agents[i].set_color_alpha(1, 0.5, 0, 1)
                    if self._render_traj: self.trajrender[i].set_color_alpha(1, 0.5, 0, 0.5)
                self.tf.append(rendering.Transform())
                agents[i].add_attr(self.tf[i])
                self.viewer.add_geom(agents[i])
                if self._render_traj: self.viewer.add_geom(self.trajrender[i])

            # # 下部弧形
            # bottom_circle = rendering.make_circle(0.18, 30, False)  # 设置 filled=False，绘制空心圆
            # bottom_circle_transform = rendering.Transform(translation=(0, 0))
            # bottom_circle.add_attr(bottom_circle_transform)
            # bottom_circle.set_color(0, 0, 0)  # 黑色
            # self.viewer.add_geom(bottom_circle)

        for i in range(self._n_p):
            if self._dynamics_mode == 'Polar':
                self.tf[i].set_rotation(self._theta[0, i])
            elif self._dynamics_mode == 'Cartesian':
                pass
            self.tf[i].set_translation(self._p[0, i], self._p[1, i])
            if self._render_traj: self.trajrender[i].set_traj(list(zip(self._p_traj[:, 0, i], self._p_traj[:, 1, i])))

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _get_observation_space(self):
        # self._topo_n_p = self._topo_n_p2p
        #self._topo_n_p = self.topo_n_p2p
        self._topo_n_p = self.topo_n_p2p
        self.obs_dim_pursuer = (2 + 2 * self._topo_n_p) * 2 ##位置 速度 +能量
        if self._dynamics_mode == 'Polar':
            self.obs_dim_pursuer += 2
        observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim_pursuer, self._n_p), dtype=np.float32)
        return observation_space

    def _get_action_space(self):
        _act_dim_max = np.max([self._act_dim_pursuer])
        action_space = spaces.Box(low=-1, high=1, shape=(_act_dim_max, self._n_p), dtype=np.float32)
        return action_space

    def _find_nearest_neighbors_DOA(self, x, i):

        distances = []#存储i与所有智能体的方向距离
        for j in range(np.shape(x)[1]):
            if j != i:
                distances.append(np.linalg.norm(x[:, i] + x[:, j]))##第i个智能呢个体和第j个智能体的相加 然后linalg。norm计算莫长

        return np.min(distances)
# if __name__ == '__main__':
#     env = PredatorPreySwarmEnv()
#     Pos = np.array([[1, 2, 3, 0, 1],
#                     [2, 3, 4, 2, 2.3]])
#     Vel = np.array([[1, 2, 3, 4, 5],
#                     [1, 2, 3, 4, 5]])
#     print(Pos)
#     print(Vel)
#     threshold = 5
#     desired_n = 2
#     get_focused(Pos, Vel, threshold, desired_n, False)
# if __name__ == '__main__':
#     scenario_name = 'PredatorPreySwarm-v0'
#     base_env = gym.make(scenario_name).unwrapped
#     # env = NJP(base_env, args)
#     custom_param = 'custom_param.json'
#     custom_param = os.path.dirname(os.path.realpath(__file__)) + '/' + custom_param
#     env = NJP(base_env, custom_param)
#     env.reset()
#
#     # 运行 10 个时间步，每个时间步之后进行渲染
#     for _ in range(10):
#         action = env.action_space.sample()  # 随机生成一个动作
#         env.step(action)  # 进行一步
#         env.render()  # 可视化当前状态
#
#     env.close()  # 关闭渲染器

