"""
Copyright (c) 2024 WindyLab of Westlake University, China
All rights reserved.

This software is provided "as is" without warranty of any kind, either
express or implied, including but not limited to the warranties of
merchantability, fitness for a particular purpose, or non-infringement.
In no event shall the authors or copyright holders be liable for any
claim, damages, or other liability, whether in an action of contract,
tort, or otherwise, arising from, out of, or in connection with the
software or the use or other dealings in the software.
"""

import mujoco
import numpy as np
import imageio

# 定义包含5个小球和一个连接的XML字符串
xml = """
<mujoco>
  <worldbody>
    <light name="light" pos="0 0 5"/>

    <!-- 五个小球 -->
    <body name="ball1" pos="-0.2 0.2 0.3">
      <freejoint/>
      <geom type="sphere" size="0.05" rgba="1 0 0 1"/>
    </body>
    <body name="ball2" pos="0.2 0.2 0.3">
      <freejoint/>
      <geom type="sphere" size="0.05" rgba="0 1 0 1"/>
    </body>
    <body name="ball3" pos="0 -0.2 0.3">
      <freejoint/>
      <geom type="sphere" size="0.05" rgba="0 0 1 1"/>
    </body>
    <body name="ball4" pos="-0.2 -0.2 0.3">
      <freejoint/>
      <geom type="sphere" size="0.05" rgba="1 1 0 1"/>
    </body>
    <body name="ball5" pos="0.2 -0.2 0.3">
      <freejoint/>
      <geom type="sphere" size="0.05" rgba="0 1 1 1"/>
    </body>

    <!-- 连接ball1和ball2的连杆 -->
    <body name="link" pos="0 0.2 0.3">
      <joint type="free"/>
      <geom type="capsule" fromto="-0.2 0.2 0.3 0.2 0.2 0.3" size="0.02" rgba="0.5 0.5 0.5 1"/>
    </body>

  </worldbody>
</mujoco>
"""

# 加载模型和数据
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# 设置渲染器
frames = []
with mujoco.Renderer(model) as renderer:
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)

    # 运行仿真并收集帧
    for _ in range(300):
        mujoco.mj_step(model, data)
        renderer.update_scene(data)
        frame = renderer.render()
        frames.append(frame)

# 保存为GIF
imageio.mimsave("/tmp/simulation.gif", frames, fps=30)
