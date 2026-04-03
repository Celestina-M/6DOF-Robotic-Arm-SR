import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import math

from robotic_arm_new import RoboticArm, fast_fk_computation

class Visualizer:
    def __init__(self, robotic_arm):
        """可视化器初始化：接收机械臂实例"""
        self.arm = robotic_arm
        
        # 在初始化时就创建唯一的画布，杜绝多窗口问题
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 定义这三个关键变量，用于后续动画的高速更新
        self.arm_line = None
        self.base_point = None
        self.end_point = None

        self.setup_plot()

    def setup_plot(self):
        """设置 3D 绘图环境（因为移到了__init__，这里只负责视觉设置）"""
        self.ax.set_xlabel('X (m)', fontweight='bold')
        self.ax.set_ylabel('Y (m)', fontweight='bold')
        self.ax.set_zlabel('Z (m)', fontweight='bold')

        # 改进视觉效果：加个网格
        self.ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        max_range = 1.0 
        self.ax.set_xlim([-max_range, max_range])
        self.ax.set_ylim([-max_range, max_range])
        self.ax.set_zlim([0, max_range*2])  

        self.ax.view_init(elev=20, azim=45)
        self.ax.set_title('6-DOF Robotic Arm (≥30 FPS)')

    def draw_arm(self, joint_angles=None):
        """
        绘制机械臂 (支持首次绘制和后续高速更新)
            - joint_angles: 可选参数，直接传入关节角度数组（如果不传则使用当前机械臂状态）
            TODO: 绘制机械臂的连杆和关节
        """    
        if joint_angles is not None:
            self.arm.set_joint_angles(joint_angles)

        joint_positions = np.array(self.arm.forward_kinematics())

        # 如果是第一帧，画出实线；如果不是，只更新坐标数据
        if self.arm_line is None:
            # 改进视觉效果：调整配色和粗细
            self.arm_line, = self.ax.plot(
                joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2], 
                color='#2C3E50', marker='o', markerfacecolor='#E74C3C', 
                markersize=8, linewidth=4, label='Robotic Arm'
            )
            self.base_point, = self.ax.plot(
                [joint_positions[0, 0]], [joint_positions[0, 1]], [joint_positions[0, 2]], 
                'ks', markersize=12, label='Base'
            )
            self.end_point, = self.ax.plot(
                [joint_positions[-1, 0]], [joint_positions[-1, 1]], [joint_positions[-1, 2]], 
                '^', color='#27AE60', markersize=12, label='End Effector'
            )
            self.ax.legend()
        else:
            # 高性能更新模式（不闪烁，不卡顿）
            self.arm_line.set_data_3d(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2])
            self.base_point.set_data_3d([joint_positions[0, 0]], [joint_positions[0, 1]], [joint_positions[0, 2]])
            self.end_point.set_data_3d([joint_positions[-1, 0]], [joint_positions[-1, 1]], [joint_positions[-1, 2]])

    def animate(self, trajectory, speed=1.0):
        """动画显示机械臂运动"""
        # 强行绘制第一帧建立图像对象
        if len(trajectory) > 0:
            self.draw_arm(trajectory[0]['position'])

        def update(frame):
            angles = trajectory[frame]['position']
            # 直接调用 draw_arm，它内部会自动走“高性能更新模式”
            self.draw_arm(angles)
            return self.arm_line, self.base_point, self.end_point

        if len(trajectory) > 1:
            dt = trajectory[1]['time'] - trajectory[0]['time']
            interval = max(dt * 1000 / speed, 15) 
        else:
            interval = 33 

        anim = FuncAnimation(
            self.fig, update, frames=len(trajectory), 
            interval=interval, blit=False, repeat=True
        )

        plt.show()
        return anim

# 测试代码
if __name__ == "__main__":
    # 创建机械臂实例
    arm = RoboticArm(num_joints=6)
    
    # 创建可视化器实例
    viz = Visualizer(arm)

    print("开始生成动画轨迹...")
    
    # 制作一个运动轨迹 (Trajectory)
    trajectory = []
    fps = 30
    duration = 5.0 # 动画总时长 5 秒
    total_frames = int(fps * duration)
    
    for i in range(total_frames):
        t = i * (1.0 / fps) 
        
        # 模拟机械臂挥舞的动作
        angles = [
            math.sin(t) * 1.0,      # 关节1
            math.cos(t * 1.5) * 0.5,# 关节2
            math.sin(t * 2.0) * 0.5,# 关节3
            0,                      # 关节4
            math.cos(t) * 0.5,      # 关节5
            0                       # 关节6
        ]
        
        trajectory.append({
            'time': t,
            'position': angles
        })

    # 播放动画
    print("正在播放 ≥30 FPS 动画，请查看弹出的窗口...")
    anim = viz.animate(trajectory, speed=1.0)