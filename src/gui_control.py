"""
GUI 控制界面
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
from robotic_arm_new import RoboticArm
from visualizer import Visualizer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class RobotGUI:
    def __init__(self, root):
        """
        GUI 初始化：创建机械臂实例、可视化器实例，并设置界面布局
        """
        self.root = root
        self.root.title("Robotic Arm Control Panel")

        # 创建机械臂
        self.robotic_arm = RoboticArm()

        # 创建可视化器
        self.viz = Visualizer(self.robotic_arm)

        # 设置界面布局
        self.create_widgets()

        # 初始显示
        self.update_visualization()
    
    def create_widgets(self):
        """
        创建 GUI 组件
        """
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="Joint Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 关节角度输入框
        self.joint_sliders = []
        self.joint_labels = []
        for i in range(6):
            # 标签
            label = ttk.Label(control_frame, text=f"Joint {i+1} ")
            label.grid(row=i, column=0, sticky=tk.W, pady=5)

            # 滑块
            slider = ttk.Scale(control_frame, 
                               from_=-180, 
                               to=180, 
                               orient=tk.HORIZONTAL, 
                               length=200)
            slider.set(0) # 设置初始值为 0 
            slider.grid(row=i, column=1, pady=5)

            # 数值显示
            value_label = ttk.Label(control_frame, text="0.00°")
            value_label.grid(row=i, column=2, sticky=tk.W, padx=5)

            self.joint_sliders.append(slider)
            self.joint_labels.append(value_label)

        for i, slider in enumerate(self.joint_sliders):
            slider.config(command=lambda val, idx=i: self.on_slider_change(idx, val))



        # 按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=6, column=0, columnspan=3, pady=10)

        ttk.Button(button_frame, text="Reset", command=self.reset_joints).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Random", command=self.random_pose).pack(side=tk.LEFT, padx=5)

        # IK 求解器
        ik_frame = ttk.LabelFrame(main_frame, text="IK Solver", padding="10")
        ik_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=10)

        # 目标位置输入
        ttk.Label(ik_frame, text="Target X:").grid(row=0, column=2)
        self.target_x = ttk.Entry(ik_frame, width=10)
        self.target_x.grid(row=0, column=3, padx=5)
        self.target_x.insert(0, "0.3")

        ttk.Label(ik_frame, text="Y:").grid(row=0, column=4)
        self.target_y = ttk.Entry(ik_frame, width=10)
        self.target_y.grid(row=0, column=5, padx=5)
        self.target_y.insert(0, "0.4")

        ttk.Label(ik_frame, text="Z:").grid(row=0, column=6)
        self.target_z = ttk.Entry(ik_frame, width=10)
        self.target_z.grid(row=0, column=7, padx=5)
        self.target_z.insert(0, "0.4")

        ttk.Button(ik_frame, text="Solve IK", command=self.solve_ik).grid(row=1, column=0, columnspan=8, pady=5)  

        # 状态显示
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=10)
        self.status_label = ttk.Label(status_frame, text="End Effector Position: (0.00, 0.00, 0.00)")
        self.status_label.pack()

        # 右侧：3D 可视化
        viz_frame = ttk.LabelFrame(main_frame)
        viz_frame.grid(row=0, column=1, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 嵌入 Matplotlib 图表 
        self.viz.setup_plot()
        canvas = FigureCanvasTkAgg(self.viz.fig, master=viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas = canvas

    def on_slider_change(self, joint_idx, value):
        """
        滑块值改变回调
        """
        # 转换为弧度
        angle_deg = float(value)
        # angle_rad = np.deg2rad(angle_deg) 

        # 更新标签
        self.joint_labels[joint_idx].config(text=f"{angle_deg:.2f}°")

        # 更新机械臂 
        angles = [np.deg2rad(float(s.get())) for s in self.joint_sliders]
        self.robotic_arm.set_joint_angles(angles)

        # 更新可视化
        self.update_visualization()

    def update_visualization(self):
        """更新机械臂的可视化显示"""
        self.viz.draw_arm()
        self.canvas.draw()

        # 更新状态显示 
        pos = self.robotic_arm.get_end_effector_position()
        self.status_label.config(text=f"End Effector Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    def reset_joints(self):
        """重置关节角度为零"""
        for slider in self.joint_sliders:
            slider.set(0)
        self.update_visualization()

    def random_pose(self):
        """随机设置一个机械臂姿态"""
        for slider in self.joint_sliders:
            angle = np.random.uniform(-90, 90)
            slider.set(angle)
        self.update_visualization()

    def solve_ik(self):
        """根据输入的目标位置求解 IK"""
        try:
            # 获取目标位置
            target = [
                float(self.target_x.get()),
                float(self.target_y.get()),
                float(self.target_z.get())
            ]

            # 求解 IK
            success, angles = self.robotic_arm.inverse_kinematics(target)

            if success:
                # 更新滑块位置 
                for i, angle in enumerate(angles):
                    self.joint_sliders[i].set(np.rad2deg(angle))

            
                self.status_label.config(text=f"IK success! target: {target}!")
            else:
                self.status_label.config(text="IK Solution Failed!")
        
            self.update_visualization()

        except ValueError:
            self.status_label.config(text="Invalid input!")

# 将主函数移出 RobotGUI 类 
def main():
    """主函数，启动 GUI"""
    root = tk.Tk()
    app = RobotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()