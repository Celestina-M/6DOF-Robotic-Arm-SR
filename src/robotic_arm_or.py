import numpy as np
import time

class RoboticArm:
    def __init__(self, num_joints=6):#机械臂关节数
        """初始化机械臂：设定身体参数"""
        self.num_joints = num_joints
        self.joint_angles = np.zeros(num_joints) # 初始角度全为 0
        
        # 连杆长度（单位：米）
        self.link_lengths = [0.1, 0.3, 0.25, 0.15, 0.1, 0.05]
        
        # 关节旋转限制（弧度）
        self.joint_limits = [(-np.pi, np.pi)] * num_joints
        
        # 初始化 DH 参数表
        self.dh_params = self._initialize_dh_parameters()

    def _initialize_dh_parameters(self):
        """定义 DH 参数表：[theta, d, a, alpha]"""
        # 代入要验证的机械臂
        L = self.link_lengths
        dh = np.array([
            [0, 0, L[0],    0],  # Joint 1
            [0, 0,    L[1], 0],        # Joint 2
            [0, 0,    L[2], 0],        # Joint 3
            [0, L[3], 0,    np.pi/2],  # Joint 4
            [0, 0,    0,   -np.pi/2],  # Joint 5
        [0, L[4]+L[5], 0, 0]       # Joint 6
        ])
        return dh

    def dh_transform(self, theta, d, a, alpha):
        """计算单个 DH 变换矩阵"""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,    d],
            [0,   0,      0,     1]
        ])

    def forward_kinematics(self, joint_angles=None):
        """正向运动学：从关节角度算出末端坐标"""
        if joint_angles is None:
            joint_angles = self.joint_angles
            
        T = np.eye(4) # 从基座开始（单位阵）
        joint_positions = [T[:3, 3].copy()] # 记录每个关节的位置
        
        for i in range(self.num_joints):
            theta_dh, d, a, alpha = self.dh_params[i]
            theta = theta_dh + joint_angles[i] # DH 基础角 + 实时旋转角
            
            T_i = self.dh_transform(theta, d, a, alpha)
            T = T @ T_i # 矩阵累乘
            joint_positions.append(T[:3, 3].copy())
            
        return T, joint_positions

    def set_joint_angles(self, angles):
        """设置角度，并检查是否超限"""
        angles = np.array(angles)
        for i, (angle, (low, high)) in enumerate(zip(angles, self.joint_limits)):
            if angle < low or angle > high:
                print(f"警告：关节 {i+1} 角度超出限制！已截断。")
        self.joint_angles = np.clip(angles, -np.pi, np.pi)

    def get_end_effector_position(self):
        """快速获取末端 [x, y, z]"""
        T, _ = self.forward_kinematics()
        return T[:3, 3]

# --- 测试代码 ---
if __name__ == "__main__":
    arm = RoboticArm(num_joints=6)
    
    # 测试1：所有关节角度为 0
    print("测试1：零位姿")
    arm.set_joint_angles([0, 0, 0, 0, 0, 0])
    pos = arm.get_end_effector_position()
    print(f"末端坐标 [x, y, z]: {pos}")
    
    # 测试2：性能测试
    print("\n测试2：性能测试 (运行 10000 次 FK)")
    start = time.time()
    for _ in range(10000):
        random_angles = np.random.uniform(-np.pi, np.pi, 6)
        arm.forward_kinematics(random_angles)
    elapsed = time.time() - start
    print(f"每秒计算次数: {10000 / elapsed:.0f} ops/sec")