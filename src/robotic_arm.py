import numpy as np
import time
from numba import njit

# --- 核心计算函数 (放在类外并使用 njit 加速) ---
@njit
def fast_dh_transform(theta, d, a, alpha):
    """计算单个 DH 变换矩阵 (极致性能版)"""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,      ca,     d],
        [0,   0,       0,      1]
    ], dtype=np.float64)

@njit
def fast_fk_computation(joint_angles, dh_params):
    """齐次变换矩阵连乘逻辑 (修正了 Numba 类型断言错误)"""
    T = np.eye(4)
    # 强制确保 T 是 float64 类型
    T = T.astype(np.float64)
    
    for i in range(len(joint_angles)):
        # dh_params 结构: [theta_offset, d, a, alpha]
        theta_offset, d, a, alpha = dh_params[i]
        theta = theta_offset + joint_angles[i]
        
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        
        # 关键修正：所有的 0 和 1 都要写成 0.0 和 1.0
        # 这样 Numba 才能正确识别这是一个全浮点数的二维数组
        T_i = np.array([
            [ct,  -st*ca,  st*sa,  a*ct],
            [st,   ct*ca, -ct*sa,  a*st],
            [0.0,  sa,     ca,     d   ],
            [0.0,  0.0,    0.0,    1.0 ]
        ], dtype=np.float64) # 显式指定 dtype 也是一种保险做法
        
        T = T @ T_i
        
    return T

class RoboticArm:
    def __init__(self, num_joints=6):
        """初始化机械臂：设定 UR5 标准参数"""
        self.num_joints = num_joints
        self.joint_angles = np.zeros(num_joints)
        self.joint_limits = [(-np.pi, np.pi)] * num_joints
        
        # 修正：同步为测试脚本期待的 DH 参数 [theta_offset, d, a, alpha]
        # 这里的数值对应 UR5 的物理结构
        self.dh_params = np.array([
            [0, 0.1625, 0,      0],        # Joint 1
            [0, 0,      0,     -np.pi/2],  # Joint 2
            [0, 0,      0.425,  0],        # Joint 3
            [0, 0,      0.3922, 0],        # Joint 4
            [0, 0.1333, 0,     -np.pi/2],  # Joint 5
            [0, 0.0997, 0,      np.pi/2]   # Joint 6
        ], dtype=np.float64)

    def forward_kinematics(self, joint_angles=None):
        """调用加速后的计算函数"""
        if joint_angles is None:
            joint_angles = self.joint_angles
        
        # 转换为 numpy 数组确保兼容性
        joint_angles = np.array(joint_angles, dtype=np.float64)
        T = fast_fk_computation(joint_angles, self.dh_params)
        
        # 为了兼容你之前的绘图代码，这里保持返回 T
        return T

    def set_joint_angles(self, angles):
        """设置角度"""
        self.joint_angles = np.clip(np.array(angles), -np.pi, np.pi)

    def get_end_effector_position(self):
        """快速获取末端 [x, y, z]"""
        T = self.forward_kinematics()
        return T[:3, 3]

# --- 内部测试代码 ---
if __name__ == "__main__":
    arm = RoboticArm(num_joints=6)
    
    # 精度验证
    arm.set_joint_angles([0, 0, 0, 0, 0, 0])
    pos = arm.get_end_effector_position()
    print(f"验证：全零位姿末端坐标: {pos.round(4)}")
    
    # 性能预热 (Numba 第一次运行需要编译)
    arm.forward_kinematics([0]*6)
    
    # 正式测速
    start = time.time()
    for _ in range(10000):
        arm.forward_kinematics(np.random.uniform(-np.pi, np.pi, 6))
    elapsed = time.time() - start
    print(f"优化后性能: {10000 / elapsed:.0f} ops/sec")





    # ==========================================
    # Inverse Kinematics 
    # 任务 1: 实现数值微分计算雅可比矩阵
    # ==========================================
    def jacobian(self, joint_angles=None, epsilon=1e-6):
        """
        计算雅可比矩阵 (数值微分法 - 中心差分)
        joint_angles: 可选的关节角度输入，如果为 None 则使用当前状态
        epsilon: 微小扰动量，用于数值微分
        返回: 雅可比矩阵 J (3 x num_joints)
        TODO: 实现微分计算的雅可比矩阵
        """
        if joint_angles is None:
            joint_angles = self.joint_angles.copy()
        else:
            joint_angles = np.array(joint_angles).copy()

        # 初始化雅可比矩阵 (假设末端位置是3D [x,y,z]，所以是 3 x num_joints)
        J = np.zeros((3, self.num_joints))

        # 对每个关节进行数值微分
        for i in range(self.num_joints):
            # 保存原始角度
            original_angle = joint_angles[i]

            # 正向扰动 (+epsilon)
            joint_angles[i] = original_angle + epsilon
            # 假设 forward_kinematics 返回完整变换矩阵，提取位置部分 [:, 3] 的前三个元素
            pos_plus, _ = self.forward_kinematics(joint_angles) 
            pos_plus = pos_plus[:3, 3] 

            # 负向扰动 (-epsilon)
            joint_angles[i] = original_angle - epsilon
            pos_minus, _ = self.forward_kinematics(joint_angles)
            pos_minus = pos_minus[:3, 3]

            # 计算偏导数 (中心差分公式)
            J[:, i] = (pos_plus - pos_minus) / (2 * epsilon)

            # 恢复原始角度
            joint_angles[i] = original_angle

        return J

    # ==========================================
    # 任务 2: 实现逆运动学求解器入口
    # ==========================================
    def inverse_kinematics(self, target_position, initial_guess=None, 
                           method='jacobian', max_iterations=100, tolerance=1e-3):
        """
        逆运动学求解统一接口
        参数：
- target_position: 目标末端位置 [x, y, z]
- initial_guess: 可选的初始关节角度猜测 (长度为 num_joints)
- method: 'jacobian' 或 'optimization'，选择求解方法
- max_iterations: 最大迭代次数 (仅对雅可比方法有效)
- tolerance: 收敛容忍度 (仅对雅可比方法有效)
返回：      (success: bool 是否成功求解, joint_angles: np.array 求解的关节角度)
TODO: 实现逆运动学求解
        """
        target_position = np.array(target_position)

        # 初始猜测
        if initial_guess is None:
            # 使用当前角度或随机初始化
            theta = self.joint_angles.copy()
        else:
            theta = np.array(initial_guess).copy()

        if method == 'jacobian':
            return self._ik_jacobian(target_position, theta, max_iterations, tolerance)
        elif method == 'optimization':
            return self._ik_optimization(target_position, theta)
        else:
            raise ValueError(f"未知方法: {method}")

    # ==========================================
    # 任务 3: 基于雅可比矩阵的 IK 求解
    # ==========================================
    def _ik_jacobian(self, target_pos, theta, max_iter, tol):
        """
        使用伪逆和阻尼最小二乘法(DLS)的雅可比迭代求解
        """
        learning_rate = 0.5

        for iteration in range(max_iter):
            # 1. 计算当前位置 (需要你已经实现了 get_end_effector_position)
            current_pos = self.get_end_effector_position()
            
            # 2. 计算误差
            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)

            # 3. 检查是否满足收敛条件
            if error_norm < tol:
                self.set_joint_angles(theta)
                return True, theta

            # 4. 计算当前位姿的雅可比矩阵
            J = self.jacobian(theta)

            # 5. 伪逆求解与更新
            try:
                J_pinv = np.linalg.pinv(J)
                delta_theta = learning_rate * (J_pinv @ error)
            except np.linalg.LinAlgError:
                # 如果遇到奇异矩阵 (Singularity)，尝试使用阻尼最小二乘法 (DLS)
                damping = 0.01
                J_damped = J.T @ J + damping * np.eye(self.num_joints)
                delta_theta = learning_rate * (np.linalg.inv(J_damped) @ J.T @ error)

            # 6. 更新关节角度
            theta += delta_theta

            # 7. 应用关节物理限制 (防止超出机械臂活动范围)
            for i, (min_angle, max_angle) in enumerate(self.joint_limits):
                theta[i] = np.clip(theta[i], min_angle, max_angle)
                
            # 更新内部状态用于下一次迭代计算当前位置
            self.set_joint_angles(theta)

        # 超过最大迭代次数仍未收敛
        return False, theta

    # ==========================================
    # 任务 4: 基于优化的 IK 求解
    # ==========================================
    def _ik_optimization(self, target_pos, theta):
        """
        使用 scipy.optimize 的数值优化求解 IK
        """
        from scipy.optimize import minimize
        def objective(angles):
            """目标函数：位置误差的平方和"""
            self.set_joint_angles(angles)
            current_pos = self.get_end_effector_position()
            error = target_pos - current_pos
            return np.sum(error**2)

        # 关节限制作为优化的边界
        bounds = self.joint_limits

        # 执行优化
        result = minimize(
            objective,
            theta,
            method='SLSQP', # 序列最小二乘规划算法，适合带边界的非线性优化
            bounds=bounds,
            options={'maxiter': 100}
        )

        # 检查结果
        if result.success and result.fun < 1e-4:
            self.set_joint_angles(result.x)
            return True, result.x
        else:
            return False, result.x    