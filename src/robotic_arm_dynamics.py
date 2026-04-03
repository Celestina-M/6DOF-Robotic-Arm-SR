"""
机械臂动力学模块
"""
import numpy as np
from scipy.integrate import odeint
from robotic_arm_new import RoboticArm

class RoboticArmDynamics(RoboticArm):
    def __init__(self, num_joints=6):
        """初始化机械臂动力学模型
        参数:
            num_joints: 机械臂关节数量 (默认为 6)
        """
        super().__init__(num_joints)
        self.link_lengths = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # 连杆质量
        self.link_masses = [2.0, 3.0, 2.5, 1.5, 1.0, 0.5]
        # 连杆质心位置 (相对于关节坐标系)
        self.link_coms = [
            [0, -0.1, self.link_lengths[0]/2],  # Link 1
            [self.link_lengths[1]/2, 0, 0],  # Link 2
            [self.link_lengths[2]/2, 0, 0], # Link 3
            [0, 0, self.link_lengths[3]/2],  # Link 4
            [0, 0, 0], # Link 5
            [0, 0, self.link_lengths[4]/2]  # Link 6
        ]
        #连杆惯性张量（简化为对角矩阵 ）
        self.link_inertias = []
        for i, (m, l) in enumerate(zip(self.link_masses, self.link_lengths)):
            # 简化为细杆模型：I = (1/12) * m * L^2
            I = (1/12) * m * l**2
            self.link_inertias.append(np.diag([I, I, I]))
        # 重力加速度
        self.gravity = np.array([0, 0, -9.81])

        # 关节摩擦系数 
        self.friction_coeffs = np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])

    def mass_matrix(self, q):
        """计算质量矩阵 M(q)
        参数:
            q: 关节角度列表 (长度应为 num_joints)
        返回:
            M: 质量矩阵 (num_joints x num_joints)
        """
        M = np.zeros((self.num_joints, self.num_joints))

        # 对每个连杆 k 计算其对整体质量矩阵的贡献
        for k in range(self.num_joints):
            # 获取第 k 个连杆的质量和惯性张量
            m_k = self.link_masses[k]
            I_k = self.link_inertias[k]

            # 获取第 k 个连杆的雅可比矩阵 (3 x num_joints)
            # 注意：是连杆 k，计算它受所有前面关节的影响
            J_vk = self._velocity_jacobian(q, k)
            J_wk = self.angular_velocity_jacobian(q, k)

            # 矩阵乘法直接累加贡献： J_v.T * J_v 是 (6x3) @ (3x6) = (6x6) 的矩阵
            M += m_k * (J_vk.T @ J_vk) + (J_wk.T @ I_k @ J_wk)

        return M
    def coriolis_matrix(self, q, q_dot):
        """计算科氏力矩阵 C(q, q_dot)
        参数:
            q: 关节角度列表
            q_dot: 关节速度列表
        返回:
            C: 科氏力矩阵 (num_joints x num_joints)
        """
        C = np.zeros((self.num_joints, self.num_joints))
        epsilon = 1e-6

        # 1. 预计算：将质量矩阵 M 对每个关节角度 q_k 的偏导数先算出来
        # dM_dq 会保存 6 个 6x6 的矩阵，避免在三层循环里重复计算 
        dM_dq = []
        for k in range(self.num_joints):
            q_plus = q.copy()
            q_plus[k] += epsilon
            M_plus = self.mass_matrix(q_plus)

            q_minus = q.copy()
            q_minus[k] -= epsilon
            M_minus = self.mass_matrix(q_minus)

            # 数值求导得到偏导数矩阵 ∂M/∂q_k
            dM_dq_k = (M_plus - M_minus) / (2 * epsilon)
            dM_dq.append(dM_dq_k)

        # 2. 根据克里斯托菲尔符号组装科氏力矩阵
        for i in range(self.num_joints):
            for j in range(self.num_joints):
                for k in range(self.num_joints):
                    # 查表直接获取偏导数，公式：c_ijk = 0.5 * ( ∂M_ij/∂q_k + ∂M_ik/∂q_j - ∂M_jk/∂q_i )
                    c_ijk = 0.5 * (dM_dq[k][i, j] + dM_dq[j][i, k] - dM_dq[i][j, k])
                    C[i, j] += c_ijk * q_dot[k]

        return C
        
    def gravity_vector(self, q):
        """计算重力向量 G(q)
        参数:
            q: 关节角度列表
        返回:
            G: 重力向量 (num_joints,)
            实现重力项计算
        """
        G = np.zeros(self.num_joints)

        # 对每个关节计算重力贡献
        for i in range(self.num_joints):
            # 获得连杆质心的雅可比矩阵
            J_com = self._velocity_jacobian(q, i)
            # 重力矩阵 = J^T * F_gravity
            m_i = self.link_masses[i]
            F_gravity = m_i * self.gravity

            G += J_com.T @ F_gravity

        return G
    def forward_dynamics(self, q, q_dot,tau):
        """
        正向动力学：给定力矩，计算加速度

        参数:
            q: 关节角度列表
            q_dot: 关节速度列表
            tau: 关节力矩列表
            返回:
            q_ddot: 关节加速度列表
            实现正向动力学计算
            公式：q'' = M^(-1)(tau - C*q' - G)
        """
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, q_dot)
        G = self.gravity_vector(q)

        # 考虑摩擦力矩
        friction = self.friction_coeffs * q_dot

        # 计算加速度
        q_ddot = np.linalg.pinv(M) @ (tau - C @ q_dot - G - friction)

        return q_ddot
    
    def inverse_dynamics(self, q, q_dot, q_ddot):
        """
        逆向动力学：给定期望的加速度，计算所需的力矩

        参数:
            q: 关节角度列表
            q_dot: 关节速度列表
            q_ddot: 期望的关节加速度列表
            返回:
            tau: 关节力矩列表
            实现逆向动力学计算
            公式：tau = M*q'' + C*q' + G
        """
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, q_dot)
        G = self.gravity_vector(q)

        # 计算所需的力矩
        tau = M @ q_ddot + C @ q_dot + G 

        # 添加摩擦补偿
        tau += self.friction_coeffs * q_dot

        return tau
    
    def simulate(self, q0, q_dot0, tau_func, t_span, dt=0.01):
        """
        模拟机械臂运动

        参数:
            q0: 初始关节角度列表
            q_dot0: 初始关节速度列表
            tau_func: 力矩函数，接受 (q, q_dot, t) 作为输入，返回力矩列表
            t: 时间数组
            返回:
            q_traj: 关节角度轨迹 (len(t) x num_joints)
            实现机械臂运动模拟
        """
        # 计数器
        step_count = [0]

        def dynamics(state, t):
            """状态方程"""
            # 显示计算进度
            step_count[0] += 1
            if step_count[0] % 10 == 0:
                print(f"动力学仿真计算中... 虚拟时间推进到了 t = {t:.3f} 秒")
            q = state[:self.num_joints]
            q_dot = state[self.num_joints:]

            #计算力矩
            tau = tau_func(q, q_dot, t)

            #计算加速度
            q_ddot = self.forward_dynamics(q, q_dot, tau)

            #返回状态导数
            return np.concatenate([q_dot, q_ddot])
        
        #初始状态
        state0 = np.concatenate([q0, q_dot0])

        #时间点
        t = np.arange(t_span[0], t_span[1], dt)

        #数值积分
        states = odeint(dynamics, state0, t)

        q = states[:, :self.num_joints]
        q_dot = states[:, self.num_joints]

        return t, q, q_dot

    def _velocity_jacobian(self, q, link_idx):
        """计算连杆速度雅可比矩阵"""
        #简化实现：使用数值微分
        J = np.zeros((3, self.num_joints))
        epsilon = 1e-6

        for i in range(self.num_joints):
            q_plus = q.copy()
            q_plus[i] += epsilon
            pos_plus = self._get_link_position(q_plus, link_idx)

            q_minus = q.copy()
            q_minus[i] -= epsilon
            pos_minus = self._get_link_position(q_minus, link_idx)

            J[:, i] = (pos_plus - pos_minus) / (2 * epsilon)
        
        return J

    def _angular_velocity_jacobian(self, q, link_idx):
        """计算连杆角速度雅可比矩阵"""
        #简化实现
        J = np.zeros((3, self.num_joints))
        #实现角速度雅可比矩阵
        return J

    def _com_jacobian(self, q, link_idx):
        """计算质心雅可比矩阵"""
        #简化实现
        return self._velocity_jacobian(q, link_idx)
    
    def _get_link_position(self, q, link_idx):
        """获得指定连杆的位置 (x, y, z)"""
        # 调用robotics_arm_new中的方法，获取所有关节的 4x4 变换矩阵列表
        transforms = self.get_all_joint_transforms(q)
        
        # 提取目标连杆的变换矩阵
        T_i = transforms[link_idx]
        
        # 变换矩阵的第 4 列的前 3 个元素，就是该连杆在世界坐标系下的位置 [x, y, z]
        return T_i[0:3, 3]
    
#测试代码
if __name__ == "__main__":
    #创建动力学模型
    arm = RoboticArmDynamics(num_joints=6)

    #测试1:重力补偿
    print("测试1：重力补偿")
    q = np.array([0, np.pi/4, -np.pi/4, 0, 0, 0])
    G = arm.gravity_vector(q)
    print(f"重力力矩：{G}")

    #测试2：自由落体仿真
    print("\n测试2：自由落体仿真")
    q0 = np.array([0, np.pi/4, 0, 0, 0])
    q_dot0 = np.zero(6)

    def zero_torque(t, q, q_dot):
        return np.zeros(6)
    
    t, q_hist, q_dot_hist = arm.simulate(
        q0, q_dot0, zero_torque, [0, 2.0], dt=0.01
    )            
    print(f"仿真时间：{len(t)}个时间步")
    print(f"最终角度：{q_hist[-1]}")


class PIDController:
    def __init__(self, Kp, Ki, Kd, num_joints=6):
        """PID控制器初始化
        参数:
            Kp: 比例增益
            Ki: 积分增益
            Kd: 微分增益
        """
        self.Kp = np.array(Kp)
        self.Ki = np.array(Ki)
        self.Kd = np.array(Kd)

        # 积分项累计
        self.integral = np.zeros(num_joints)

        # 上一次误差，用于计算微分项
        self.prev_error = np.zeros(num_joints)

        #时间步长
        self.dt = 0.01

    def compute_control(self, q_desired, q_actual, q_dot_actual):
        """计算控制力矩
        参数:
            q_desired: 期望关节角度列表
            q_current: 当前关节角度列表
            q_dot_current: 当前关节速度列表
            dt: 时间步长
            返回:
            tau: 控制力矩列表
        """
        #位置误差
        error = q_desired - q_actual

        #积分项
        self.integral += error * self.dt

        #微分项（使用速度反馈）
        derivative = -q_dot_actual

        #PID控制律
        tau = (self.Kp * error + 
               self.Ki * self.integral + 
               self.Kd * derivative)
        
        #更新上一次误差
        self.prev_error = error

        return tau
    def reset(self):
        """重置积分项和误差"""
        self.integral = np.zeros_like(self.integral)
        self.prev_error = np.zeros_like(self.prev_error)
    
#测试代码
def test_pid_controller():
    """测试PID控制器"""
    arm = RoboticArmDynamics(num_joints=6)

    #PID参数
    Kp = [50, 50, 50, 20, 20, 20]
    Ki = [0, 0, 0, 0, 0, 0]
    Kd = [10, 10, 10, 5, 5, 5]

    controller = PIDController(Kp, Ki, Kd)

    #目标位置
    q_target = np.array([0.5, 0.5, -0.5, 0, 0.5, 0])

    #初始状态
    q0 = np.array([0.1, 0.1, -0.1, 0.1, 0.1, 0.1])
    q_dot0 = np.zeros(6)

    #控制函数
    def control_torque(q, q_dot, t):
        return controller.compute_control(q_target, q, q_dot)
    
    #仿真
    t, q_hist, q_dot_hist = arm.simulate(
        q0, q_dot0, control_torque, [0, 2.0], dt=0.01
    )

    #绘制结果
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    for i in range(6):
        plt.subplot(3, 2, i+1)
        plt.plot(t, q_hist[:, i], label=f'Joint {i+1}')
        plt.axhline(q_target[i], color='r', linestyle='--', label='Target')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (rad)')
        plt.title(f'Joint {i+1} Angle')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    #计算稳态误差
    final_error = np.abs(q_hist[-1] - q_target)
    print(f"最终误差：{final_error}")
    print(f"最大误差：{np.max(final_error):.4f} rad")

if __name__ == "__main__":
    test_pid_controller()


class ComputedTorqueController:
    def __init__(self, robotic_dynamics, Kp, Kd):
        """计算力矩控制器初始化
        参数:
            robotic_dynamics: 机械臂动力学模型实例
            Kp: 比例增益列表
            Kd: 微分增益列表
        """
        self.arm = robotic_dynamics
        self.Kp = np.array(Kp)
        self.Kd = np.array(Kd)

    def compute_control(self, q_desired, q_dot_desired, q_ddot_desired, q_actual, q_dot_actual):
        """计算控制力矩
        参数:
            q_desired: 期望关节角度列表
            q_dot_desired: 期望关节速度列表
            q_ddot_desired: 期望关节加速度列表
            q_actual: 当前关节角度列表
            q_dot_actual: 当前关节速度列表
            返回:
            tau: 控制力矩列表

            公式：tau = M(q) * (q_ddot_desired + Kp*(q_desired - q_actual) + Kd*(q_dot_desired - q_dot_actual)) + C(q, q_dot)*q_dot + G(q)
        """
        #位置误差和速度误差
        error = q_desired - q_actual
        error_dot = q_dot_desired - q_dot_actual

        #期望加速度（PD控制律）
        q_ddot_cmd = q_ddot_desired + self.Kp * error + self.Kd * error_dot

        #计算所需的力矩（逆向动力学）
        tau = self.arm.inverse_dynamics(q_actual, q_dot_actual, q_ddot_cmd)

        return tau

class GravityCompensationController:
    def __init__(self, robotic_dynamics, kp, kd):
        """重力补偿控制器初始化
        参数:
            robotic_dynamics: 机械臂动力学模型实例
            kp: 比例增益
            kd: 微分增益
        """
        self.arm = robotic_dynamics
        self.kp = np.array(kp)
        self.kd = np.array(kd)

    def compute(self, q_desired, q_actual, q_dot_actual):
        """计算重力补偿力矩
        参数:
            q_desired: 期望关节角度列表
            q_actual: 当前关节角度列表
            q_dot_actual: 当前关节速度列表
            返回:
            tau: 重力补偿力矩列表
        """
        #位置误差和速度误差
        error = q_desired - q_actual
        error_dot = -q_dot_actual


        #重力补偿
        G = self.arm.gravity_vector(q_actual)

        #PD控制 + 重力补偿
        tau = G + self.kp * error + self.kd * error_dot
        return tau
    
#测试代码
def compare_controllers():
    """比较计算力矩控制器和重力补偿控制器"""
    arm = RoboticArmDynamics(num_joints=6)

    #目标轨迹（正弦波）
    def trajectory(t):
        q = 0.5 * np.sin(2 * np.pi * 0.2 * t) * np.ones(6)
        q_dot = 0.5 * 2 * np.pi  * 0.2 * np.cos(2 * np.pi* 0.2 * t) * np.ones(6)
        q_ddot = -0.5 * (2 * np.pi * 0.2)**2 * np.sin(2 * np.pi * 0.2 * t) * np.ones(6)
        return q, q_dot, q_ddot
    
    #控制器参数
    Kp = [200, 200, 200, 100, 100, 100]
    Kd = [40, 40, 40, 20, 20, 20]

    #测试计算力矩控制
    print("测试计算力矩控制器...")
    ct_controller = ComputedTorqueController(arm, Kp, Kd)

    q0 = np.zeros(6)
    q_dot0 = np.zeros(6)

    def ct_control(q, q_dot, t):
        q_d, q_dot_d, q_ddot_d = trajectory(t)
        return ct_controller.compute_control(q_d, q_dot_d, q_ddot_d, q, q_dot)
    
    q0, _, _ = trajectory(0)
    t, q_ct, q_dot_ct = arm.simulate(q0, q_dot0, ct_control, [0, 1.0], dt=0.02)

    #计算跟踪误差
    errors = []
    for i, ti in enumerate(t):
        q_d, _, _ = trajectory(ti)
        errors.append(np.linalg.norm(q_ct[i] - q_d))

    print(f"计算力矩控制 - 平均误差：{np.mean(errors):.4f} rad")
    print(f"计算力矩控制 - 最大误差：{np.max(errors):.4f} rad")

    #绘制结果
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, q_ct[:, 0], label = 'Actual')
    q_desired = [trajectory(ti)[0][0] for ti in t]
    plt.plot(t, q_desired, '--', label='Desired')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Angle (rad)')
    plt.legend()
    plt.grid(True)
    plt.title('Computed Torque Control')

    plt.subplot(2, 1, 2)
    plt.plot(t, errors)
    plt.xlabel('Time (s)')
    plt.ylabel('Tracking Error (rad)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_controllers()