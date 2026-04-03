# 文件：src/robotic_arm_optimization.py
"""
优化控制模块
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from robotic_arm_dynamics import RoboticArmDynamics

class MPCController:
    def __init__(self, arm_dynamics, horizon=10, dt=0.1):
        """
        模型预测控制器
        
        参数:
            arm_dynamics: RoboticArmDynamics实例
            horizon: 预测时域
            dt: 时间步长
        """
        self.arm = arm_dynamics
        self.horizon = horizon
        self.dt = dt
        self.num_joints = arm_dynamics.num_joints
        
        # 权重矩阵
        self.Q = np.diag([100, 100, 100, 50, 50, 50]) # 状态权重
        self.R = np.diag([1, 1, 1, 1, 1, 1]) # 控制权重
        
        # 约束
        self.u_min = -np.array([50, 50, 50, 25, 25, 25])
        self.u_max = np.array([50, 50, 50, 25, 25, 25])
        
    def compute(self, q_current, q_dot_current, q_ref_trajectory):
        """
        计算MPC控制输入
        
        参数:
            q_current: 当前关节角度
            q_dot_current: 当前关节速度
            q_ref_trajectory: 参考轨迹 (horizon x num_joints)
            
        返回:
            u_optimal: 最优控制输入(第一步)
        """
        # 初始猜测 (零输入)
        u_init = np.zeros(self.horizon * self.num_joints)
        
        # 定义优化问题
        def cost_function(u_sequence):
            """代价函数"""
            u_sequence = u_sequence.reshape(self.horizon, self.num_joints)
            
            cost = 0
            q = q_current.copy()
            q_dot = q_dot_current.copy()
            
            for k in range(self.horizon):
                # 预测下一状态
                q_ddot = self.arm.forward_dynamics(q, q_dot, u_sequence[k])
                q_dot = q_dot + q_ddot * self.dt
                q = q + q_dot * self.dt
                
                # 状态误差
                if k < len(q_ref_trajectory):
                    q_error = q - q_ref_trajectory[k]
                else:
                    q_error = q - q_ref_trajectory[-1]
                    
                # 累积代价
                cost += q_error.T @ self.Q @ q_error
                cost += u_sequence[k].T @ self.R @ u_sequence[k]
                
            return cost
            
        # 约束
        bounds = []
        for _ in range(self.horizon):
            for i in range(self.num_joints):
                bounds.append((self.u_min[i], self.u_max[i]))
                
        # 求解优化问题
        result = minimize(
            cost_function,
            u_init,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'disp': False}
        )
        
        # 提取第一个控制输入
        u_optimal = result.x[:self.num_joints]
        
        return u_optimal

def test_mpc():
    """测试MPC轨迹跟踪"""
    arm = RoboticArmDynamics(num_joints=6)
    mpc = MPCController(arm, horizon=10, dt=0.1)
    
    # 生成参考轨迹
    t_total = 5.0
    dt = 0.1
    t = np.arange(0, t_total, dt)
    
    q_ref = []
    for ti in t:
        q = 0.5 * np.sin(2 * np.pi * 0.2 * ti) * np.ones(6)
        q_ref.append(q)
    q_ref = np.array(q_ref)
    
    # 仿真
    q = np.zeros(6)
    q_dot = np.zeros(6)
    
    q_history = [q.copy()]
    u_history = []
    
    for i in range(len(t) - mpc.horizon):
        # 获取参考轨迹片段
        q_ref_horizon = q_ref[i:i+mpc.horizon]
        
        # 计算控制输入
        u = mpc.compute(q, q_dot, q_ref_horizon)
        u_history.append(u)
        
        # 应用控制并更新状态
        q_ddot = arm.forward_dynamics(q, q_dot, u)
        q_dot = q_dot + q_ddot * dt
        q = q + q_dot * dt
        
        q_history.append(q.copy())
        
    q_history = np.array(q_history)
    
    # 绘制结果
    plt.figure(figsize=(12, 8))
    
    for i in range(6):
        plt.subplot(3, 2, i+1)
        plt.plot(t[:len(q_history)], q_history[:, i], label='MPC')
        plt.plot(t[:len(q_history)], q_ref[:len(q_history), i], '--', label='Reference')
        plt.xlabel('Time (s)')
        plt.ylabel(f'Joint {i+1} (rad)')
        plt.legend()
        plt.grid(True)
        
    plt.tight_layout()
    plt.show()
    
    # 计算性能指标
    tracking_error = np.mean(np.linalg.norm(
        q_history - q_ref[:len(q_history)], axis=1
    ))
    print(f"平均跟踪误差: {tracking_error:.4f} rad")


class ILCController:
    def __init__(self, arm_dynamics, learning_rate=0.5):
        """
        迭代学习控制器
        
        参数:
            arm_dynamics: RoboticArmDynamics实例
            learning_rate: 学习率
        """
        self.arm = arm_dynamics
        self.learning_rate = learning_rate
        self.num_joints = arm_dynamics.num_joints
        
        # 存储上一次迭代的控制输入
        self.u_prev = None
        
        # 存储上一次迭代的跟踪误差
        self.e_prev = None
        
    def update(self, q_trajectory, q_dot_trajectory, q_ref_trajectory):
        """
        更新控制输入 (一次完整迭代后)
        
        参数:
            q_trajectory: 实际轨迹
            q_dot_trajectory: 实际速度轨迹
            q_ref_trajectory: 参考轨迹
            
        返回:
            u_new: 更新后的控制输入序列
        """
        T = len(q_ref_trajectory)
        
        # 计算跟踪误差
        e = q_ref_trajectory - q_trajectory
        
        # 初始化控制输入
        if self.u_prev is None:
            # 第一次迭代：使用逆动力学初始化
            self.u_prev = np.zeros((T, self.num_joints))
            for t in range(T):
                if t < T - 1:
                    q_ddot = (q_dot_trajectory[t+1] - q_dot_trajectory[t]) / 0.01
                else:
                    q_ddot = np.zeros(self.num_joints)
                    
                self.u_prev[t] = self.arm.inverse_dynamics(
                    q_trajectory[t], q_dot_trajectory[t], q_ddot
                )
                
        # ILC学习律
        # 公式: u_{k+1}(t) = u_k(t) + L * e_k(t)
        u_new = self.u_prev + self.learning_rate * e
        
        # 更新
        self.u_prev = u_new
        self.e_prev = e
        
        return u_new
        
    def get_convergence_metric(self):
        """获取收敛指标"""
        if self.e_prev is not None:
            return np.mean(np.linalg.norm(self.e_prev, axis=1))
        return float('inf')


def test_ilc():
    """测试ILC学习"""
    arm = RoboticArmDynamics(num_joints=6)
    ilc = ILCController(arm, learning_rate=0.3)
    
    # 参考轨迹
    t_total = 5.0
    dt = 0.01
    t = np.arange(0, t_total, dt)
    
    q_ref = []
    q_dot_ref = []
    for ti in t:
        q = 0.5 * np.sin(2 * np.pi * 0.2 * ti) * np.ones(6)
        q_dot = 0.5 * 2 * np.pi * 0.2 * np.cos(2 * np.pi * 0.2 * ti) * np.ones(6)
        q_ref.append(q)
        q_dot_ref.append(q_dot)
        
    q_ref = np.array(q_ref)
    q_dot_ref = np.array(q_dot_ref)
    
    # 迭代学习
    num_iterations = 20
    errors = []
    
    print("开始ILC学习...")
    
    for iteration in range(num_iterations):
        # 初始状态
        q = np.zeros(6)
        q_dot = np.zeros(6)
        
        q_trajectory = [q.copy()]
        q_dot_trajectory = [q_dot.copy()]
        
        # 获取当前控制输入
        if ilc.u_prev is not None:
            u_sequence = ilc.u_prev
        else:
            u_sequence = np.zeros((len(t), 6))
            
        # 执行轨迹
        for i in range(len(t) - 1):
            u = u_sequence[i]
            
            # 动力学仿真
            q_ddot = arm.forward_dynamics(q, q_dot, u)
            q_dot = q_dot + q_ddot * dt
            q = q + q_dot * dt
            
            q_trajectory.append(q.copy())
            q_dot_trajectory.append(q_dot.copy())
            
        q_trajectory = np.array(q_trajectory)
        q_dot_trajectory = np.array(q_dot_trajectory)
        
        # 更新控制输入
        u_new = ilc.update(q_trajectory, q_dot_trajectory, q_ref)
        
        # 记录误差
        error = ilc.get_convergence_metric()
        errors.append(error)
        
        print(f"迭代 {iteration+1}: 误差 = {error:.4f} rad")
        
    # 绘制学习曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(errors)
    plt.xlabel('Iteration')
    plt.ylabel('Tracking Error (rad)')
    plt.title('ILC Learning Curve')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for i in range(6):
        plt.plot(t, q_trajectory[:, i], label=f'Joint {i+1}')
    plt.plot(t, q_ref[:, 0], 'k--', linewidth=2, label='Reference')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title(f'Final Trajectory (Iteration {num_iterations})')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 可以在这里注释掉其中一个来分别测试
    # test_mpc() 
    test_ilc()