import numpy as np
import time

class RobotArmForward:
    def __init__(self):
        """
        初始化函数：设定好它的基本参数。
        """
        # 1. 自定义 DH 参数 
        self.dh_params = [
            {'alpha': 0,       'a': 0,      'd': 0.1625, 'theta': 0},
            {'alpha': -np.pi/2,'a': 0,      'd': 0,      'theta': 0},
            {'alpha': 0,       'a': 0.425,  'd': 0,      'theta': 0},
            {'alpha': 0,       'a': 0.3922, 'd': 0,      'theta': 0},
            {'alpha': -np.pi/2,'a': 0,      'd': 0.1333, 'theta': 0},
            {'alpha': np.pi/2, 'a': 0,      'd': 0.0997, 'theta': 0}
        ]
        
        # 2. 【升级点】设置关节限制 (安全带)
        # 意思是：每个关节只能转动 -180度 到 +180度
        limit = np.pi 
        self.joint_limits = [(-limit, limit)] * 6

    def check_limits(self, thetas):
        """
        【安全检查员】
        在运动前，检查输入的角度有没有把机器臂“扭断”。
        """
        # enumerate 能同时 索引(i) 和 值(q)
        for i, q in enumerate(thetas):
            low, high = self.joint_limits[i]
            # 如果角度小于最小值 或者 大于最大值
            if q < low or q > high:
                print(f"第 {i+1} 个关节角度 {q:.2f} 超出范围！")
                return False
        return True

    def _get_transform_matrix(self, theta, d, a, alpha):
        """
        【数学核心】根据 DH 公式计算单个关节的变换矩阵 T
        """
        # 计算三角函数
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        # 构建 4x4 矩阵
        return np.array([
            [ct, -st*ca,  st*sa,  a*ct],
            [st,  ct*ca, -ct*sa,  a*st],
            [0.0,   sa,     ca,     d],
            [0.0,   0.0,      0.0,      1.0]
        ])

    def solve_fk(self, thetas):
        """
        【普通模式】计算单个姿态的正向运动学
        输入: 6个角度的列表
        输出: 4x4 的变换矩阵
        """
        # 1. 先叫安全检查员查一遍
        if not self.check_limits(thetas):
            return None

        # 初始化一个 4x4 的单位矩阵 (对角线是1，其他是0)
        # 这代表“原点”，还没有开始移动
        T = np.eye(4)

        # 2. 循环 6 次，把每个关节的矩阵乘起来
        for i in range(6):
            # 获取当前关节的参数
            params = self.dh_params[i]
            # 把用户输入的角度加到 theta 上 (theta + theta_i)
            theta_val = thetas[i] + params['theta'] 
            
            # 计算当前这一节的矩阵
            T_i = self._get_transform_matrix(
                theta_val, params['d'], params['a'], params['alpha']
            )
            
            # 累乘矩阵：旧的 T 乘以 新的 T_i
            # np.dot 是矩阵乘法的意思
            T = np.dot(T, T_i)

        # 3. 【升级点】数值清理
        # 电脑算 sin(π) 会算出 0.000000000000012，我们要把它变成 0
        # decimals=6 表示保留小数点后6位
        T_clean = np.round(T, decimals=6)
        
        return T_clean

    def solve_fk_batch(self, thetas_batch):
        """
        【超级模式】批量计算 (向量化优化)
        输入: 形状为 (N, 6) 的矩阵，N 可以是 10000
        输出: 形状为 (N, 4, 4) 的矩阵堆
        """
        # 获取有多少组数据 (N)
        N = thetas_batch.shape[0]
        
        # 准备 N 个单位矩阵叠在一起
        # 形状变成 (N, 4, 4)
        T_global = np.tile(np.eye(4), (N, 1, 1))

        for i in range(6):
            # 取出所有数据的第 i 个关节角
            # [:, i] 的意思是：取所有行(:)，第 i 列
            theta = thetas_batch[:, i]
            
            params = self.dh_params[i]
            alpha = params['alpha']
            a = params['a']
            d = params['d']
            
            # 一次性算出几千个 cos 和 sin
            ct = np.cos(theta)
            st = np.sin(theta)
            ca = np.cos(alpha)
            sa = np.sin(alpha)
            
            # 构造这一级的批量矩阵
            # 造 N 个矩阵
            Ti = np.zeros((N, 4, 4))
            
            # 第一行
            Ti[:, 0, 0] = ct
            Ti[:, 0, 1] = -st * ca
            Ti[:, 0, 2] = st * sa
            Ti[:, 0, 3] = a * ct
            
            # 第二行
            Ti[:, 1, 0] = st
            Ti[:, 1, 1] = ct * ca
            Ti[:, 1, 2] = -ct * sa
            Ti[:, 1, 3] = a * st
            
            # 第三行
            Ti[:, 2, 1] = sa
            Ti[:, 2, 2] = ca
            Ti[:, 2, 3] = d
            
            # 第四行
            Ti[:, 3, 3] = 1.0
            
            # 批量矩阵乘法：matmul 专门做这个
            T_global = np.matmul(T_global, Ti)
            
        return T_global


# 下面是测试代码

if __name__ == "__main__":
    # 1. 创建机械臂实例
    arm = RobotArmForward()
    print("机械臂系统初始化完成！")

    # --- 测试 1: 普通模式 (算一次) ---
    print("\n--- 测试 1: 单次计算 ---")
    test_angles = [0, 0, 0, 0, 0, 0] # 6个0度
    result = arm.solve_fk(test_angles)
    print("末端姿态矩阵 T_0_6:")
    print(result)

    # --- 测试 2: 极限检查 (故意出错) ---
    print("\n--- 测试 2: 安全检查 ---")
    bad_angles = [0, 10, 0, 0, 0, 0] # 10弧度约等于570度，肯定越界
    arm.solve_fk(bad_angles)

    # --- 测试 3: 超级模式 (算一万次，比速度) ---
    print("\n--- 测试 3: 性能测试 (计算 10,000 次) ---")
    
    # 随机生成 10000 组角度，每组 6 个
    N = 10000
    batch_angles = np.random.rand(N, 6)
    
    start_time = time.time() # 掐表开始
    results = arm.solve_fk_batch(batch_angles)
    end_time = time.time()   # 掐表结束
    
    duration = end_time - start_time
    ops = N / duration
    
    print(f"计算完成！耗时: {duration:.4f} 秒")
    print(f"速度: {ops:.2f} ops/sec (目标是 >10,000)")
    
    if ops > 10000:
        print("恭喜！性能达标！")
    else:
        print("还需要优化！")