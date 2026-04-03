import numpy as np
from numba import njit
import time

# 1. 定义 DH 参数转换矩阵 (带 @njit 装饰器)

@njit
def get_dh_matrix(a, alpha, d, theta):
    """计算单个关节的 DH 变换矩阵"""
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    ct = np.cos(theta)
    st = np.sin(theta)
    
    # 直接构造 4x4 矩阵
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0, sa,       ca,      d     ],
        [0.0, 0.0,      0.0,     1.0   ]
    ])


# 2. 定义正向运动学计算 (带 @njit 装饰器)

@njit
def fast_fk(joint_angles, dh_params):
    """
    joint_angles: 长度为 6 的数组
    dh_params: 形状为 (6, 4) 的数组 [a, alpha, d, offset]
    """
    # 初始为单位矩阵
    T_total = np.eye(4)
    
    for i in range(len(joint_angles)):
        a = dh_params[i, 0]
        alpha = dh_params[i, 1]
        d = dh_params[i, 2]
        theta = joint_angles[i] + dh_params[i, 3]
        
        # 矩阵累乘
        T_joint = get_dh_matrix(a, alpha, d, theta)
        T_total = T_total @ T_joint
        
    return T_total


# 3. 傻瓜式性能测试脚本

if __name__ == "__main__":
    # 定义你的机械臂参数 (示例: 常见的 6 轴机械臂)
    # [a, alpha, d, offset]
    my_dh = np.array([
        [0,      0,      0.3, 0],
        [0.4,    0,      0,   0],
        #[0.3,    0,      0,   0],
       # [0,     -1.5708, 0,   0],
       # [0,      1.5708, 0,   0],
      #  [0,      0,      0.1, 0]
    ])
    
    test_q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    # --- 第一次调用 (Numba 需要时间编译，这次不算数) ---
    _ = fast_fk(test_q, my_dh)

    # --- 正式测试 ---
    print("正在进行性能测试，请稍候...")
    iterations = 10000 # 测试 1 万次
    start_time = time.time()
    
    for _ in range(iterations):
        res = fast_fk(test_q, my_dh)
        
    end_time = time.time()
    
    # --- 计算结果 ---
    total_time = end_time - start_time
    ops_per_sec = iterations / total_time
    
    print("-" * 30)
    print(f"末端位姿矩阵:\n{res}")
    print("-" * 30)
    print(f"总耗时: {total_time:.4f} 秒")
    print(f"计算频率: {ops_per_sec:.2f} ops/sec")
    
    if ops_per_sec >= 10000:
        print("恭喜！你已通过测试！")
    else:
        print("请检查是否在循环是否有其他非计算逻辑。")