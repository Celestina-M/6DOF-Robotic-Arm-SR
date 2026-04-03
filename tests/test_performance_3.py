import numpy as np
from numba import njit
import time
import sys
import os


# 核心加速计算函数 (使用 @njit 装饰器)


@njit
def compute_dh_matrix_fast(a, alpha, d, theta):
    """计算单个关节的变换矩阵"""
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    # 标准 DH 变换矩阵
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0, sa,       ca,      d     ],
        [0.0, 0.0,      0.0,     1.0   ]
    ], dtype=np.float64)

@njit
def fast_forward_kinematics(joint_angles, dh_params):
    """执行 6 轴连乘计算"""
    T_total = np.eye(4)
    for i in range(len(joint_angles)):
        a = dh_params[i, 0]
        alpha = dh_params[i, 1]
        d = dh_params[i, 2]
        theta = joint_angles[i] + dh_params[i, 3] # joint_angle + offset
        
        T_joint = compute_dh_matrix_fast(a, alpha, d, theta)
        T_total = T_total @ T_joint
    return T_total


# 机械臂类封装


class RoboticArm:
    def __init__(self):
        self.num_joints = 6
        #  DH 参数 [a, alpha, d, theta_offset]
        self.dh_params = np.array([
            [0,      0,        0.1625, 0], # Joint 1
            [0,     -np.pi/2,  0,      0], # Joint 2
            [0.425,  0,        0,      0], # Joint 3
            [0.3922, 0,        0,      0], # Joint 4
            [0,     -np.pi/2,  0.1333, 0], # Joint 5
            [0,      np.pi/2,  0.0997, 0]  # Joint 6
        ], dtype=np.float64)

    def forward_kinematics(self, joint_angles):
        q = np.array(joint_angles, dtype=np.float64)
        return fast_forward_kinematics(q, self.dh_params)

    def get_end_effector_position(self, joint_angles):
        T = self.forward_kinematics(joint_angles)
        return T[:3, 3] # 提取 X, Y, Z


# 测试模块


def benchmark_forward_kinematics(arm):
    print("\n" + "="*45)
    print("      正向运动学性能基准测试 (UR5 Params)")
    print("="*45)
    
    num_tests = 50000 
    # 预生成测试数据
    tests_angles = np.random.uniform(-np.pi, np.pi, (num_tests, 6))

    # Numba 预热 (第一次编译)
    _ = arm.forward_kinematics(tests_angles[0])

    start_time = time.time()
    for angles in tests_angles:
        arm.forward_kinematics(angles)
    elapsed_time = time.time() - start_time
    
    ops_per_second = num_tests / elapsed_time
    
    print(f"总测试次数: {num_tests}")
    print(f"总耗时: {elapsed_time:.4f} 秒")
    print(f"性能：{ops_per_second:.0f} ops/second")
    print(f"目标：≥10,000 ops/second")
    print(f"达标：{'√' if ops_per_second >= 10000 else '×'}")
    return ops_per_second

def test_accuracy(arm):
    print("\n--- 修正后的精度测试 ---")
    
    # 全 0 角度的理论值应该是：
    # X = 0.425 + 0.3922 = 0.8172
    # Y = 0.1333
    # Z = 0.1625 - 0.0997 = 0.0628
    
    test_cases = [
        {
            'name': 'UR5全零位姿',
            'angles': [0, 0, 0, 0, 0, 0],
            'expected': [0.8172, 0.1333, 0.0628], 
            'tolerance': 0.0001 # 容差可以设得非常小
        }
    ]

    for i, case in enumerate(test_cases):
        actual_pos = arm.get_end_effector_position(case['angles'])
        expected_pos = np.array(case['expected'])
        error = np.linalg.norm(actual_pos - expected_pos)
        
        print(f"用例 {i+1} [{case['name']}]:")
        print(f"  期望: {expected_pos}")
        print(f"  实际: {actual_pos.round(4)}")
        print(f"  误差: {error * 1000:.6f} mm") # 误差现在应该是 0.0000xx mm
        print(f"  达标：{'√' if error <= case['tolerance'] else '×'}")

if __name__ == "__main__":
    # 强制 Python 把当前目录加入搜索范围
    sys.path.append(os.getcwd())
    
    my_arm = RoboticArm()
    benchmark_forward_kinematics(my_arm)
    test_accuracy(my_arm)



