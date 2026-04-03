import sys
import os

# 强制告诉 Python 把当前目录加入搜索范围
sys.path.append(os.getcwd())

import numpy as np
import time
from src.robotic_arm import RoboticArm

def benchmark_forward_kinematics():
    """ 
    正向运动学性能基准测试
    """
    arm = RoboticArm(num_joints=6)  # 创建一个6关节的机械臂
    #预生成测试数据
    num_tests = 10000
    tests_angles = np.random.uniform(-np.pi, np.pi, (num_tests, 6))
# --- 核心改进：预热 ---
    arm.forward_kinematics(tests_angles[0]) 
    
    start_time = time.time()
    for angles in tests_angles:
        arm.forward_kinematics(angles)
    elapsed_time = time.time() - start_time
    
    ops_per_second = num_tests / elapsed_time
    print(f"正向运动学性能测试:")
    print(f"总测试次数: {num_tests}")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"性能：{ops_per_second:.0f} ops/second")
    print(f"目标：≥1000 ops/second")
    print(f"达标：{'√' if ops_per_second >= 10000 else '×'}")
    return ops_per_second

def test_accuracy():
    """
    精度测试
    """
    arm = RoboticArm(num_joints=6) 
    
    # 预生成测试数据
    test_cases = [
        {
            'name': '全零位姿 (UR5)',
            'angles': [0, 0, 0, 0, 0, 0],
            'expected': [0.8172, 0.1333, 0.0628], 
            'tolerance': 0.001
        },
        # 测试用例: 1 轴转 90 度
        {
            'name': '基座旋转90度',
            'angles': [np.pi/2, 0, 0, 0, 0, 0],
            'expected': [-0.1333, 0.8172, 0.0628], 
            'tolerance': 0.001
        }
    ]
    
    print(f"\n精度测试:")
    for i, case in enumerate(test_cases):
        arm.set_joint_angles(case['angles'])
        actual_pos = arm.get_end_effector_position()
        expected_pos = np.array(case['expected'])
        
        error = np.linalg.norm(actual_pos - expected_pos)
        
        print(f"测试用例 {i+1} ({case['name']}):")
        print(f"  期望位置：{expected_pos}")
        print(f"  实际位置：{actual_pos.round(4)}")
        print(f"  误差: {error * 1000:.4f} mm") 
        print(f"  达标：{'√' if error <= case['tolerance'] else '×'}")
if __name__ == "__main__":
    benchmark_forward_kinematics()
    test_accuracy()    