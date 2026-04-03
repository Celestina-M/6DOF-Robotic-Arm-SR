import numpy as np
import sys
import os

# 寻找 src 里的代码
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def verify_math():
    # 1. 初始化 2 连杆机械臂
    L1, L2 = 0.1, 0.3  # 假设连杆长度
    arm = RobotArmForward(num_joints=2)
    arm.link_lengths = [L1, L2]
    
    # 2. 设定一组测试角度 (弧度)
    theta1 = np.pi/4  # 45度
    theta2 = np.pi/6  # 30度
    arm.set_joint_angles([theta1, theta2])
    
    # 3. 计算 FK 
    T_class, _ = arm.forward_kinematics()
    
    # 对dh参数计算
    # 公式：x = L1*cos(t1) + L2*cos(t1+t2)
    #          y = L1*sin(t1) + L2*sin(t1+t2)
    c12 = np.cos(theta1 + theta2)
    s12 = np.sin(theta1 + theta2)
    c1 = np.cos(theta1)
    s1 = np.sin(theta1)
    
    T_handwritten = np.array([
        [c12, -s12, 0, L1*c1 + L2*c12],
        [s12,  c12, 0, L1*s1 + L2*s12],
        [0,    0,   1, 0],
        [0,    0,   0, 1]
    ])
    
    # 5. 对比结果
    print("--- 验证结果 ---")
    print("代码计算出的矩阵:\n", np.round(T_class, 4))
    print("\n手写公式计算出的矩阵:\n", np.round(T_handwritten, 4))
    
    # 判断是否足够接近（允许极小的浮点数误差）
    if np.allclose(T_class, T_handwritten):
        print("\n✅ 验证成功！你的代码实现与数学公式完全一致。")
    else:
        print("\n❌ 验证失败！请检查 DH 参数或矩阵乘法顺序。")

if __name__ == "__main__":
    verify_math()