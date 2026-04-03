import trajectory
import numpy as np
if __name__ == "__main__":
    start = [
            [0, 0.1625, 0,      0],        # Joint 1
            [0, 0,      0,     -np.pi/2],  # Joint 2
            [0, 0,      0.425,  0],        # Joint 3
            [0, 0,      0.3922, 0],        # Joint 4
            [0, 0.1333, 0,     -np.pi/2],  # Joint 5
            [0, 0.0997, 0,      np.pi/2]   # Joint 6
        ]
    endd = [
            [0, 0.1625, 0,      0],        # Joint 1
            [0, 0,      0,     -np.pi/2],  # Joint 2
            [0, 0,      0.425,  -np.pi/2],        # Joint 3
            [0, 0,      0.3922, 0],        # Joint 4
            [0, 0.1333, 0,     -np.pi/2],  # Joint 5
            [0, 0.0997, 0,      np.pi/2]   # Joint 6
        ]
    # 1. Instantiate the class
    planner = trajectory.TrajectoryPlanner() 
    
    # 2. Call the method on the instance
    planner.plan_trajectory(start, endd, duration=2.0, method='cubic', num_points=50)