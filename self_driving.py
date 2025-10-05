import time, math, cv2
import numpy as np
from vilib import Vilib
from picarx import Picarx
from time import sleep
import matplotlib.pyplot as plt
import heapq

# =====================
# Config
# =====================
N = 100  # Grid size (N x N)
CELL_SIZE = 3  # Each cell in the grid represents 3x3 cm
SCAN_INTERVAL = 10  # Number of steps to take before re-scanning the environment
ROBOT_WIDTH_CM = 10 # Physical width of the robot in cm
ROBOT_PADDING_CELLS = int((ROBOT_WIDTH_CM / 2) / CELL_SIZE) # Calculate padding cells needed to account for robot's width

# --- State Flag ---
stop_sign_action_performed = False # Global flag to ensure the 5-second stop happens only once

# --- Physical Robot Constants ---
SENSOR_OFFSET_CM = 5.0  # Distance of ultrasonic sensor from the robot's center pivot point (cm)
TURN_SPEED = 20 # Speed for turning maneuvers
DRIVE_SPEED = 25 # Speed for forward movement

# --- Grid Initialization ---
# The map is represented as a grid.
# -1: Unknown, 0: Free, 1: Obstacle, 2: Stop Sign
grid = -np.ones((N, N), dtype=int)

# =====================
# Helper functions
# =====================
def world_to_grid(x, y):
    """Converts real-world coordinates (cm) to grid cell indices."""
    col = int(x / CELL_SIZE) + N // 2
    row = int(y / CELL_SIZE) + N // 2
    return col, row

def grid_to_world(col, row):
    """Converts grid cell indices back to real-world coordinates (cm)."""
    x = (col - N // 2) * CELL_SIZE
    y = (row - N // 2) * CELL_SIZE
    return x, y

# =====================
# Mapping
# =====================
def bresenham(x0, y0, x1, y1):
    """Implements Bresenham's line algorithm to find all grid cells along a line."""
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return points

def scan_environment(px, pose):
    """
    Sweeps sensor, updates map, and returns True if a stop sign was seen.
    The stopping action is handled by the main loop, not this function.
    """
    x0, y0, theta = pose
    col0, row0 = world_to_grid(x0, y0)
    sign_found_this_scan = False

    sensor_mount_x = x0 + SENSOR_OFFSET_CM * math.cos(theta)
    sensor_mount_y = y0 + SENSOR_OFFSET_CM * math.sin(theta)

    max_range_cm = 150

    for angle in range(-90, 91, 5):
        px.set_cam_pan_angle(angle)
        time.sleep(0.1)
        r = px.ultrasonic.read()

        # --- STOP SIGN DETECTION LOGIC ---
        # Always check for and map the stop sign, regardless of the global flag
        if Vilib.traffic_sign_obj_parameter['t'] == 'stop':
            sign_found_this_scan = True # Note that we saw a sign during this scan
            
            # Estimate stop sign position and mark it on the map
            if r is not None and 0 < r <= max_range_cm:
                sensor_pan_rad = math.radians(-angle)
                world_angle = theta + sensor_pan_rad
                
                sx_end = sensor_mount_x + r * math.cos(world_angle)
                sy_end = sensor_mount_y + r * math.sin(world_angle)
                col_stop, row_stop = world_to_grid(sx_end, sy_end)

                if 0 <= col_stop < N and 0 <= row_stop < N:
                    grid[row_stop, col_stop] = 2 # Mark as Stop Sign Obstacle (Red)
        # --- END OF STOP SIGN LOGIC ---

        sensor_pan_rad = math.radians(-angle)
        world_angle = theta + sensor_pan_rad

        if r is not None and 0 < r <= max_range_cm:
            ray_end_dist = r
            obstacle_detected = True
        else:
            ray_end_dist = max_range_cm
            obstacle_detected = False

        gx_end = sensor_mount_x + ray_end_dist * math.cos(world_angle)
        gy_end = sensor_mount_y + ray_end_dist * math.sin(world_angle)
        col_end, row_end = world_to_grid(gx_end, gy_end)

        ray_cells = bresenham(col0, row0, col_end, row_end)

        num_free_cells = len(ray_cells) - 1 if obstacle_detected else len(ray_cells)
        for i in range(num_free_cells):
            col, row = ray_cells[i]
            if 0 <= col < N and 0 <= row < N:
                grid[row, col] = 0

        if obstacle_detected:
            col_obs, row_obs = ray_cells[-1]
            if 0 <= col_obs < N and 0 <= row_obs < N:
                if grid[row_obs, col_obs] != 2:
                    grid[row_obs, col_obs] = 1

    px.set_cam_pan_angle(0)

    return sign_found_this_scan

# =====================
# A* Pathfinding
# =====================
def is_cell_blocked_for_robot(col, row):
    """Checks if a cell and its surrounding padding area are free of obstacles (including stop signs)."""
    r_min = max(0, row - ROBOT_PADDING_CELLS)
    r_max = min(N-1, row + ROBOT_PADDING_CELLS)
    c_min = max(0, col - ROBOT_PADDING_CELLS)
    c_max = min(N-1, col + ROBOT_PADDING_CELLS)
    
    sub_grid = grid[r_min:r_max+1, c_min:c_max+1]
    if np.any((sub_grid == 1) | (sub_grid == 2)):
        return True
    return False

def neighbors(node):
    """Gets valid, non-blocked neighboring cells for the A* algorithm."""
    col, row = node
    directions = [(0,1), (0,-1), (1,0), (-1,0)]
    for dcol, drow in directions:
        nc, nr = col + dcol, row + drow
        if 0 <= nc < N and 0 <= nr < N:
            if not is_cell_blocked_for_robot(nc, nr):
                yield (nc, nr)

def heuristic(a, b):
    """Calculates the Manhattan distance heuristic for A*."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(start, goal):
    """Finds the shortest path from a start to a goal node using the A* algorithm."""
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from, gscore = {}, {start: 0}
    fscore = {start: heuristic(start, goal)}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        for neighbor in neighbors(current):
            tentative_gscore = gscore[current] + 1
            if tentative_gscore < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_gscore
                fscore[neighbor] = tentative_gscore + heuristic(neighbor, goal)
                heapq.heappush(open_set, (fscore[neighbor], neighbor))
    return []

# =====================
# Driving Logic
# =====================
def pivot_turn(px, turn_degrees):
    """Performs an in-place turn by alternating steering and moving FWD/BWD slightly."""
    if abs(turn_degrees) < 5:
        return

    direction = -1 if turn_degrees > 0 else 1
    target_angle = abs(turn_degrees)
    
    forward_time = 0.25
    back_time = 0.25
    iterations = int(target_angle / 15 - 1)

    for _ in range(iterations):
        px.set_dir_servo_angle(-35 * direction)
        px.backward(DRIVE_SPEED)
        time.sleep(back_time)
        px.stop()
        time.sleep(0.05)

        px.set_dir_servo_angle(35 * direction)
        px.forward(DRIVE_SPEED)
        time.sleep(forward_time)
        px.stop()
        time.sleep(0.05)

    px.set_dir_servo_angle(0)

def move_to_cell(px, current_pose, next_col, next_row):
    """Calculates the required turn and drive to move from the current pose to a target grid cell."""
    x_cur, y_cur, theta_cur = current_pose
    x_target, y_target = grid_to_world(next_col, next_row)
    
    target_theta = math.atan2(y_target - y_cur, x_target - x_cur)
    angle_diff = target_theta - theta_cur
    
    while angle_diff > math.pi: angle_diff -= 2 * math.pi
    while angle_diff < -math.pi: angle_diff += 2 * math.pi

    pivot_turn(px, math.degrees(angle_diff))

    distance_to_drive = math.sqrt((x_target - x_cur)**2 + (y_target - y_cur)**2)
    drive_time = distance_to_drive / DRIVE_SPEED if DRIVE_SPEED > 0 else 0
    if drive_time > 0:
        px.set_dir_servo_angle(0)
        px.forward(DRIVE_SPEED)
        time.sleep(drive_time)
        px.stop()

    return (x_target, y_target, target_theta)

# =====================
# Visualization
# =====================
fig, ax = plt.subplots(figsize=(8, 8))
img_plot = None
def visualize_map(path=None, pose=None):
    """Updates and displays a visual representation of the map, path, and robot."""
    global img_plot
    img = np.full((N, N, 3), 128, dtype=np.uint8) # Gray for unknown
    img[grid == 0] = (255, 255, 255) # White for free space
    img[grid == 1] = (0, 0, 0)       # Black for obstacles
    img[grid == 2] = (255, 0, 0)       # Red for stop sign

    if path:
        for col, row in path:
            if 0 <= col < N and 0 <= row < N:
                img[row, col] = (0, 255, 0) # Green for path

    if pose:
        col, row = world_to_grid(pose[0], pose[1])
        if 0 <= col < N and 0 <= row < N:
            img[row, col] = (0, 0, 255) # Blue for robot

    if img_plot is None:
        img_plot = ax.imshow(img, origin='lower')
        ax.axis('off')
    else:
        img_plot.set_data(img)
    plt.pause(0.01)
    plt.savefig("map.png", dpi=100)

# =====================
# Main Control Loop
# =====================
def drive_to(px, destinations):
    """Main control loop that navigates the robot through a series of destination points."""
    global stop_sign_action_performed
    pose = (0, 0, math.pi / 2)
    step_count = 0

    print("Initial scan...")
    sign_was_seen = scan_environment(px, pose)
    visualize_map(pose=pose)

    # Perform the one-time stop action if a sign was seen
    if sign_was_seen and not stop_sign_action_performed:
        print("Stop sign detected during scan. Stopping for 5 seconds.")
        px.stop()
        time.sleep(5)
        stop_sign_action_performed = True

    for dest_idx, dest in enumerate(destinations):
        print(f"\n--- Navigating to destination {dest_idx+1}: {dest} ---")
        while True:
            current_col, current_row = world_to_grid(pose[0], pose[1])
            dest_col, dest_row = world_to_grid(dest[0], dest[1])

            if current_col == dest_col and current_row == dest_row:
                print(f"Destination {dest_idx+1} reached!")
                break

            if step_count > 0 and step_count % SCAN_INTERVAL == 0:
                print("Re-scanning...")
                sign_was_seen = scan_environment(px, pose)
                
                # Perform the one-time stop action if a sign was seen
                if sign_was_seen and not stop_sign_action_performed:
                    print("Stop sign detected during scan. Stopping for 5 seconds.")
                    px.stop()
                    time.sleep(5)
                    stop_sign_action_performed = True

            path = astar((current_col, current_row), (dest_col, dest_row))
            visualize_map(path=path, pose=pose)

            if not path or len(path) < 2:
                print("No valid path found. Cannot proceed.")
                break

            target_col, target_row = path[1]
            pose = move_to_cell(px, pose, target_col, target_row)
            step_count += 1

# =====================
# Main
# =====================
if __name__ == "__main__":
    """Initializes the robot and visualization, defines destinations, and starts navigation."""
    px = Picarx()
    try:
        Vilib.camera_start(vflip=False, hflip=False)
        Vilib.display(local=True, web=True)
        Vilib.traffic_detect_switch(True)
        plt.ion()
        sleep(2)

        destinations = [(0, 140)]
        drive_to(px, destinations)

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        print("Stopping robot and cleaning up.")
        px.stop()
        Vilib.camera_close()
        cv2.destroyAllWindows()
        plt.ioff()
        visualize_map()
        plt.show()
