import airsim
import time
import numpy as np
import random
import math

client = airsim.MultirotorClient()
vehicle_name = "SimpleFlight"  # Set to match your settings.json
client.confirmConnection()
client.enableApiControl(True, vehicle_name=vehicle_name)
client.armDisarm(True, vehicle_name=vehicle_name)
client.takeoffAsync(vehicle_name=vehicle_name).join()

print("Drone is ready.")

# === Fixed Target Location ===
TARGET_X = 3230
TARGET_Y = 17200.0
TARGET_Z = -10.0  # Set to a safe altitude above ground (NED: negative is up)

AVOID_DIST = 5
AVOID_ALT = 2  # Altitude change per up/down attempt
MIN_ALTITUDE = -30.0  # Maximum height above ground (NED: negative is up)
MAX_ALTITUDE = -2.0   # Minimum height above ground (NED: negative is up)
ALTITUDE_THRESHOLD = 100.0  # If altitude drops below this, print a warning
VELOCITY_THRESHOLD = 0.5    # If velocity is near zero, print a warning
STUCK_CYCLES = 3            # Number of cycles to consider as stuck
IGNORE_OBJECTS = ['Plane9', 'Plane2_12', 'Ground', 'Sky']
YAW_THRESHOLD_DEG = 5       # Only rotate if yaw difference is greater than this
SAFE_DEPTH_THRESHOLD = 5.0  # Minimum clear distance (meters) to move forward
MAX_YAW_ATTEMPTS = 6        # Number of yaw attempts before trying up/down
MAX_UP_ATTEMPTS = 3         # Number of up attempts before trying down
MAX_DOWN_ATTEMPTS = 3       # Number of down attempts before giving up

# For stuck detection
stuck_counter = 0

# Helper: get yaw (in degrees) from current position to target
def get_yaw_to_target(current_pos, target_pos):
    dx = target_pos.x_val - current_pos.x_val
    dy = target_pos.y_val - current_pos.y_val
    return math.degrees(math.atan2(dy, dx))

def check_collision_strict():
    collision = client.simGetCollisionInfo(vehicle_name=vehicle_name)
    # Only consider as collision if not in IGNORE_OBJECTS
    reliable_hit = (
        (collision.has_collided or collision.penetration_depth > 0.01) and
        collision.object_name not in IGNORE_OBJECTS
    )
    print("[DEBUG] Collision info:", collision)
    print(f"[DEBUG] Object name: {collision.object_name}, Impact point: ({collision.impact_point.x_val:.2f}, {collision.impact_point.y_val:.2f}, {collision.impact_point.z_val:.2f})")
    return reliable_hit, collision.normal, collision

# Helper: get drone's current yaw (heading) in radians
def get_drone_yaw():
    state = client.getMultirotorState(vehicle_name=vehicle_name)
    orientation = state.kinematics_estimated.orientation
    w, x, y, z = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw

# Helper: rotate drone to a specific yaw (in degrees) if not already facing it
def rotate_to_yaw_if_needed(target_yaw_deg):
    current_yaw_deg = math.degrees(get_drone_yaw())
    yaw_diff = (target_yaw_deg - current_yaw_deg + 180) % 360 - 180  # Shortest angle
    if abs(yaw_diff) > YAW_THRESHOLD_DEG:
        print(f"[ROTATE] Rotating to yaw {target_yaw_deg:.2f} degrees (current: {current_yaw_deg:.2f}, diff: {yaw_diff:.2f})")
        client.rotateToYawAsync(target_yaw_deg, vehicle_name=vehicle_name).join()
    else:
        print(f"[ROTATE] Already facing intended yaw (current: {current_yaw_deg:.2f}, target: {target_yaw_deg:.2f}, diff: {yaw_diff:.2f})")
    return math.radians(target_yaw_deg)

# Helper: get average depth in left, center, right regions from depth camera
def get_depth_regions():
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)
    ], vehicle_name=vehicle_name)
    response = responses[0]
    img1d = np.array(response.image_data_float, dtype=np.float32)
    img2d = img1d.reshape(response.height, response.width)
    thirds = response.width // 3
    left = img2d[:, :thirds]
    center = img2d[:, thirds:2*thirds]
    right = img2d[:, 2*thirds:]
    avg_depths = [np.mean(left), np.mean(center), np.mean(right)]
    return avg_depths

# Main navigation loop with bug/wall-following logic
def roam_and_avoid():
    global stuck_counter
    while True:
        state = client.getMultirotorState(vehicle_name=vehicle_name)
        pos = state.kinematics_estimated.position
        target = airsim.Vector3r(TARGET_X, TARGET_Y, TARGET_Z)
        desired_yaw_deg = get_yaw_to_target(pos, target)
        current_yaw_deg = math.degrees(get_drone_yaw())
        yaw_diff = (desired_yaw_deg - current_yaw_deg + 180) % 360 - 180
        print(f"[INFO] Moving to fixed target: ({target.x_val:.2f}, {target.y_val:.2f}, {target.z_val:.2f}) | Yaw to target: {desired_yaw_deg:.2f}, Current yaw: {current_yaw_deg:.2f}, Diff: {yaw_diff:.2f}")

        # Try to face the goal
        rotate_to_yaw_if_needed(desired_yaw_deg)
        # Update position and yaw after rotation
        state = client.getMultirotorState(vehicle_name=vehicle_name)
        pos = state.kinematics_estimated.position

        # Check if path to goal is clear
        yaw_attempts = 0
        up_attempts = 0
        down_attempts = 0
        while True:
            avg_depths = get_depth_regions()
            print(f"[DEPTH CHECK] left={avg_depths[0]:.2f}, center={avg_depths[1]:.2f}, right={avg_depths[2]:.2f}")
            if avg_depths[1] > SAFE_DEPTH_THRESHOLD:
                print("[DEPTH CHECK] Path to goal is clear. Moving forward.")
                yaw = get_drone_yaw()
                dx = AVOID_DIST * math.cos(yaw)
                dy = AVOID_DIST * math.sin(yaw)
                move_target = airsim.Vector3r(pos.x_val + dx, pos.y_val + dy, target.z_val)
                print(f"[MOVE] Moving forward to: x={move_target.x_val:.2f}, y={move_target.y_val:.2f}, z={move_target.z_val:.2f}")
                client.moveToPositionAsync(
                    move_target.x_val, move_target.y_val, move_target.z_val, velocity=5, vehicle_name=vehicle_name
                ).join()
                time.sleep(3)
                # Update position after move
                state = client.getMultirotorState(vehicle_name=vehicle_name)
                pos = state.kinematics_estimated.position
                break  # Exit the avoidance loop and continue main loop
            else:
                # Path to goal is blocked, try yawing left/right first
                if yaw_attempts < MAX_YAW_ATTEMPTS:
                    if avg_depths[0] > avg_depths[2]:
                        new_yaw_deg = math.degrees(get_drone_yaw()) + 30
                        print(f"[BUG] Path to goal blocked. Rotating left to yaw {new_yaw_deg:.2f} degrees.")
                        rotate_to_yaw_if_needed(new_yaw_deg)
                    else:
                        new_yaw_deg = math.degrees(get_drone_yaw()) - 30
                        print(f"[BUG] Path to goal blocked. Rotating right to yaw {new_yaw_deg:.2f} degrees.")
                        rotate_to_yaw_if_needed(new_yaw_deg)
                    # After rotation, update position and yaw, but DO NOT move yet
                    state = client.getMultirotorState(vehicle_name=vehicle_name)
                    pos = state.kinematics_estimated.position
                    yaw_attempts += 1
                    continue
                # If yaw attempts exhausted, try going up
                elif up_attempts < MAX_UP_ATTEMPTS and pos.z_val - AVOID_ALT > MIN_ALTITUDE:
                    new_z = pos.z_val - AVOID_ALT
                    print(f"[BUG] Yaw attempts exhausted. Trying to go UP to z={new_z:.2f}")
                    client.moveToPositionAsync(pos.x_val, pos.y_val, new_z, velocity=2, vehicle_name=vehicle_name).join()
                    time.sleep(2)
                    state = client.getMultirotorState(vehicle_name=vehicle_name)
                    pos = state.kinematics_estimated.position
                    up_attempts += 1
                    # After moving up, try to re-align with the goal
                    desired_yaw_deg = get_yaw_to_target(pos, target)
                    rotate_to_yaw_if_needed(desired_yaw_deg)
                    yaw_attempts = 0  # Reset yaw attempts after altitude change
                    continue
                # If up attempts exhausted, try going down
                elif down_attempts < MAX_DOWN_ATTEMPTS and pos.z_val + AVOID_ALT < MAX_ALTITUDE:
                    new_z = pos.z_val + AVOID_ALT
                    print(f"[BUG] Up attempts exhausted. Trying to go DOWN to z={new_z:.2f}")
                    client.moveToPositionAsync(pos.x_val, pos.y_val, new_z, velocity=2, vehicle_name=vehicle_name).join()
                    time.sleep(2)
                    state = client.getMultirotorState(vehicle_name=vehicle_name)
                    pos = state.kinematics_estimated.position
                    down_attempts += 1
                    # After moving down, try to re-align with the goal
                    desired_yaw_deg = get_yaw_to_target(pos, target)
                    rotate_to_yaw_if_needed(desired_yaw_deg)
                    yaw_attempts = 0  # Reset yaw attempts after altitude change
                    continue
                else:
                    print("[BUG] All avoidance attempts exhausted. Hovering in place.")
                    client.hoverAsync(vehicle_name=vehicle_name).join()
                    time.sleep(2)
                    break

        time.sleep(0.2)  # Give time for collision to register

        collided, normal, collision = check_collision_strict()
        state = client.getMultirotorState(vehicle_name=vehicle_name)
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        print(f"[DEBUG] Position: x={pos.x_val:.2f}, y={pos.y_val:.2f}, z={pos.z_val:.2f}")
        print(f"[DEBUG] Velocity: x={vel.x_val:.2f}, y={vel.y_val:.2f}, z={vel.z_val:.2f}")
        stuck = abs(vel.x_val) < VELOCITY_THRESHOLD and abs(vel.y_val) < VELOCITY_THRESHOLD and abs(vel.z_val) < VELOCITY_THRESHOLD
        if pos.z_val > ALTITUDE_THRESHOLD:
            print(f"[WARNING] Altitude above threshold: {pos.z_val:.2f}")
        # Increment stuck counter if stuck, else reset
        if stuck:
            stuck_counter += 1
        else:
            stuck_counter = 0
        # Only trigger avoidance if stuck for STUCK_CYCLES or a reliable collision
        if (stuck_counter >= STUCK_CYCLES) or collided:
            print("[INFO] Collision or stuck detected, re-running bug/avoidance logic.")
            continue  # Go back to the start of the main loop
        else:
            print("[INFO] No collision detected, continuing to target.")

        time.sleep(1)

try:
    roam_and_avoid()
except KeyboardInterrupt:
    print("Mission aborted by user.")
    try:
        client.hoverAsync(vehicle_name=vehicle_name).join()
        client.armDisarm(False, vehicle_name=vehicle_name)
        client.enableApiControl(False, vehicle_name=vehicle_name)
    except RuntimeError as e:
        if "IOLoop is already running" in str(e):
            print("[INFO] Cleanup skipped: IOLoop is already running due to KeyboardInterrupt.")
        else:
            raise 