from controller import Robot
import math
import numpy as np
from heapq import heappush, heappop
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum
import random
import cv2
from dataclasses import dataclass

class RobotState(Enum):
    IDLE = "idle"
    TURNING = "turning"
    MOVING = "moving"
    AVOIDING = "avoiding"

@dataclass
class Position:
    x: float
    y: float
    bearing: float

@dataclass
class GridCell:
    x: int
    y: int

@dataclass
class Treasure:
    x: float
    y: float
    found_at: float

class GridState(Enum):
    UNEXPLORED = 0
    EXPLORED = 1
    OBSTACLE = 2
    ROBOT = 3
    TREASURE = 4
    
@dataclass
class OdometryState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

class Odometry: #calculate odometry
    def __init__(self):
        self.wheel_radius = 0.025
        self.distance_between_wheels = 0.09
        self.wheel_circum = 2 * math.pi * self.wheel_radius
        self.encoder_unit = self.wheel_circum/6.28
        self.robot_pose = [0, 0, 0]  # x, y, theta
        self.last_ps_values = [0, 0, 0, 0]
        
    def update_position(self): 
        gps_pos = self.get_position()
        compass_bearing = self.get_bearing()
        
        # Get all encoder values
        encoder_values = [sensor.getValue() for sensor in self.position_sensors]
        
        # Update odometry and retrieve the robot pose as [x, y, theta]
        odom_state = self.odometry.update(encoder_values, self.TIME_STEP / 1000.0)
        
        # Sensor fusion with configurable weights
        alpha = 0.7  # Weight for GPS
        fused_x = alpha * gps_pos[0] + (1 - alpha) * odom_state[0]
        fused_y = alpha * gps_pos[1] + (1 - alpha) * odom_state[1]
        
        self.current_position = (fused_x, fused_y)
        self.current_bearing = compass_bearing
    
        return self.current_position, self.current_bearing
           
    def update(self, ps_values, dt):
        dist_values = [0, 0, 0, 0]
        
        # Calculate distances for each wheel
        for i in range(4):
            diff = ps_values[i] - self.last_ps_values[i]
            if diff < 0.001:
                diff = 0
                ps_values[i] = self.last_ps_values[i]
            dist_values[i] = diff * self.encoder_unit
            
        # Average left and right sides
        left_dist = (dist_values[0] + dist_values[2]) / 2.0
        right_dist = (dist_values[1] + dist_values[3]) / 2.0
        
        v = (left_dist + right_dist) / 2.0
        w = (right_dist - left_dist) / self.distance_between_wheels
        
        self.robot_pose[2] += (w * dt)
        
        vx = v * math.cos(self.robot_pose[2])
        vy = v * math.sin(self.robot_pose[2])
        
        self.robot_pose[0] += (vx * dt)
        self.robot_pose[1] += (vy * dt)
        
        # Store values for next iteration
        self.last_ps_values = ps_values.copy()
        
        return OdometryState(x=self.robot_pose[0], y=self.robot_pose[1], theta=self.robot_pose[2])
class RobotController:
    def __init__(self):
        
        self.TIME_STEP = 64
        self.BASE_SPEED = 10.0
        self.TURN_SPEED_FAST = 5.0    
        self.TURN_SPEED_SLOW = 1.0   
        self.OBSTACLE_THRESHOLD = 950.0
        self.TARGET_REACHED_THRESHOLD = 0.05  
        self.ANGLE_THRESHOLD = 2.0    
        self.MIN_COORD = -2.10  
        self.MAX_COORD = 2.10
        self.GRID_SIZE = 10 
        self.CELL_SIZE = 0.5
        # Obstacle avoidance 
        self.avoid_obstacle_counter = 0
        self.TURN_STEPS = 15  
        self.FORWARD_STEPS = 5  
        self.avoidance_phase = "turn" 
        self.AVOID_OBSTACLE_STEPS = 100
        
        self.robot = Robot() #initalise robot
        self.setup_robot()
        
        self.odometry = Odometry()

        self.current_target = None
        self.visited_cells = set()
        self.current_bearing = 0.0  
        
        self.treasures_found = []
        self.treasure_count = 0
        self.detection_cooldown = 0 
        self.REQUIRED_TREASURES = 5
        
        self.known_treasures = [(0, 0), (0.5, -1.5), (1.5, -0.5),  (1.5, 2), (-1.5, 1)]
        self.camera_width = self.camera.getWidth()
        self.camera_height = self.camera.getHeight()
        self.camera_fov = self.camera.getFov()
        self.MIN_TREASURE_AREA = 50
        self.DETECTION_COOLDOWN_STEPS = 20
        self.grid = [[GridState.UNEXPLORED.value] * self.GRID_SIZE 
                     for _ in range(self.GRID_SIZE)]
    
    def setup_robot(self): #set up robot
        self.setup_sensors()
        self.setup_wheels()
    
    def setup_encoders(self): #set up encoder values
        self.position_sensors = []
        for name in ['ps_1', 'ps_2', 'ps_3', 'ps_4']:
            sensor = self.robot.getDevice(name)
            sensor.enable(self.TIME_STEP)
            self.position_sensors.append(sensor)
        
        # Initialize odometry with robot dimensions
        WHEEL_RADIUS = 0.05  # meters
        WHEEL_BASE = 0.2     # meters front-to-back
        WHEEL_SEPARATION = 0.25  # meters left-to-right
        self.odometry = Odometry(WHEEL_RADIUS, WHEEL_BASE, WHEEL_SEPARATION)
        
    def setup_sensors(self): #set up sensors
        self.ds = []
        for name in ['ds_right', 'ds_left']:
            sensor = self.robot.getDevice(name)
            sensor.enable(self.TIME_STEP)
            self.ds.append(sensor)
    
        # Initialize position sensors
        self.position_sensors = []
        for name in ['ps_1', 'ps_2', 'ps_3', 'ps_4']:
            sensor = self.robot.getDevice(name)
            sensor.enable(self.TIME_STEP)
            self.position_sensors.append(sensor)
    
        self.gps = self.robot.getDevice('gps')
        self.compass = self.robot.getDevice('compass')
        self.camera = self.robot.getDevice('CAM')
        
        self.camera.recognitionEnable(self.TIME_STEP)
        
        for sensor in [self.gps, self.compass, self.camera]:
            sensor.enable(self.TIME_STEP)
    
    def setup_wheels(self): #set up wheels
        self.wheels = []
        for name in ['wheel1', 'wheel2', 'wheel3', 'wheel4']:
            wheel = self.robot.getDevice(name)
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0.0)
            self.wheels.append(wheel)
    
    def world_to_grid(self, x: float, y: float) -> GridCell: #convert co-ords to grid 
        grid_x = int((x - self.MIN_COORD) / self.CELL_SIZE)
        grid_y = int((y - self.MIN_COORD) / self.CELL_SIZE)
        grid_x = max(0, min(grid_x, self.GRID_SIZE - 1))
        grid_y = max(0, min(grid_y, self.GRID_SIZE - 1))
        
        return GridCell(x=grid_x, y=grid_y)

    def grid_to_world(self, cell: GridCell) -> Tuple[float, float]:
        world_x = round((cell.x * self.CELL_SIZE + self.MIN_COORD + (self.CELL_SIZE / 2)), 3)
        world_y = round((cell.y * self.CELL_SIZE + self.MIN_COORD + (self.CELL_SIZE / 2)), 3)
        return (world_x, world_y)
        
    def detect_obstacles(self): #detects obstacles
        if self.avoid_obstacle_counter > 0:
            return False
                
        for sensor in self.ds:
            value = sensor.getValue()
            if value < self.OBSTACLE_THRESHOLD: #check for obstacle
                robot_pos = self.get_position()
                bearing = math.radians(self.get_bearing())
                dist = value / 1000.0 
                x = robot_pos[0] + dist * math.cos(bearing)
                y = robot_pos[1] + dist * math.sin(bearing)
                
                # Check if coordinates match any known treasure location
                for treasure_x, treasure_y in self.known_treasures: #compare current to treasure
                    if (abs(x - treasure_x) < 0.3 and 
                        abs(y - treasure_y) < 0.3):
                        return False  
                        
                grid_cell = self.world_to_grid(x, y)
                if self.grid[grid_cell.y][grid_cell.x] == GridState.TREASURE.value:
                    return False
                    
                self.avoid_obstacle_counter = self.AVOID_OBSTACLE_STEPS
                return True
        return False

    def manhattan_distance(self, start: GridCell, goal: GridCell) -> float:
        return abs(start.x - goal.x) + abs(start.y - goal.y)

    def find_path(self, start: GridCell, goal: GridCell) -> List[Tuple[float, float]]: #finds path from A-B
        open_set = []
        closed_set = set()
        came_from = {}
        g_score = {(start.x, start.y): 0}
        f_score = {(start.x, start.y): self.manhattan_distance(start, goal)}
        heappush(open_set, (f_score[(start.x, start.y)], (start.x, start.y)))
        while open_set:
            current = heappop(open_set)[1]
            if current == (goal.x, goal.y):
                path = []
                while current in came_from:
                    cell = GridCell(x=current[0], y=current[1])
                    path.append(self.grid_to_world(cell))
                    current = came_from[current]
                path.append(self.grid_to_world(start))
                path.reverse()
                return path
            
            closed_set.add(current)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_x = current[0] + dx
                next_y = current[1] + dy
                if (0 <= next_x < self.GRID_SIZE and 
                    0 <= next_y < self.GRID_SIZE and 
                    self.grid[next_y][next_x] != GridState.OBSTACLE.value):
                    neighbor = (next_x, next_y)
                    if neighbor in closed_set:
                        continue
                        
                    tentative_g = g_score[current] + 1
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self.manhattan_distance(
                            GridCell(x=next_x, y=next_y),
                            goal
                        )
                        heappush(open_set, (f_score[neighbor], neighbor))
        direct_path = [
            self.grid_to_world(start),
            self.grid_to_world(goal)
        ]
        return direct_path
            
    def normalize_angle(self, angle): #evaluates angle
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
        
    def select_random_target(self, robot_position: Tuple[float, float], occupancy_grid: List[List[float]]) -> Optional[Tuple[float, float]]: #new target
        if self.current_target is not None:
            return self.current_target
                
        available_positions = []
        current_cell = self.world_to_grid(robot_position[0], robot_position[1])
        for x in range(-22, 23): 
            for y in range(-22, 23):
                world_x = x * 0.1
                world_y = y * 0.1
                if abs(world_x) <= 2.25 and abs(world_y) <= 2.25:
                    grid_cell = self.world_to_grid(world_x, world_y)
                    if (0 <= grid_cell.x < self.GRID_SIZE and 
                        0 <= grid_cell.y < self.GRID_SIZE and 
                        occupancy_grid[grid_cell.y][grid_cell.x] == GridState.UNEXPLORED.value and 
                        (grid_cell.x, grid_cell.y) not in self.visited_cells):
                        distance = math.sqrt(
                            (world_x - robot_position[0])**2 + 
                            (world_y - robot_position[1])**2
                        )
                        if distance >= 0.3:  
                            available_positions.append((world_x, world_y))
        
        if not available_positions:
            return None
        target_x, target_y = random.choice(available_positions)
        self.current_target = (target_x, target_y)
        dx = target_x - robot_position[0]
        dy = target_y - robot_position[1]
        quadrant = ""
        if dx >= 0 and dy >= 0:
            quadrant = "Q1 (0-90°)"
        elif dx >= 0 and dy < 0:
            quadrant = "Q2 (90-180°)"
        elif dx < 0 and dy < 0:
            quadrant = "Q3 (180-270°)"
        else:
            quadrant = "Q4 (270-360°)"
                
        print(f"Selected target at world coordinates ({target_x:.1f}, {target_y:.1f})")
        print(f"Target is in {quadrant}")
        return self.current_target
        
    def get_bearing(self) -> float: #currebt bearing
        north = self.compass.getValues()
        rad = math.atan2(north[0], north[2])
        bearing = math.degrees(rad)
        if bearing < 0:
            bearing += 360
        bearing = (bearing + 90) % 360
        return bearing
    
    def get_target_bearing(self, current_pos: Tuple[float, float], target_pos: Tuple[float, float]) -> float: #target bearing
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        raw_angle = math.atan2(dy, dx)
        degrees = math.degrees(raw_angle)
        bearing = 90 - degrees
        if bearing < 0:
            bearing += 360
        if dx >= 0 and dy >= 0: 
            bearing = max(0, min(90, bearing))
        elif dx >= 0 and dy < 0:  
            bearing = max(90, min(180, bearing))
        elif dx < 0 and dy < 0:  
            bearing = max(180, min(270, bearing))
        else: 
            bearing = max(270, min(360, bearing))
        return bearing
        
    def calculate_angle_difference(self, current_bearing: float, target_bearing: float) -> float: #angle difference
        diff = target_bearing - current_bearing
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        return (abs(diff), diff>0)
              
    def get_position(self) -> Tuple[float, float]: #current positon
        gps_values = self.gps.getValues()
        x = round(gps_values[0], 3)
        y = round(gps_values[1], 3)
        return (x, y)

    def execute_movement(self) -> Tuple[float, float]: #movement logic
        current_pos = self.get_position()
        
        # Get encoder values at the start
        encoder_values = [ps.getValue() for ps in self.position_sensors]
        
        if self.avoid_obstacle_counter > 0:
            self.avoid_obstacle_counter -= 1
            if self.avoidance_phase == "turn":
                print(f"Avoiding obstacle - Turning phase: {self.avoid_obstacle_counter} steps remaining")
                if self.avoid_obstacle_counter <= 0:
                    self.avoid_obstacle_counter = self.FORWARD_STEPS
                    self.avoidance_phase = "forward"
                return (self.BASE_SPEED, -self.BASE_SPEED)  
            elif self.avoidance_phase == "forward":
                print(f"Avoiding obstacle - Forward phase: {self.avoid_obstacle_counter} steps remaining")
                if self.avoid_obstacle_counter <= 0:
                    self.avoidance_phase = "turn"
                return (self.BASE_SPEED, self.BASE_SPEED)
    
        if self.detect_obstacles():
            print("New obstacle detected! Starting avoidance maneuver")
            self.avoid_obstacle_counter = self.TURN_STEPS
            self.avoidance_phase = "turn"
            return (self.BASE_SPEED, -self.BASE_SPEED)  
    
        if not self.current_target:
            print("\nSelecting new target...")
            target = self.select_random_target(current_pos, self.grid)
            if not target:
                print("No unexplored targets remaining!")
                return (0.0, 0.0)
                
        # Get updated odometry state
        odom_state = self.odometry.update(encoder_values, self.TIME_STEP / 1000.0)
                
        target_bearing = self.get_target_bearing(current_pos, self.current_target)
        angle_diff, turn_right = self.calculate_turn_direction(self.current_bearing, target_bearing)
        dx = self.current_target[0] - current_pos[0]
        dy = self.current_target[1] - current_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)
    
        print(f"\n=== Robot Status ===")
        print(f"GPS Position: (x={current_pos[0]:.2f}, y={current_pos[1]:.2f})")
        print(f"Odometry Position: (x={odom_state.x:.2f}, y={odom_state.y:.2f})")
        print(f"Encoder Values: {[f'{value:.2f}' for value in encoder_values]}")
        print(f"Current Bearing: {self.current_bearing:.2f}°")
        print(f"Target Position: ({self.current_target[0]:.2f}, {self.current_target[1]:.2f})")
        print(f"Target Bearing: {target_bearing:.1f}°")
        print(f"Turn needed: {angle_diff:.1f}° {'right' if turn_right else 'left'}")
        print(f"Distance to target: {distance:.2f}m")
        print(f"Treasures found: {self.treasure_count}/5")
        
        if self.treasures_found:
            print("Found treasure coordinates:")
            for i, t in enumerate(self.treasures_found, 1):
                print(f"Treasure {i}: ({t.x:.3f}, {t.y:.3f})")
        else:
            print("No treasures found yet")
    
        if -45 <= self.current_bearing % 360 <= 45:
            print("(Facing North)")
        elif 45 < self.current_bearing % 360 <= 135:
            print("(Facing East)")
        elif 135 < self.current_bearing % 360 <= 225:
            print("(Facing South)")
        else:
            print("(Facing West)")
    
        if distance < self.TARGET_REACHED_THRESHOLD:
            print("\nTarget reached successfully!")
            target_cell = self.world_to_grid(self.current_target[0], self.current_target[1])
            print(f"Marking cell ({target_cell.x}, {target_cell.y}) as explored")
            self.grid[target_cell.y][target_cell.x] = GridState.EXPLORED.value
            self.visited_cells.add((target_cell.x, target_cell.y))
            self.current_target = None
            self.target_cell = None
            return (0.0, 0.0)
    
        if abs(angle_diff) > self.ANGLE_THRESHOLD:
            turn_speed = self.TURN_SPEED_SLOW if abs(angle_diff) < self.ANGLE_THRESHOLD else self.TURN_SPEED_FAST
            
            if turn_right:
                print(f"Turning right ({turn_speed:.1f})")
                self.current_bearing = (self.current_bearing + turn_speed) % 360
                return (turn_speed, -turn_speed)
            else:
                print(f"Turning left ({turn_speed:.1f})")
                self.current_bearing = (self.current_bearing - turn_speed) % 360
                return (-turn_speed, turn_speed)
    
        print("Moving forward")
        return (self.BASE_SPEED, self.BASE_SPEED)
        
    def calculate_turn_direction(self, current_bearing: float, target_bearing: float) -> Tuple[float, bool]:
        diff = target_bearing - current_bearing
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        return (abs(diff), diff > 0)
                
    def print_grid(self):
        print("\nOccupancy Grid (5m x 5m, centered at 0,0):")
        print("   ", end="")
        for x in range(self.GRID_SIZE):
            world_x = x * self.CELL_SIZE + self.MIN_COORD
            print(f"{world_x:6.1f}", end="")
        print("\n   " + "-" * (self.GRID_SIZE * 6 + 1))
        for y in range(self.GRID_SIZE):
            world_y = self.MAX_COORD - y * self.CELL_SIZE
            print(f"{world_y:4.1f}|", end="")
            for x in range(self.GRID_SIZE):
                if (x, y) in self.visited_cells:
                    print("  E   ", end="")
                else:
                    symbol = {
                        GridState.UNEXPLORED.value: "  *   ",
                        GridState.EXPLORED.value: "  E   ",
                        GridState.OBSTACLE.value: "  O   ",
                        GridState.TREASURE.value: "  T   " 
                    }[self.grid[y][x]]
                    print(symbol, end="")
            print("|")
    
    def set_wheel_speeds(self, left_speed: float, right_speed: float):
        for i, wheel in enumerate(self.wheels):
            wheel.setVelocity(left_speed if i % 2 == 0 else right_speed)

    def is_exploration_complete(self) -> bool: #finished? 
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y][x] == GridState.UNEXPLORED.value and (x, y) not in self.visited_cells:
                    return False
        return True
        
    def run_step(self): #run 
        self.detect_and_record_treasure()
        left_speed, right_speed = self.execute_movement()
        self.process_camera_image()
        for i in range(4):
            if i % 2 == 0:  
                self.wheels[i].setVelocity(left_speed)
            else:  
                self.wheels[i].setVelocity(right_speed)
        explored_count = sum(1 for y in range(self.GRID_SIZE) 
                           for x in range(self.GRID_SIZE) 
                           if self.grid[y][x] != GridState.UNEXPLORED.value 
                           or (x, y) in self.visited_cells)
        total_cells = self.GRID_SIZE * self.GRID_SIZE
        print(f"Exploration progress: {explored_count}/{total_cells} cells ({explored_count/total_cells*100:.1f}%)")
        self.print_grid()
#TREASURE LOGIC
    def is_known_treasure(self, coords):
        for treasure in self.known_treasures:
            distance = math.sqrt((coords[0] - treasure[0])**2 + (coords[1] - treasure[1])**2)
            if distance < 0.25: 
                return treasure
        return None
    
    def process_camera_image(self): #use camera data
        if self.detection_cooldown > 0:
            self.detection_cooldown -= 1
            return False
            
        image = self.camera.getImage()
        if not image:
            return False
            
        image = np.frombuffer(image, np.uint8).reshape((self.camera_height, self.camera_width, 4))
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Adjusted HSV values for better blue detection
        mask = cv2.inRange(hsv, np.array([100, 150, 150]), np.array([140, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.MIN_TREASURE_AREA:
                # Get distance from sensors
                min_distance = float('inf')
                for sensor in self.ds:
                    value = sensor.getValue()
                    if value < self.OBSTACLE_THRESHOLD:
                        min_distance = min(min_distance, value / 1000.0)
                        
                if min_distance <= 0.3:  # Increased detection range
                    robot_pos = self.get_position()
                    robot_bearing = math.radians(self.get_bearing())
                    
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        angle_h = ((cx / self.camera_width) - 0.5) * math.radians(self.camera_fov)
                        total_angle = robot_bearing + angle_h
                        
                        treasure_x = robot_pos[0] + min_distance * math.cos(total_angle)
                        treasure_y = robot_pos[1] + min_distance * math.sin(total_angle)
                        
                        # Check against known treasure locations with increased tolerance
                        for known_x, known_y in self.known_treasures:
                            if (abs(treasure_x - known_x) < 0.4 and 
                                abs(treasure_y - known_y) < 0.4):
                                self.add_treasure((known_x, known_y))
                                self.detection_cooldown = self.DETECTION_COOLDOWN_STEPS
                                return True
        return False
        
    def stop_robot(self): #reduces wheel velocity to 0
        for wheel in self.wheels:
            wheel.setVelocity(0.0)

    def detect_and_record_treasure(self): #find and record treasure
        image = self.camera.getImage()
        if image:
            frame = np.frombuffer(image, np.uint8).reshape((self.camera_height, self.camera_width, 4))
            frame = frame[:, :, :3]
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_frame, np.array([110, 150, 150]), np.array([130, 255, 255]))
            if cv2.countNonZero(mask) > 100:
                robot_pos = self.get_position()
                for treasure_x, treasure_y in self.known_treasures:
                    distance = math.sqrt(
                        (robot_pos[0] - treasure_x)**2 + 
                        (robot_pos[1] - treasure_y)**2
                    )
                    
                    if distance < 0.3:  
                        grid_cell = self.world_to_grid(treasure_x, treasure_y)
                        if self.grid[grid_cell.y][grid_cell.x] != GridState.TREASURE.value:
                            self.grid[grid_cell.y][grid_cell.x] = GridState.TREASURE.value
                            self.treasure_count += 1
                            self.treasures_found.append(Treasure(
                                x=treasure_x,
                                y=treasure_y,
                                found_at=self.robot.getTime()
                            ))
                            
                            print(f"\nTREASURE FOUND! ({self.treasure_count}/5)")
                            print(f"Location: ({treasure_x:.3f}, {treasure_y:.3f})")
                            return True
        return False
    
    def is_treasure_already_found(self, coords): #have we already found the treausre? 
        for found_x, found_y in self.treasures_found:
            if (abs(found_x - coords[0]) < 0.3 and 
                abs(found_y - coords[1]) < 0.3):
                return True
        return False

    def add_treasure(self, coords): #add treasure
        if abs(coords[0]) > 2.25 or abs(coords[1]) > 2.25:
            return
        grid_cell = self.world_to_grid(coords[0], coords[1])
        if self.grid[grid_cell.y][grid_cell.x] == GridState.TREASURE.value:
            return
        self.grid[grid_cell.y][grid_cell.x] = GridState.TREASURE.value
        self.treasure_count += 1
        self.treasures_found.append(Treasure(
            x=coords[0],
            y=coords[1],
            found_at=self.robot.getTime()
        ))
        
        print(f"\nTREASURE FOUND! ({self.treasure_count}/{self.REQUIRED_TREASURES})")
        print(f"Location: ({coords[0]:.3f}, {coords[1]:.3f})")
        print("Current treasure locations:")
        for i, t in enumerate(self.treasures_found, 1):
            print(f"Treasure {i}: ({t.x:.3f}, {t.y:.3f})")
            
    def check_treasure_completion(self): #finished? 
        if self.treasure_count >= self.REQUIRED_TREASURES:
            print("\nAll treasures found! Mission complete!")
            print("\nTreasure locations:")
            for i, treasure in enumerate(self.treasures_found, 1):
                print(f"Treasure {i}: ({treasure.x:.2f}, {treasure.y:.2f}) - Found at {treasure.found_at:.1f}s")
            return True
        return False
    
def main():
    controller = RobotController()
    step_count = 0
    
    print("Robot controller initialized")
    print("Starting main control loop...")
    
    while controller.robot.step(controller.TIME_STEP) != -1:
        step_count += 1
        position = controller.get_position()
        bearing = controller.get_bearing()
        controller.run_step()
        
        if controller.check_treasure_completion():
            break
        if controller.is_exploration_complete():
            print("\nExploration complete!")
            break
    
    print("\nSimulation ended")

if __name__ == "__main__":
    main()