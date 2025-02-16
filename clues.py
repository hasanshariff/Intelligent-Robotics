from controller import Robot
import math
import time
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
    FOLLOWING_CLUE = "following_clue"

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
class Clue:
    x: float
    y: float
    color: str
    
@dataclass
class Treasure:
    x: float
    y: float
    found_at: float

class GridState(Enum):
    UNEXPLORED = 0
    OBSTACLE = 2
    ROBOT = 3
    TREASURE = 4
    CLUE = 5
    
@dataclass
class Pose:
    x: float
    y: float
    theta: float
    
@dataclass
class OdometryState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

class Odometry: #calculating odometry 
    def __init__(self):
        self.wheel_radius = 0.025
        self.distance_between_wheels = 0.09
        self.wheel_circum = 2 * math.pi * self.wheel_radius
        self.encoder_unit = self.wheel_circum/6.28
        self.robot_pose = [0, 0, 0]  # x, y, theta
        self.last_ps_values = [0, 0, 0, 0]
        
    def update_position(self): #update current positon
        gps_pos = self.get_position()
        compass_bearing = self.get_bearing()
        
        # Get all encoder values
        encoder_values = [sensor.getValue() for sensor in self.position_sensors]
        
        # Update odometry and retrieve the robot pose as [x, y, theta]
        odom_state = self.odometry.update(encoder_values, self.TIME_STEP / 1000.0)
        
        # Sensor fusion with configurable weights
        alpha = 0.7  # Weight for GPS for fusiion
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
        
        return Pose(x=self.robot_pose[0], y=self.robot_pose[1], theta=self.robot_pose[2])
    

class RobotController:
    def __init__(self):
        
        self.TIME_STEP = 64
        self.TURN_SPEED_SLOW = 0.5    
        self.OBSTACLE_THRESHOLD = 950.0    
        self.FINE_ANGLE_THRESHOLD = 10.0  
        self.MIN_COORD = -2.10  
        self.MAX_COORD = 2.10
        self.GRID_SIZE = 42 
        self.CELL_SIZE = 0.1
        # Obstacle avoidance 
        self.OBSTACLE_TURN = 90.0  #degrees to turn
        self.avoid_obstacle_counter = 0
        self.FORWARD_STEPS = 0  
        self.avoidance_phase = "turn" 
        self.AVOID_OBSTACLE_STEPS = 100
        self.turning_left = True
        
        self.robot = Robot()
        self.setup_robot()
        
        self.odometry = Odometry()
    
        self.current_bearing = 0.0  # Start facing North
        self.target_bearing = 0.0
        self.TURN_STEPS = 15
        self.TURN_SPEED_FAST = 5.0
        self.BASE_SPEED = 10.0
        self.target_bearing = 0.0   # For tracking desired bearing during turns
        self.ANGLE_THRESHOLD = 2.0  # Degrees of acceptable error
        
        self.NORTH_RANGE = (-45, 45)   
        self.EAST_RANGE = (45, 135)   
        self.SOUTH_RANGE = (135, 225) 
        self.WEST_RANGE = (-135, -45)
        
        self.treasures_found = []
        self.treasure_count = 0
        self.detection_cooldown = 0 
        self.REQUIRED_TREASURES = 5
        
        self.camera_width = self.camera.getWidth()
        self.camera_height = self.camera.getHeight()
        self.camera_fov = self.camera.getFov()
        
        self.RECOGNITION_COLORS = {
            'green': [0, 1, 0],    # 0 1 0 for green
            'white': [1, 1, 1],    # 1 1 1 for white
            'blue': [0, 0, 1]      # 0 0 1 for blue treasure
        }

        self.COLOR_THRESHOLD = 0.1
        self.clue_state = None
        self.clue_turn_remaining = 0
        self.clue_turn_direction = 0
        self.clues_found = []
        self.MIN_TREASURE_AREA = 50
        self.detection_cooldown = 0
        self.DETECTION_COOLDOWN_STEPS = 20
        self.grid = [[GridState.UNEXPLORED.value] * self.GRID_SIZE 
                     for _ in range(self.GRID_SIZE)]
    
    def setup_robot(self): #set up robot
        self.setup_sensors()
        self.setup_wheels()
        
    def setup_encoders(self): #set up encoders 
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
    
    def update_position(self): #updating the postion of the robot
        gps_pos = self.get_position()
        compass_bearing = self.get_bearing()
        
        # Get all encoder values
        encoder_values = [sensor.getValue() for sensor in self.position_sensors]
        odom_state = self.odometry.update(encoder_values, self.TIME_STEP/1000.0)
        
        # Sensor fusion with configurable weights
        alpha = 0.7  # Weight for GPS
        fused_x = alpha * gps_pos[0] + (1-alpha) * odom_state.x
        fused_y = alpha * gps_pos[1] + (1-alpha) * odom_state.y
        
        self.current_position = (fused_x, fused_y)
        self.current_bearing = compass_bearing
        
        return self.current_position, self.current_bearing
    
    def print_status(self): #printed in the console
        # Fetch the GPS position
        gps_pos = self.get_position() 
        
        # Encoder values for debugging
        encoder_values = [f"{ps.getValue():.2f}" for ps in self.position_sensors]
        
        print(f"\n=== Robot Status ===")
        print(f"GPS Position: (x={gps_pos[0]:.2f}, y={gps_pos[1]:.2f})") 
        print(f"Odometry Position: (x={gps_pos[0]:.2f}, y={gps_pos[1]:.2f})")
        print(f"Encoder Values: {encoder_values}")
        print(f"Treasures found: {self.treasure_count}/5")

    def world_to_grid(self, x: float, y: float) -> GridCell: #world to grid co-ordinates 
        grid_x = int((x - self.MIN_COORD) / self.CELL_SIZE)
        grid_y = int((y - self.MIN_COORD) / self.CELL_SIZE)
        grid_x = max(0, min(grid_x, self.GRID_SIZE - 1)) #make sure co-ordinates are within the grid
        grid_y = max(0, min(grid_y, self.GRID_SIZE - 1))
        return GridCell(x=grid_x, y=grid_y)
        
    def detect_obstacles(self): #only detect obstacles
        if self.avoid_obstacle_counter > 0:
            return False
        if self.is_seeing_clue(): #check for clue
            return False
        for sensor in self.ds: #check the sensors
            value = sensor.getValue()
            if value < self.OBSTACLE_THRESHOLD:
                robot_pos = self.get_position()
                bearing = math.radians(self.get_bearing())
                dist = value / 1000.0  # calculate distance in meters
                x = robot_pos[0] + dist * math.cos(bearing)
                y = robot_pos[1] + dist * math.sin(bearing)
                grid_cell = self.world_to_grid(x, y)
                if self.grid[grid_cell.y][grid_cell.x] == GridState.TREASURE.value:
                    return False
                    
                print("\nObstacle detected - Starting avoidance maneuver")
                self.avoid_obstacle_counter = self.AVOID_OBSTACLE_STEPS
                return True
        return False

    def is_seeing_clue(self): #can the robot see a clue?
        image = self.camera.getImage()
        if not image:
            return False
            
        frame = np.frombuffer(image, np.uint8).reshape((self.camera_height, self.camera_width, 4))
        frame = frame[:, :, :3] / 255.0
        center_y = self.camera_height // 2
        center_x = self.camera_width // 2
        detection_region = frame[
            center_y - 10:center_y + 10,
            center_x - 10:center_x + 10
        ]
        
        center_color = detection_region[10, 10]  # Center pixel
        print(f"Detected color values: R={center_color[0]:.2f}, G={center_color[1]:.2f}, B={center_color[2]:.2f}")
        for color_name, target_color in self.RECOGNITION_COLORS.items():
            if color_name != 'blue':  # only want clue colours
                color_mask = np.all(
                    abs(detection_region - target_color) < self.COLOR_THRESHOLD,
                    axis=2
                )
                if np.sum(color_mask) > 50:
                    print(f"Detected {color_name} clue")
                    return True
        return False
            
    def normalize_angle(self, angle): #evaluate angle
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
        
    def get_bearing(self) -> float: #get current bearing
        north = self.compass.getValues()
        rad = math.atan2(north[0], north[2])
        bearing = math.degrees(rad)
        if bearing < 0:
            bearing += 360
        bearing = (bearing + 90) % 360
        return bearing  
            
    def calculate_angle_difference(self, current_bearing: float, target_bearing: float) -> float:
        diff = target_bearing - current_bearing
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        return (abs(diff), diff>0)
              
    def get_position(self) -> Tuple[float, float]: #retrive the positon of the robot
        gps_values = self.gps.getValues()
        x = round(gps_values[0], 3)
        y = round(gps_values[1], 3)
        return (x, y)
        
    def get_cardinal_direction(self) -> str: #what direction are we facing
        bearing = self.get_bearing()
        if bearing > 180:
            bearing -= 360
        if self.NORTH_RANGE[0] <= bearing <= self.NORTH_RANGE[1]:
            return "NORTH"
        elif self.EAST_RANGE[0] <= bearing <= self.EAST_RANGE[1]:
            return "EAST"
        elif self.SOUTH_RANGE[0] <= bearing <= self.SOUTH_RANGE[1]:
            return "SOUTH"
        else:
            return "WEST"

    def calculate_turn_for_clue(self, clue_color: str) -> Tuple[float, int, str]: #calculate the turn required
        current_direction = self.get_cardinal_direction()
        print(f"Processing {clue_color} clue while facing {current_direction}")
        direction_sequence = ["NORTH", "EAST", "SOUTH", "WEST"]
        current_index = direction_sequence.index(current_direction)
        if clue_color == 'white': #white = left turn 
            target_index = (current_index - 1) % 4
            target_direction = direction_sequence[target_index]
            return (-1, self.CLUE_TURN_STEPS, target_direction)
        elif clue_color == 'green': # green = right turn
            target_index = (current_index + 1) % 4
            target_direction = direction_sequence[target_index]
            return (1, self.CLUE_TURN_STEPS, target_direction)
        return (0, 0, current_direction)

    def handle_clue(self, color): #following instructions from the clue
        current_pos = self.get_position()
        self.current_bearing = self.get_bearing()  # Update current bearing
        
        print(f"\n=== Clue Response ===") #clue debug
        print(f"Current Position: (x={current_pos[0]:.2f}, y={current_pos[1]:.2f})")
        print(f"Current Bearing: {self.current_bearing:.1f}° ({self.get_cardinal_direction()})")
        
        self.clue_state = color
        if color == 'white': # left 90 degrees
            self.target_bearing = (self.current_bearing - 90) % 360
            self.clue_turn_remaining = self.TURN_STEPS
            self.clue_turn_direction = -1
            print(f"White Clue: Turning LEFT 90°")
            print(f"Target Bearing: {self.target_bearing:.1f}°")
                
        elif color == 'green': # right 90 degrees
            self.target_bearing = (self.current_bearing + 90) % 360
            self.clue_turn_remaining = self.TURN_STEPS
            self.clue_turn_direction = 1  
            print(f"Green Clue: Turning RIGHT 90°")
            print(f"Current Bearing: {self.current_bearing:.1f}°")
            print(f"Target Bearing: {self.target_bearing:.1f}°")
            
    def is_color_match(self, pixel_color, target_color):
        return all(abs(p - t) < self.COLOR_THRESHOLD 
                  for p, t in zip(pixel_color, target_color))
    
    def is_near_found_clue(self, current_pos, color=None): #check if we are near an already found clue so we dont find it twice.
        for clue in self.clues_found:
            distance = math.sqrt((current_pos[0] - clue.x)**2 + (current_pos[1] - clue.y)**2)
            if color and clue.color != color: #only check clues of that colour
                continue
            if distance < 0.5:  # better duplicate prevention
                print(f"Already found {clue.color} clue nearby at ({clue.x:.2f}, {clue.y:.2f})")
                return True
        return False
 
    def detect_clues(self): #find clues
        if self.clue_state is not None:
            return False
                
        recognized_objects = self.camera.getRecognitionObjects()
        current_pos = self.get_position()
        for obj in recognized_objects:
            colors = obj.getColors()
            def handle_clue_detection(color): #add new clue
                if not self.is_near_found_clue(current_pos, color):
                    new_clue = Clue(x=current_pos[0], y=current_pos[1], color=color)
                    self.clues_found.append(new_clue)
                    print(f"\nFound NEW {color} clue at ({current_pos[0]:.2f}, {current_pos[1]:.2f})")
                    self.handle_clue(color)
                    return True
                return False
            
            if abs(colors[1] - 1.0) < 0.1 and colors[0] < 0.1 and colors[2] < 0.1: #green clue logic
                return handle_clue_detection('green')
                    
            elif abs(colors[0] - 1.0) < 0.1 and abs(colors[1] - 1.0) < 0.1 and abs(colors[2] - 1.0) < 0.1: #white clue logic
                return handle_clue_detection('white')
                    
        return False
           
    def execute_movement(self): #this is how the robot moves
        current_pos = self.get_position()
        self.current_bearing = self.get_bearing()
            
        if (self.clue_state == 'green' or self.clue_state == 'white') and self.clue_turn_remaining > 0: #handle turn
            self.clue_turn_remaining -= 1
            print(f"During turn - Current: {self.current_bearing:.1f}° ({self.get_cardinal_direction()})")
            if self.clue_turn_remaining <= 0:
                print(f"Turn completed: Now at {self.current_bearing:.1f}° ({self.get_cardinal_direction()})")
                self.clue_state = None
                return (self.BASE_SPEED, self.BASE_SPEED)
    
            if self.clue_state == 'green':
                return (self.TURN_SPEED_FAST, -self.TURN_SPEED_FAST) #right turn
            elif self.clue_state == 'white':
                return (-self.TURN_SPEED_FAST, self.TURN_SPEED_FAST) #left turn
    
        if self.detect_clues(): #find new clue
            return (0.0, 0.0)
    
        if self.avoid_obstacle_counter > 0: #avoid obstacle eg wall
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
                    nearest_clue = None #navigate to nearest clue
                    min_distance = float('inf')
                    for clue in self.clues_found:
                        dist = math.sqrt((current_pos[0] - clue.x) ** 2 + (current_pos[1] - clue.y) ** 2)
                        if dist < min_distance:
                            min_distance = dist
                            nearest_clue = clue
                    if nearest_clue:
                        print(f"Found nearest clue at ({nearest_clue.x}, {nearest_clue.y})")
                        dx = nearest_clue.x - current_pos[0]
                        dy = nearest_clue.y - current_pos[1]
                        target_bearing = math.degrees(math.atan2(dy, dx))
                        self.target_bearing = target_bearing
                return (self.BASE_SPEED, self.BASE_SPEED)
    
        if self.detect_obstacles():
            print("New obstacle detected! Starting avoidance maneuver")
            self.avoid_obstacle_counter = self.TURN_STEPS
            self.avoidance_phase = "turn"
            return (self.BASE_SPEED, -self.BASE_SPEED)
    
        return (self.BASE_SPEED, self.BASE_SPEED)
    
    def print_grid(self): #print the occupancy grid
        print("\nOccupancy Grid (5m x 5m, centered at 0,0):")
        print("   ", end="")
        for x in range(-21, 25, 5): #x axis
            print(f"{x/10:6.1f}", end="")
        print("\n   " + "-" * (10 * 6 - 1))
        for y in range(21, -25, -5): #y axis
            print(f"{y/10:4.1f}|", end="")
            for x in range(-21, 25, 5):
                world_x = x/10
                world_y = y/10
                clue_here = False #check for clue
                for clue in self.clues_found:
                    if abs(clue.x - world_x) < 0.25 and abs(clue.y - world_y) < 0.25:
                        clue_here = True
                        break
                        
                treasure_here = False #check for treausre
                for treasure in self.treasures_found:
                    if abs(treasure.x - world_x) < 0.25 and abs(treasure.y - world_y) < 0.25:
                        treasure_here = True
                        break
                
                if treasure_here:
                    print("  T   ", end="")
                elif clue_here:
                    print("  C   ", end="")
                else:
                    print("  *   ", end="")
            print("  |")
        
    def set_wheel_speeds(self, left_speed: float, right_speed: float): 
        for i, wheel in enumerate(self.wheels):
            wheel.setVelocity(left_speed if i % 2 == 0 else right_speed)
        
    def run_step(self): #exectuing 
        self.update_position() #update positon
        self.print_status()
        clue_detected = self.detect_clues() #check for clue
        if not clue_detected: #no clue so check for treasure
            self.detect_and_record_treasure()
        left_speed, right_speed = self.execute_movement() #execute movement
        for i in range(4): #wheel speeds 
            if i % 2 == 0:
                self.wheels[i].setVelocity(left_speed)
            else:
                self.wheels[i].setVelocity(right_speed)
        
        self.print_grid() #print occupancy grid
        return self.check_treasure_completion() 
        
#TREASURE LOGIC
    def stop_robot(self):
        for wheel in self.wheels:
            wheel.setVelocity(0.0)

    def round_to_nearest_half(self, value): #round values
        return round(value * 2) / 2
    
    def detect_and_record_treasure(self): #use blue color to check for treasure
        if self.detection_cooldown > 0:
            self.detection_cooldown -= 1
            return False
        recognized_objects = self.camera.getRecognitionObjects()
        for obj in recognized_objects:
            colors = obj.getColors()
            current_pos = self.get_position()
            robot_bearing = self.get_bearing()
            if abs(colors[2] - 1.0) < 0.1 and colors[0] < 0.1 and colors[1] < 0.1: #check for blue
                position_info = obj.getPosition() 
                treasure_x = self.round_to_nearest_half(current_pos[0]) #calculate treasure info
                treasure_y = self.round_to_nearest_half(current_pos[1])
                if treasure_y > 0:
                    treasure_y = math.ceil(current_pos[1])
                
                # Debug output
                print("\nTreasure Detection:")
                print(f"Robot position: ({current_pos[0]:.3f}, {current_pos[1]:.3f})")
                print(f"Calculated treasure position: ({treasure_x:.3f}, {treasure_y:.3f})")
                treasure_exists = False #check if we have already found the treasure
                for existing in self.treasures_found:
                    if abs(existing.x - treasure_x) < 0.1 and abs(existing.y - treasure_y) < 0.1:
                        print(f"Already found treasure at ({treasure_x:.3f}, {treasure_y:.3f})")
                        treasure_exists = True
                        break
                        
                if not treasure_exists:
                    self.treasure_count += 1 #record new treasure
                    new_treasure = Treasure(
                        x=treasure_x,
                        y=treasure_y,
                        found_at=self.robot.getTime())
                        
                    self.treasures_found.append(new_treasure)
                    grid_cell = self.world_to_grid(treasure_x, treasure_y) #update grid with new treasure
                    self.grid[grid_cell.y][grid_cell.x] = GridState.TREASURE.value
                    
                    print(f"\nNEW TREASURE FOUND! ({self.treasure_count}/{self.REQUIRED_TREASURES})")
                    print(f"Location: ({treasure_x:.3f}, {treasure_y:.3f})")
                    print("All treasure locations:")
                    for i, t in enumerate(self.treasures_found, 1):
                        print(f"Treasure {i}: ({t.x:.3f}, {t.y:.3f})")
                    self.detection_cooldown = self.DETECTION_COOLDOWN_STEPS
                    return True
        return False
            
    def check_treasure_completion(self): #have we found all of the treasure
        if self.treasure_count >= self.REQUIRED_TREASURES:
            print("\nAll treasures found! Mission complete!")
            print("\nTreasure locations:")
            for i, treasure in enumerate(self.treasures_found, 1):
                print(f"Treasure {i}: ({treasure.x:.2f}, {treasure.y:.2f}) - Found at {treasure.found_at:.1f}s")
            return True
        return False
    
def main(): 
    controller = RobotController()
    print("Robot controller initialized")
    print("Starting main control loop...")
    while controller.robot.step(controller.TIME_STEP) != -1:
        if controller.run_step():  # print if all 5 are found
            print("\nMission Complete! All treasures found!")
            break
    controller.stop_robot() #wheel velocity set to 0 so robot stops
    print("\nSimulation ended")

if __name__ == "__main__":
    main()