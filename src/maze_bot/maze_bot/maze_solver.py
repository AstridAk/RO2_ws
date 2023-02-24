
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2

from .bot_localization import bot_localizer
from .bot_mapping import bot_mapper
from .bot_pathplanning import bot_pathplanner
from .bot_motionplanning import bot_motionplanner


from nav_msgs.msg import Odometry

import numpy as np

class maze_solver(Node):

    def __init__(self):
        
        super().__init__("maze_solving_node")
        
        self.velocity_publisher = self.create_publisher(Twist,'/cmd_vel',10)
        self.videofeed_subscriber = self.create_subscription(Image,'/upper_camera/image_raw',self.get_video_feed_cb,10)

        self.bot_subscriber = self.create_subscription(Image,'/Botcamera/image_raw',self.process_data_bot,10)
     
   
        self.timer = self.create_timer(0.2, self.maze_solving)
        self.bridge = CvBridge()
        self.vel_msg = Twist()
        
        # Creating objects for each stage of the robot navigation
        self.find_robot = bot_localizer()
        self.mapper = bot_mapper()
        self.bot_pathplanner = bot_pathplanner()
        self.motion_planning = bot_motionplanner()
 

        self.pose_subscriber = self.create_subscription(Odometry,'/odom',self.motion_planning.get_pose,10)
        self.sat_view = np.zeros((100,100))

    def get_video_feed_cb(self,data):
        frame = self.bridge.imgmsg_to_cv2(data,'bgr8')
        self.sat_view = frame
    
    def process_data_bot(self, data):
      self.bot_view = self.bridge.imgmsg_to_cv2(data,'bgr8') # performing conversion

    def maze_solving(self):

        # Creating frame to display current robot state to user        
        frame_disp = self.sat_view.copy()
        
        # [Stage 1: Localization] Localizing robot at each iteration        
        self.find_robot.localize_bot(self.sat_view, frame_disp)
      

        # [Stage 2: Mapping] Converting Image to Graph
        self.mapper.graphify(self.find_robot.maze_og)

        # [Stage 3: PathPlanning] Using {User Specified PathPlanner} to find path to goal        
        start = self.mapper.Graph.start
        end = self.mapper.Graph.end
        maze = self.mapper.maze

       
        self.bot_pathplanner.find_path_nd_display(self.mapper.Graph.graph, start, end, maze,method="dijisktra")

    
        self.bot_pathplanner.find_path_nd_display(self.mapper.Graph.graph, start, end, maze,method="a_star")
        print("\nNodes Visited [Dijisktra V A-Star*] = [ {} V {} ]".format(self.bot_pathplanner.dijisktra.dijiktra_nodes_visited,self.bot_pathplanner.astar.astar_nodes_visited))
        #cv2.waitKey(0)
  
        #[Stage 4: Moving the robot] 
        bot_loc = self.find_robot.loc_car
        path = self.bot_pathplanner.path_to_goal
        self.motion_planning.nav_path(bot_loc, path, self.vel_msg, self.velocity_publisher)

        #Display everything 
        img_shortest_path = self.bot_pathplanner.img_shortest_path
        self.motion_planning.display_control_mechanism_in_action(bot_loc, path, img_shortest_path, self.find_robot, frame_disp)


        # View bot view on left to frame Display
        bot_view = cv2.resize(self.bot_view, (int(frame_disp.shape[0]/2), int(frame_disp.shape[1]/2)))
        frame_disp[0:bot_view.shape[0], 0:bot_view.shape[1]] = bot_view
        frame_disp[0:img_shortest_path.shape[0], frame_disp.shape[1]-img_shortest_path.shape[1]:frame_disp.shape[1]] = img_shortest_path
        cv2.imshow("Maze (Live)", frame_disp) 
        cv2.waitKey(1)

def main(args =None):
    rclpy.init()
    node_obj =maze_solver()
    rclpy.spin(node_obj)
    rclpy.shutdown()


if __name__ == '__main__':
    main()