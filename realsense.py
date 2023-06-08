import pyrealsense2 as rs
import numpy as np


class RealsenseCapture:

    def __init__(self):
        self.WIDTH = 640
        self.HEGIHT = 480
        self.FPS = 30
        # Configure depth and color streams
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.WIDTH, self.HEGIHT, rs.format.bgr8, self.FPS)
        self.config.enable_stream(rs.stream.depth, self.WIDTH, self.HEGIHT, rs.format.z16, self.FPS)

    def start(self):
        # Start streaming
        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start(self.config)
        #カメラの内部パラメータを保存
        self.color_intr = rs.video_stream_profile(self.profile.get_stream(rs.stream.color)).get_intrinsics()
        print('pipline start')

    def read(self, is_array=True):
        # Flag capture available
        ret = True
        # get frames
        frames = self.pipeline.wait_for_frames()
        #ふれーむのみなので、色対応はしていない。
        # separate RGB and Depth image
        self.color_frame = frames.get_color_frame()  # RGB
        self.depth_frame = frames.get_depth_frame()  # Depth

        if not self.color_frame or not self.depth_frame:
            ret = False
            return ret, (None, None)
        elif is_array:
            # Convert images to numpy arrays
            color_image = np.array(self.color_frame.get_data())
            depth_image = np.array(self.depth_frame.get_data())
            return ret, (color_image, depth_image, self.color_intr)
        else:
            return ret, (self.color_frame, self.depth_frame, self.color_intr)

    def release(self):
        # Stop streaming
        self.pipeline.stop()

