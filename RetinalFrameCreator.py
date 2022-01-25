from tkinter import E
from cv2 import sumElems
import numpy as np
import scipy.io as sio
from utils_misc import rotation_matrix_from_vectors
import cv2
from pathlib import Path

from pydantic import BaseModel

class TweakedModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True

class VideoHandler(BaseModel):
    video_path: Path=None
    video_capture_obj: cv2.VideoCapture=None
    video_resolution_width: int=1920
    video_resolution_height:  int=1080

    def load_video_obj(self):
        self.video_capture_obj = cv2.VideoCapture(str(self.video_path))
        success, image = self.video_capture_obj.read()
        return success

    def get_next_frame(self):
        success, image = self.video_capture_obj.read()

        if success:
            return image
        else:
            return success


class RetinalFrameCreator(TweakedModel):
    video_handler: VideoHandler=None
    def test(self):
        print('i work')
        return True


def VideoFailedToLoadException(Exception):
    pass

if __name__ == '__main__':
    path_to_vid = Path('/home/karl/Desktop/test_vid.mp4')
    ret_frame_creator_obj = RetinalFrameCreator(video_path=path_to_vid)
    ret_frame_creator_obj.test()
    try:
        ret_frame_creator_obj.load_video_obj()
    except:
        raise VideoFailedToLoadException('Could not load video object')
    f=9
