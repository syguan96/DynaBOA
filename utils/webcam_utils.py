import cv2
import numpy as np
from threading import Thread

from render_demo import Renderer, convert_crop_cam_to_orig_img

try:
    import sys
    sys.path.append('/data/mesh_reconstruction/openpose/build/python')
    from openpose import pyopenpose as op
except:
    print('please build openpose first')


class WebcamVideoStream(object):
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        assert self.stream.isOpened(), 'Cannot capture source'
        
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
    
    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                print('Exist')
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.stream.release()


class OpenposeWarper():
    def __init__(self):
        params = dict()
        params["model_folder"] = "/data/mesh_reconstruction/openpose/models/"
        params["model_pose"] = "BODY_25"
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()
    
    def estimate(self, input_image):
        datum = op.Datum()
        datum.cvInputData = input_image
        self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        annoted_image = datum.cvOutputData
        kp2d = datum.poseKeypoints
        return kp2d, annoted_image


def render(vts, cam, image, bbox, color=[205,129,98]):
    vts = vts.clone().detach().cpu().numpy()
    cam = cam.clone().detach().cpu().numpy()
    # bbox = bbox.cpu().numpy()
    ori_h, ori_w = image.shape[:2]
    ori_pred_cams = convert_crop_cam_to_orig_img(cam, bbox, ori_w, ori_h)    
    renderer = Renderer(resolution=(ori_w, ori_h), orig_img=True, wireframe=False)
    rendered_image = renderer.render(image, vts[0], ori_pred_cams[0], color=np.array(color)/255)
    return rendered_image[:,:,::-1]



if __name__ == '__main__':
    from time import time
    import glob
    for imgpath in glob.glob('/data/douyin_video/images/002/*.png'):
        image = cv2.imread(imgpath)
        op_estimator = OpenposeWarper()
        start = time()
        kp2d, annoted_image = op_estimator.estimate(image)
        print(time() - start)
        cv2.imshow('openpose', annoted_image)
        cv2.waitKey(1)
    print(kp2d)