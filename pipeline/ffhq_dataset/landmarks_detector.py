import numpy as np
import cv2
import dlib
from imutils import face_utils
from imutils.face_utils import rect_to_bb


class LandmarksDetector:
    def __init__(self, predictor_model_path):
        """
        :param 'predictor_model_path': path to the 68-facial landmarks file
        """
        weights = '../../cache/mmod_human_face_detector.dat'   # For CNN
        #self.detector = dlib.get_frontal_face_detector() ### For HOG-SVM
        self.detector = dlib.cnn_face_detection_model_v1(weights)   # For CNN
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image_path):
        '''
        Function for extracting 68-face landmarks
        parameters: 'image_path' specifies the path to source images
        output: Face landmarks of the detected face with the MAXIMUM bounding box area
        '''
        img = cv2.imread(image_path, 0)
        dets = self.detector(img)

        areas = []
        face_landmarks = []
        for detection in dets:
            #face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()] ### For HOG-SVM
            detection = detection.rect   # For CNN
            (x, y, w, h) = rect_to_bb(detection)
            areas.append(w * h)
            face_landmarks.append(face_utils.shape_to_np(self.shape_predictor(img, detection)))   # For CNN
        areas = np.array(areas)
        # Find the bounding box with the maximum area and return its landmarks
        max_area_index = np.argmax(areas)
        return face_landmarks[max_area_index]