import numpy as np
import os
import math
import scipy
import scipy.ndimage
import PIL.Image
from imutils.face_utils.helpers import FACIAL_LANDMARKS_IDXS

from align_faces import rotate_face
from landmarks_detector import LandmarksDetector


def image_align(src_file, dst_file, face_landmarks, output_size=1024, transform_size=4096, enable_padding=True):
        '''
        Function that applies all the data pre-processing steps proposed in StyleGAN-FFHQ
        parameters: 'src_file' indicates the path to raw images; 'dst_file' indicates the path to store the aligned images; 
            'face_landmarks' specify the (x, y) coordinates of the 68-face landmarks; 'output_size' specifies the resolution of the aligned image; 
            'transform_size' indicates the target up-scaling resolution; 'enable_padding' pads the image if up-scaled
        output: Aligned face image at 1024x1024 resolution
        '''
        # Align function from FFHQ dataset pre-processing step
        # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

        lm = np.array(face_landmarks)
        lm_chin          = lm[0  : 17]  # left-right
        lm_eyebrow_left  = lm[17 : 22]  # left-right
        lm_eyebrow_right = lm[22 : 27]  # left-right
        lm_nose          = lm[27 : 31]  # top-down
        lm_nostrils      = lm[31 : 36]  # top-down
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)

        ### Check if the face is tilted and if true, perform de-rotation
        # Compute the center of mass for each eye
        leftEyeCenter = np.mean(lm_eye_left, axis=0).astype("int")
        rightEyeCenter = np.mean(lm_eye_right, axis=0).astype("int")
        # Compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = math.floor(abs(np.degrees(np.arctan2(dY, dX))))
        print("\n\t\tThe angle is: ", angle, "\n")

        is_rotated = False
        if angle > 11:
            is_rotated = True
        ###

        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        if is_rotated:
            ROTATED_IMAGES_DIR = '../../rotated_images/'
            landmarks_model_path = '../../cache/shape_predictor_68_face_landmarks.dat'
            os.makedirs(ROTATED_IMAGES_DIR, exist_ok=True)
            rot_file = str(os.path.splitext(src_file)[0]).split('/')[-1] + str(os.path.splitext(src_file)[1])
            rot_file_path = os.path.join(ROTATED_IMAGES_DIR, rot_file)
            rotate_face(src_file, rot_file_path)

            landmarks_detector = LandmarksDetector(landmarks_model_path)
            face_landmarks = landmarks_detector.get_landmarks(rot_file_path)
            lm = np.array(face_landmarks)
            lm_chin          = lm[0  : 17]  # left-right
            lm_eyebrow_left  = lm[17 : 22]  # left-right
            lm_eyebrow_right = lm[22 : 27]  # left-right
            lm_nose          = lm[27 : 31]  # top-down
            lm_nostrils      = lm[31 : 36]  # top-down
            lm_eye_left      = lm[36 : 42]  # left-clockwise
            lm_eye_right     = lm[42 : 48]  # left-clockwise
            lm_mouth_outer   = lm[48 : 60]  # left-clockwise
            lm_mouth_inner   = lm[60 : 68]  # left-clockwise

            # Calculate auxiliary vectors.
            eye_left     = np.mean(lm_eye_left, axis=0)
            eye_right    = np.mean(lm_eye_right, axis=0)

            eye_avg      = (eye_left + eye_right) * 0.5
            eye_to_eye   = eye_right - eye_left
            mouth_left   = lm_mouth_outer[0]
            mouth_right  = lm_mouth_outer[6]
            mouth_avg    = (mouth_left + mouth_right) * 0.5
            eye_to_mouth = mouth_avg - eye_avg

            # Choose oriented crop rectangle.
            x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
            x /= np.hypot(*x)
            x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
            y = np.flipud(x) * [-1, 1]
            c = eye_avg + eye_to_mouth * 0.1
            quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
            qsize = np.hypot(*x) * 2

            img = PIL.Image.open(rot_file_path)

            # Shrink.
            shrink = int(np.floor(qsize / output_size * 0.5))
            if shrink > 1:
                rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
                img = img.resize(rsize, PIL.Image.ANTIALIAS)
                quad /= shrink
                qsize /= shrink

            # Crop.
            border = max(int(np.rint(qsize * 0.1)), 3)
            crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
            crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
            if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
                img = img.crop(crop)
                quad -= crop[0:2]

            # Pad.
            pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
            pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
            if enable_padding and max(pad) > border - 4:
                pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
                img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
                h, w, _ = img.shape
                y, x, _ = np.ogrid[:h, :w, :1]
                mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
                blur = qsize * 0.02
                img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
                img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
                #img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
                img = np.uint8(np.clip(np.rint(img), 0, 255))
                img = PIL.Image.fromarray(img, 'RGB')
                quad += pad[:2]

            img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
            if output_size < transform_size:
                img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
        else:
            img = PIL.Image.open(src_file)
            
            # Shrink.
            shrink = int(np.floor(qsize / output_size * 0.5))
            if shrink > 1:
                rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
                img = img.resize(rsize, PIL.Image.ANTIALIAS)
                quad /= shrink
                qsize /= shrink

            # Crop.
            border = max(int(np.rint(qsize * 0.1)), 3)
            crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
            crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
            if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
                img = img.crop(crop)
                quad -= crop[0:2]

            # Pad.
            pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
            pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
            if enable_padding and max(pad) > border - 4:
                pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
                img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
                h, w, _ = img.shape
                y, x, _ = np.ogrid[:h, :w, :1]
                mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
                blur = qsize * 0.02
                img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
                img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
                #img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
                img = np.uint8(np.clip(np.rint(img), 0, 255))
                img = PIL.Image.fromarray(img, 'RGB')
                quad += pad[:2]

            # Transform.
            img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
            if output_size < transform_size:
                img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        # Save aligned image.
        img.save(dst_file, 'PNG')