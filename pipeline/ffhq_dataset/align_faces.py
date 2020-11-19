import cv2
import dlib
from imutils.face_utils import FaceAligner


def rotate_face(src_path, dst_path):
	'''
    Function for de-rotating faces from tilted source images
    parameters: 'src_path' is the path to raw images; 'dst_path' is the path where de-rotated images are stored
    output: De-rotated face image at 1024x1024 resolution
    '''
	landmarks_model_path = '../../cache/shape_predictor_68_face_landmarks.dat'
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(landmarks_model_path)

	# Use FaceAligner with parameters 'desiredFaceWidth' to set the output image resolution
	#   and 'desiredLeftEye' to set the amount of zoom-out percentage
	fa = FaceAligner(predictor, desiredFaceWidth=1024, desiredLeftEye=(0.40, 0.40))

	# load the input image and convert it to grayscale
	image = cv2.imread(src_path)
	gray = cv2.imread(src_path, 0)
	# Detect faces in the grayscale image
	rects = detector(gray)

	# loop over the detected faces
	for rect in rects:
		# Align the image and save it in PNG format
		faceAligned = fa.align(image, gray, rect)
		cv2.imwrite(dst_path, faceAligned)