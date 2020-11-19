"""
    Main script pertaining to Step-1 (i.e. Pre-processing) of the pipeline.
    Usage:  python align_images.py $PATH_TO_SOURCE_DIRECTORY$ $PATH_FOR_DESTINATION_DIRECTORY$
"""
import os
import argparse

from ffhq_dataset.landmarks_detector import LandmarksDetector
from ffhq_dataset.face_alignment import image_align


def main():
    '''
    Detect face(s), extract landmarks, and align face(s) from the given raw image(s)
    parameters: None
    output: Aligned face image at 1024x1024 resolution
    '''
    parser = argparse.ArgumentParser(description='Extract and align faces from the raw images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('src_dir', help='Source directory with raw images')
    parser.add_argument('dest_dir', help='Destination directory for storing aligned images')

    args, other_args = parser.parse_known_args()

    # Specify paths to the saved 68-Facial Landmark model, raw images directory, and aligned images directory
    landmarks_model_path = '../cache/shape_predictor_68_face_landmarks.dat'
    RAW_IMAGES_DIR = os.path.join('../', args.src_dir)
    ALIGNED_IMAGES_DIR = os.path.join('../', args.dest_dir)
    os.makedirs(ALIGNED_IMAGES_DIR, exist_ok=True)

    # Try detecting faces and extracting landmarks; throw an error, otherwise
    landmarks_detector = LandmarksDetector(landmarks_model_path)
    for img_name in os.listdir(RAW_IMAGES_DIR):
        print('Aligning %s ...' % img_name)
        try:
            raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
            print('Getting landmarks...')
            face_landmarks = landmarks_detector.get_landmarks(raw_img_path)
            try:
                print('Starting face alignment...')
                aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, img_name)
                image_align(raw_img_path, aligned_face_path, face_landmarks)
                print('Wrote result at %s' % aligned_face_path)
            except:
                print("\n\tException in face alignment!\n")
        except ValueError:
            print("\n\tEither failed to detect landmarks or NO faces present!\n")



if __name__ == "__main__":
    main()