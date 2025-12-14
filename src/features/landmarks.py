import mediapipe as mp
import cv2
import numpy as np

class FaceLandmarkExtractor:
    def __init__(self, static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            refine_landmarks=True # This adds iris points, total 478
        )

    def process_image(self, image):
        """
        Extracts landmarks from an image.
        Returns np.array of shape (478, 3) or None if no face found.
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(f"DEBUG: Processing Image for Landmarks. Shape: {img_rgb.shape}")
        results = self.face_mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
            # We assume 1 face per image for this dataset
            face = results.multi_face_landmarks[0]
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face.landmark])
            return landmarks
        return None

    def get_landmarks_pixel(self, landmarks, image_shape):
        """
        Convert normalized landmarks to pixel coordinates.
        """
        h, w = image_shape[:2]
        pixel_landmarks = landmarks.copy()
        pixel_landmarks[:, 0] *= w
        pixel_landmarks[:, 1] *= h
        return pixel_landmarks
