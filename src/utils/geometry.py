import numpy as np

def calculate_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def get_face_metrics(landmarks):
    """
    Calculate anthropometric ratios from 468 landmarks.
    Landmarks should be (N, 3) numpy array.
    """
    # Key landmark indices (MediaPipe Face Mesh)
    # Trichion (top of forehead): 10
    # Menton (bottom of chin): 152
    # Left Zygion (cheekbone): 234
    # Right Zygion (cheekbone): 454
    # Left Gonion (jaw corner): 58
    # Right Gonion (jaw corner): 288
    
    trichion = landmarks[10]
    menton = landmarks[152]
    left_zygion = landmarks[234]
    right_zygion = landmarks[454]
    left_gonion = landmarks[58]
    right_gonion = landmarks[288]
    
    # FWHR (Width / Height)
    face_width = calculate_distance(left_zygion, right_zygion)
    face_height = calculate_distance(trichion, menton)
    fwhr = face_width / face_height
    
    # Jaw Width (Bigonial)
    jaw_width = calculate_distance(left_gonion, right_gonion)
    
    # Jaw to Face Width Ratio
    jaw_face_ratio = jaw_width / face_width
    
    # Chin Angle (approximate)
    # Using 148 (left chin), 152 (menton), 377 (right chin)
    # This is rough, simpler is just jaw width ratio for now.
    
    return {
        'face_width': face_width,
        'face_height': face_height,
        'jaw_width': jaw_width,
        'fwhr': fwhr,
        'jaw_face_ratio': jaw_face_ratio
    }

def classify_shape_heuristic(metrics):
    """
    Simple heuristic to classify face shape.
    Returns: 'Heart', 'Oblong', 'Oval', 'Round', 'Square'
    """
    fwhr = metrics['fwhr'] # >1 means Wide, <1 means Long (usually face is longer than wide, so < 0.75 is long)
    jfr = metrics['jaw_face_ratio']
    
    # Thresholds need calibration, but starting with standard estimates
    # Width vs Height
    # Square/Round are "Short/Wide" (Higher FWHR)
    # Oval/Oblong are "Long/Narrow" (Lower FWHR)
    
    # Jaw Width
    # Square/Oblong have wide jaws (Higher JFR)
    # Heart/Oval/Round have narrower jaws (Lower JFR)
    
    # Heart specific: Wide forehead, narrow chin (Low JFR, maybe high cheekbone width relative to jaw)
    
    # Preliminary thresholds (Need tuning)
    LONG_FACE_THRESH = 0.75 # Height is significantly larger than width
    WIDE_JAW_THRESH = 0.75  # Jaw is almost as wide as cheeks
    
    if fwhr > LONG_FACE_THRESH:
        # Short/Wide Faces: Round, Square, Heart
        if jfr > WIDE_JAW_THRESH:
            return "Square"
        else:
            # Round or Heart
            # Round is softer, Heart has narrower chin. 
            # Differentiating Round/Heart is hard without chin curve.
            # Let's assume lower JFR is Heart? Or middle is Round?
            if jfr < 0.65:
                return "Heart"
            else:
                return "Round"
    else:
        # Long Faces: Oval, Oblong
        if jfr > WIDE_JAW_THRESH:
            return "Oblong"
        else:
            return "Oval"
