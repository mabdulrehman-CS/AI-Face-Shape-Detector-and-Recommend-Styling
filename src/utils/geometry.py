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
    
    # Jaw Angle (Gonial Angle) - Crucial for Square vs Round
    # Left: Ear(234) -> JawCorner(58) -> Chin(152)
    # This is an approximation using zygion as the upper point
    def get_angle(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle

    # Forehead Width (Temp/Frontalis metrics)
    # 103 (Left Forehead), 332 (Right Forehead)
    left_forehead = landmarks[103]
    right_forehead = landmarks[332]
    forehead_width = calculate_distance(left_forehead, right_forehead)
    
    # Chin Angle (Sharpness) - Crucial for Heart (Cornered Downward) vs Round
    # Angle at Menton(152) formed by Gonions(58, 288)
    # Sharp (< 90-100) = Heart/Oval (sometimes). Flat/Wide (> 110) = Square/Round/Oblong.
    chin_angle_val = get_angle(left_gonion, menton, right_gonion) # This calculates angle at gonion. wait.
    # We want angle at menton. vector menton->left_gonion and menton->right_gonion.
    
    def get_angle_at_middle(p1, mid, p2):
        v1 = p1 - mid
        v2 = p2 - mid
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle
        
    chin_angle = get_angle_at_middle(left_gonion, menton, right_gonion)

    chin_angle = get_angle_at_middle(left_gonion, menton, right_gonion)

    angle_left = get_angle(left_zygion, left_gonion, menton)
    angle_right = get_angle(right_zygion, right_gonion, menton)
    avg_jaw_angle = (angle_left + angle_right) / 2.0
    
    # Face Solidity (Area vs Bounding Box) - "Usage of entire size"
    # Square fills the corners (High Solidity). Round/Oval cuts corners (Low Solidity).
    # Polygon: Top(10) -> R_Fore(338) -> R_Cheek(454) -> R_Jaw(361) -> Chin(152) -> L_Jaw(132) -> L_Cheek(234) -> L_Fore(109)
    # Using specific indices for a robust contour
    poly_indices = [10, 338, 454, 361, 152, 132, 234, 109]
    poly_points = []
    for idx in poly_indices:
        poly_points.append(landmarks[idx])
    
    # Shoelace Formula for Area
    def polygon_area(points):
        area = 0.0
        j = len(points) - 1
        for i in range(len(points)):
             area += (points[j][0] + points[i][0]) * (points[j][1] - points[i][1])
             j = i
        return abs(area / 2.0)
        
    face_area = polygon_area(poly_points)
    bbox_area = face_width * face_height
    solidity = face_area / bbox_area if bbox_area > 0 else 0
    
    return {
        'face_width': face_width,
        'face_height': face_height,
        'jaw_width': jaw_width,
        'forehead_width': forehead_width,
        'fwhr': fwhr,
        'jaw_face_ratio': jaw_face_ratio,
        'jaw_angle': avg_jaw_angle,
        'chin_angle': chin_angle,
        'solidity': solidity
    }

def classify_shape_heuristic(metrics):
    """
    Simple heuristic to classify face shape.
    Returns: 'Heart', 'Oblong', 'Oval', 'Round', 'Square'
    """
    fwhr = metrics['fwhr'] 
    jfr = metrics['jaw_face_ratio']
    angle = metrics['jaw_angle']
    chin_angle = metrics['chin_angle']
    forehead_to_jaw = metrics['forehead_width'] / metrics['jaw_width']
    
    # Thresholds (Visual Definitions)
    LONG_FACE_THRESH = 0.80 
    WIDE_JAW_THRESH = 0.78  
    SQUARE_JAW_ANGLE_THRESH = 130.0 
    SHARP_CHIN_THRESH = 115.0 # < 115 is "Cornered Downward" (Heart/Diamond)
    
    if fwhr > LONG_FACE_THRESH:
        # Short/Wide Faces: Round, Square, Heart
        
        # Check Heart (Pointy Chin + Wide Forehead)
        # "Round upward and cornered downward"
        if chin_angle < SHARP_CHIN_THRESH and forehead_to_jaw > 1.05:
            return "Heart"
            
        if jfr > WIDE_JAW_THRESH:
            # Wide jaw implies Square or Round
            # Use Angle AND Solidity
            # Square: Sharp Angle (<130) AND High Solidity (Backs User's "Shape Entire Size" rule)
            # Round: Soft Angle (>130) OR Low Solidity (Cuts corners)
            
            solidity = metrics.get('solidity', 0)
            
            if angle < SQUARE_JAW_ANGLE_THRESH:
                return "Square" 
            elif solidity > 0.95:
                # Even if angle is soft, if it fills the box, it's a "Soft Square" (Boxy)
                return "Square"
            else:
                return "Round"
        else:
            # Narrower chin but wide face
            if jfr < 0.65 or chin_angle < SHARP_CHIN_THRESH:
                return "Heart"
            else:
                return "Round"
    else:
        # Long Faces: Oval, Oblong
        # "Oblong has wide heighted shape" -> implied Rectangular
        # "Oval has heighted shape (Curved)"
        
        if jfr > WIDE_JAW_THRESH or angle < SQUARE_JAW_ANGLE_THRESH:
             # Wide or Sharp Jaw = Oblong (Rectangular)
             return "Oblong"
        else:
             # Soft Jaw = Oval
             return "Oval"
