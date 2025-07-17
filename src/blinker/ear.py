from scipy.spatial import distance as dist
from typing import Sequence, Optional


# https://medium.com/analytics-vidhya/eye-aspect-ratio-ear-and-drowsiness-detector-using-dlib-a0b2c292d706
def EAR_check(eyeData, threshold):
    EAR = safe_EAR_calc(eyeData)
    print(eyeData)
    if EAR:
        return (True if EAR<threshold else False)
    return None

def safe_EAR_calc(eyeData: Sequence[tuple[int, int]]) -> Optional[int]:
    if len(eyeData) != 6:
        return None

    for p in eyeData:
        if not isinstance(p, tuple) or len(p) != 2:
            return None
        if not all(isinstance(coord, (int, float)) for coord in p):
            return None


    try:
        p1, p2, p3, p4, p5, p6 = eyeData
        p2_p6 = dist.euclidean(p2, p6)
        p3_p5 = dist.euclidean(p3, p5)
        p1_p4 = dist.euclidean (p1, p4)
        if p1_p4 < 1e-8: return None
        
        EAR = ((p2_p6)+(p3_p5))/(2*(p1_p4))
        if not (0 <= EAR <= 1): return None
        return EAR
    except Exception:
        return None
    

def calc_thresh(eyeData):
    # assume closed eye
    print(eyeData)
    p1, p2, p3, p4, p5, p6 = eyeData
    p2_p6 = dist.euclidean(p2, p6)
    p3_p5 = dist.euclidean(p3, p5)
    p1_p4 = dist.euclidean (p1, p4)
    EAR = ((p2_p6)+(p3_p5))/(2*(p1_p4))
    return EAR
