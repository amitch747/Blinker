import math

def calc_3D_dist(lm1, lm2):
    dx = lm1.x - lm2.x
    dy = lm1.y - lm2.y
    dz = lm1.z - lm2.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)