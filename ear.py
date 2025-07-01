from scipy.spatial import distance as dist


# https://medium.com/analytics-vidhya/eye-aspect-ratio-ear-and-drowsiness-detector-using-dlib-a0b2c292d706
def CheckEAR(eyeData, threshold):
    p1, p2, p3, p4, p5, p6 = eyeData
    p2_p6 = dist.euclidean(p2, p6)
    p3_p5 = dist.euclidean(p3, p5)
    p1_p4 = dist.euclidean (p1, p4)
    EAR = ((p2_p6)+(p3_p5))/(2*(p1_p4))
    #print(EAR)
    return (True if EAR<threshold else False)
