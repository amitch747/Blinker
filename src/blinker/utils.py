import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

font = ImageFont.truetype("./misc/cs_regular.ttf", 24)


def draw_text_pil(frame, txt, pos, color=(0,255,0)):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text(pos, txt, font=font, fill=color[::-1]) 
    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return frame    