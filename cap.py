from tqdm import tqdm
import numpy as np
import cv2
import ssim
from PIL import Image
import time
 
cap = cv2.VideoCapture('video.mp4')
 
ret,previous = cap.read()
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
advancement = cap.get(cv2.CAP_PROP_POS_FRAMES)
best_frame = previous
number = 50
file = 0
pbar = tqdm(total=int(length/50),ascii=True)
while(advancement<length):
    i = 0
    best_s = 0
    while(i<Blocsize and advancement<length):
        i += 1
        ret, frame = cap.read()
        fr = cv2.cvtColor(cv2.resize(frame, (int(frame.shape[0]/2),int(frame.shape[1]/2)), interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        fr = Image.fromarray(fr)

        prev = cv2.cvtColor(cv2.resize(previous, (int(previous.shape[0]/2),int(previous.shape[1]/2)), interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        prev = Image.fromarray(prev)
        s = ssim.compute_ssim(fr, prev, gaussian_kernel_sigma=1.5, gaussian_kernel_width=11)
        if(s > best_s):
            best_s = s
            best_frame = frame
    cv2.imwrite('folder/%d.png' % file,best_frame)
    file += 1
    pbar.update(1)
    previous = best_frame

pbar.close()
cap.release()
cv2.destroyAllWindows()
