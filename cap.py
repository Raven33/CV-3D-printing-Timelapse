from multiprocessing import Process, Queue
import numpy as np
import cv2
import ssim
from PIL import Image
from tqdm import tqdm

source = "MVI_0001.mp4"
test = 2
counteur = 0
packetsize = 50
file = 0

path = source.split(".")[0]

cap = cv2.VideoCapture(source)  # Pick a source - either a file or a (exiting) device
ret, previous = cap.read()
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
advancement = cap.get(cv2.CAP_PROP_POS_FRAMES)


class Frame:
    index = 0
    frame = 0


class Worker(Process):
    def __init__(self, prev, queue, result):
        super(Worker, self).__init__()
        self.queue = queue
        self.result = result
        self.prev = prev

    def run(self):
        for frame in iter(self.queue.get, None):
            f = Frame()
            f.index = frame.index
            fr = cv2.cvtColor(cv2.resize(frame.frame, (int(frame.frame.shape[0]/2),int(frame.frame.shape[1]/2)), interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            fr = Image.fromarray(fr)
            f.frame = ssim.compute_ssim(fr, self.prev, gaussian_kernel_sigma=1.5, gaussian_kernel_width=11)
            self.result.put(f)
        self.result.put("STOP")

def testpool(prev, Frames):
    request_queue = Queue()
    result = Queue()
    results = []
    Processes = []
    prev = cv2.cvtColor(cv2.resize(prev, (int(prev.shape[0]/2),int(prev.shape[1]/2)), interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    prev = Image.fromarray(prev)
    for i in range(12):
        Processes.append(Worker(prev, request_queue, result))
    for data in Frames:
        request_queue.put(data)
    for worker in Processes:
        worker.start()
    # Sentinel objects to allow clean shutdown: 1 per worker.
    for i in range(12):
        request_queue.put(None)
    nonec = 0
    while(nonec<12):
        r = result.get()
        if r == "STOP":
            nonec= nonec + 1
        else:
            results.append(r)

    Max = 0
    Index = 0
    for j in range(0, len(results)):
        if (results[j].frame > Max):
            Max = results[j].frame
            Index = results[j].index%packetsize
    return Frames[int(Index)]


if __name__ == '__main__':
    pbar = tqdm(total=int(length / 50), ascii=True)
    while (advancement < length):
        frames = []
        for i in range(0, packetsize):
            if (advancement < length):
                f = Frame()
                ret, frame = cap.read()
                f.frame = frame
                f.index = advancement
                frames.append(f)
                advancement = cap.get(cv2.CAP_PROP_POS_FRAMES)
            else:
                break
        bestframe = testpool(previous, frames)
        del frames
        cv2.imwrite(path + '/%05d.png' % file, bestframe.frame)
        file += 1
        previous = bestframe.frame
        pbar.update(1)

    pbar.close()
    cap.release()
    cv2.destroyAllWindows()
