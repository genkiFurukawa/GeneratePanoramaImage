#-*- using:utf-8 -*-
import cv2 as cv
import numpy as np
import time

def measure_video_processing_time(video_src, interval):
    start = time.time()
    cap= cv.VideoCapture(video_src)
    frame_number = cap.get(cv.CAP_PROP_FRAME_COUNT)
    print('frame number:%d' % frame_number)
    frame_idx = 0
    while True:
        _ret, frame = cap.read()
        if frame is None:
            break
        frame_idx = frame_idx + interval
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
        frame_idx += 1
    cap.release()
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

def main():
    print('Start')
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
    measure_video_processing_time(video_src, 50)
    measure_video_processing_time(video_src, 10)
    measure_video_processing_time(video_src, 0)
    print('Done')

if __name__ == '__main__':
    main()
    cv.destroyAllWindows()