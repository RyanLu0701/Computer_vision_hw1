import tqdm
import cv2
import numpy as np
import os

def readfile(num,test = False,label = None):

    if test == False:
        frame_list = [i for i in os.listdir() if "frame" in i and "test" not in i]
    else:
        frame_list = [i for i in os.listdir() if "test_frame" in i]



    x = np.zeros((num , 128, 128, 3), dtype=np.uint8)
    y = np.zeros( num , dtype=np.uint8)

    count = 0
    for t in range(len(frame_list)):



        image_dir = sorted(os.listdir(frame_list[t]))
        for  file in  tqdm.tqdm(image_dir):

            # if count == 0 :
            #     file_name = file[0:5]

            # if file[0:5] == file_name:

            img = cv2.imread(os.path.join(frame_list[t], file))
            x[count, :, :] =cv2.resize(img, (128, 128))

            if label is not None:
                y[count] = frame_list[t].split("_")[0]


            # else :

            #     fill_zero_num  = 16 - (count%16) # 差多少到16

            #     count = fill_zero_num

            #     file_name = file[0:5]

            #     img = cv2.imread(os.path.join(frame_list[t], file))

            #     x[count, :, :] = cv2.resize(img, (128, 128))

            #     if t != None:

            #         y[count] = t

            count += 1

    if label is not None:

        return x, y

    else:
        return x



