import numpy as np
import pandas as pd
import os
import cv2
import tqdm

def spilt_train(path,frame):

    list_file_name = os.listdir(path)

    for j in range(len(list_file_name)):

        name = os.listdir(f"train\\{list_file_name[j]}")
        os.mkdir(f"{list_file_name[j]}_frame")

        for i in tqdm.tqdm(range(len(name))):

            vidcap = cv2.VideoCapture(f'train\\{list_file_name[j]}\\{name[i]}')

            success,image = vidcap.read()

            count = 0



            while success:

                if count % frame == 0 :
                    cv2.imwrite(f"{list_file_name[j]}_frame\\{name[i]}_frame_{count}.jpg", image)     # save frame as JPEG file

                success,image = vidcap.read()

                count += 1


def spilt_test(path,frame):

    list_file_name = os.listdir(path)

    os.mkdir("test_frame")

    for i in tqdm.tqdm(range(len(list_file_name))):

        vidcap = cv2.VideoCapture(f'test/{list_file_name[i]}')

        success,image = vidcap.read()

        count = 0


    while success:

        if count % frame == 0 :
            cv2.imwrite(f"test_frame/{list_file_name[i]}_frame%d.jpg" % count, image)     # save frame as JPEG file

        success,image = vidcap.read()

        count += 1

