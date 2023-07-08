



import os
import random

import cv2
import pandas as pd

input_data = r'D:\Project\GMOT\GenericMOT\GenericMOT_JPEG_Sequence\bird-2\img1'
imput_track = r'D:\Project\GMOT\GenericMOT\detecion_label_yolo\bird-2.txt'
output_track = r'D:\Project\GMOT\GenericMOT\result_plot\detection\bird2'

if not os.path.exists(output_track):
    os.makedirs(output_track)

data_pack = pd.read_csv(imput_track)
# data_pack.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'v1', 'v2', 'v3', 'v4']
data_pack.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'confidence', 'v1', 'v2', 'v3']

selected = ['frame', 'x', 'y', 'w', 'h', 'confidence']
data_selected = data_pack[selected]

# ids = list(set(data_selected['id'].values))
frame_count = len(list(set(data_selected['frame'].values)))
img_list = [img_id for img_id in os.listdir(input_data)]

# clrs_n = max([len(ids), 15000])
# colors = [(255, 0, 0)]

for n_frame in range(1, frame_count + 1):
    data_frame = data_selected[data_selected['frame'] == n_frame]
    img = cv2.imread(os.path.join(input_data, img_list[n_frame - 1]))

    for idx, data in data_frame.iterrows():
        cv2.rectangle(img, pt1=(int(data.x), int(data.y)),
                      pt2=(int(data.x) + int(data.w),
                           int(data.y) + int(data.h)),
                      color=(51, 153, 255), thickness=2)

    cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(output_track, img_list[n_frame - 1]), img)
