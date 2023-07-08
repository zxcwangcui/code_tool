import os
import random
from collections import deque, defaultdict
import numpy as np

import cv2
import pandas as pd

input_data = r'D:\Project\GMOT\GenericMOT\GenericMOT_JPEG_Sequence\bird-2\img1'
input_track = r'D:\Project\GMOT\GenericMOT\track_label\bird-2.txt'
output_track = r'D:\Project\GMOT\GenericMOT\result_plot\track\bird2'

if not os.path.exists(output_track):
    os.makedirs(output_track)

data_pack = pd.read_csv(input_track)
data_pack.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'v1', 'v2', 'v3', 'v4']

selected = ['frame', 'id', 'x', 'y', 'w', 'h']
data_selected = data_pack[selected]

# ids = list(set(data_selected['id'].values))
frame_count = len(list(set(data_selected['frame'].values)))
img_list = [img_id for img_id in os.listdir(input_data)]

ids = list(set(data_pack['id'].values))
clrs_n = max([len(ids), 15000])
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
          for _ in range(clrs_n)]

max_len = 200
q_list = [deque(maxlen=max_len) for _ in range(len(ids))]

for n_frame in range(1, frame_count + 1):
    data_frame = data_selected[data_selected['frame'] == n_frame]
    img = cv2.imread(os.path.join(input_data, img_list[n_frame - 1]))

    for idx, data in data_frame.iterrows():

        cv2.rectangle(img, pt1=(int(data.x), int(data.y)),
                      pt2=(int(data.x) + int(data.w),
                           int(data.y) + int(data.h)),
                      color=colors[int(data.id)], thickness=2)

        cv2.putText(img, str(int(data.id)), (int(data.x), int(data.y)), cv2.FONT_HERSHEY_PLAIN, 3,
                    color=colors[int(data.id)],
                    thickness=3)
        cv2.circle(img, (int(data.x+(data.w/2)), int(data.y+data.h)), 5, color=colors[data.id], thickness=-1)

        for index, q in enumerate(q_list):
            if index == data.id:
                if len(q) > 0:

                    for i in range(1, len(q)):
                        if q[i - 1] is None or q[i] is None:
                            continue
                        thickness = int(np.power(32 / float(i / 3 + 1), 0.3) * 2.5)
                        cv2.circle(img, q[-i], thickness, color=colors[int(data.id)], thickness=-1)
                if len(q) < max_len:
                    q.append((int(data.x+(data.w/2)), int(data.y+data.h)))
                else:
                    q.popleft()
                    q.append((int(data.x+(data.w/2)), int(data.y+data.h)))

    cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(output_track, img_list[n_frame - 1]), img)
