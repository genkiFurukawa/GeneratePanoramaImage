# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd

lk_params = dict(winSize  = (15, 15),
                 maxLevel = 2,
                 criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cap= cv.VideoCapture(video_src)
        self.frame_widht= self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
        self.frame_idx = 0
        self.movement_distance = 0
        self.movement_distance_per_frame = []
        self.mean_movement_distance_per_frame = []

    # フレームごとの移動距離を求め、リストで返す
    def calc_movement_distance(self):
        while (True):
            _ret, frame = self.cap.read()

            if self.frame_idx == 0:
                self.movement_distance_per_frame.append([self.frame_idx, 0])
                cv.imwrite('img_' + str(self.frame_idx) + '.jpg', frame)

            # 次のフレームがない時は無限ループから抜ける
            if frame is None:
                break

            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                # 特徴点をopenCVの仕様を満たした形に変換
                # 特徴点はdetect_intervalで指定した間隔で取得した特徴点
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                # p1:検出した対応点
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                # オプティカルフローの逆方向への探索を行い安定した追跡結果だけを選択する
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1

                new_tracks = []
                # 各フレームごとに移動量を求める
                dist_list = []
                for tr, (x0, y0), (x1, y1), good_flag in zip(self.tracks, p0.reshape(-1, 2), p1.reshape(-1, 2), good):
                    # pythonは参照渡し
                    if not good_flag:
                        continue
                    tr.append((x1, y1))
                    if len(tr) > self.track_len:
                        # 0番目の要素の削除
                        del tr[0]
                    new_tracks.append(tr)
                    cv.circle(vis, (x1, y1), 5, (0, 255, 0), -1)
                    cv.circle(vis, (x0, y0), 5, (255, 255, 255), -1)
                    cv.line(vis, (x0, y0), (x1, y1), (0, 0, 0), 2)
                    dist_list.append(np.sqrt((x0-x1)**2 + (y0-y1)**2))

                if len(dist_list) > 0:
                    self.movement_distance += sum(dist_list)/ len(dist_list)
                    self.movement_distance_per_frame.append(dist_list)
                    if self.movement_distance > self.frame_widht:
                        self.movement_distance = 0
                        cv.imwrite('img_' + str(self.frame_idx) + '.jpg', frame)
                else:
                    self.movement_distance_per_frame.append(dist_list)

                self.tracks = new_tracks
                # 特徴点から現在のフレームまでの追跡の軌跡を描写
                cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

            # 5フレームごとに特徴点抽検出を行う
            if self.frame_idx % self.detect_interval == 0:
                # zeros_like:元の配列と同じ形にして0を代入
                # 特徴点を背景が白の画像に黒色の点でプロットする
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                # コーナー検出
                p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv.namedWindow('lk_track', cv.WINDOW_NORMAL)
            cv.imshow('lk_track', vis)

            ch = cv.waitKey(1)
            if ch == 27:
                break
        return self.movement_distance_per_frame

    # フレームごとの移動距離を可視化する
    def show_movement_distance_per_frame(self):
        x = []
        y = []
        for i, distance_list in enumerate(self.movement_distance_per_frame):
            for distance in distance_list:
                x.append(i)
                y.append(distance)
        plt.scatter(x, y)
        plt.xlabel('frame index')
        plt.xlabel('movement distance')
        plt.savefig('figure.png')
    
    # 外れ値を除いて平均値を求める処理
    def outlier_iqr(self):
        for item in self.movement_distance_per_frame:
            if len(item) != 0:
                s = pd.Series(item)
                # 四分位数
                q1 = s.describe()['25%']
                q3 = s.describe()['75%']
                iqr = q3 - q1 #四分位範囲
                # 外れ値の基準点
                outlier_min = q1 - (iqr) * 1.5
                outlier_max = q3 + (iqr) * 1.5
                # 範囲から外れている値を除く
                s = s.clip(outlier_min, outlier_max)
                self.mean_movement_distance_per_frame.append(s.mean())
            else:
                self.mean_movement_distance_per_frame.append(0)
    
    # 切りだすフレームのインデックスを求める処理

    # 切り出した画像を結合する処理

def main():
    print('Start')
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    app = App(video_src)
    movement_distance_per_frame = app.calc_movement_distance()
    app.show_movement_distance_per_frame()
    app.outlier_iqr()
    print('Done')


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()