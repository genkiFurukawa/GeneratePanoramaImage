# GeneratePanoramaImage
generate panorama image from video by using optical flow 

## 作りたいもの
動画から移動距離を計算して、パノラマ画像を生成したい。

## ToDo
* [ ] 移動方向の判定
* [x] 移動距離の算出
* [ ] 移動距離の補正
* [ ] パノラマ化
* [ ] 処理の高速化

## Usage
```
python optical_flow_sample.py (video_src)
```

## 動画の読み込み速度の比較
`measure_execution_time.py`で実行時間の比較を行った。  
当たり前だが、そのフレームを見ないことが確定しているなら、`cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)`を用いて次に見ないフレームを飛ばしてしまった方が実行速度は早い。

```
Start
# 50フレームごと
frame number:236
elapsed_time:4.085411071777344[sec]
# 10フレームごと
frame number:236
elapsed_time:14.298928260803223[sec]
# 全てのフレームをみた時
frame number:236
elapsed_time:161.856849193573[sec]
Done
```
