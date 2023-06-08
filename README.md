# realsense-yolox
# realsenseとyoloxを使用した、人体密度の可視化
intel realsenseD435と物体検出yoloxを同時に動作させ、人体の３次元座標を計測します。

その結果から人体の密度をカーネル密度推定を行い、人体の密度の可視化を行いました。

コードを整理していないのでごっちゃになっています。整理にはもう少し時間をくださいすみません。

使用言語　python

realsense.pyはrealsenseをクラス化したコードです。

yolox_motpy2.pyは物体追跡も搭載しています。基本はこのコードで動きます（はず）。
