# YOLOV1_PyTorch_Train
PyTorch完美复现YOLOV1的精度和速度，配置完全相同，两者模型可以无障碍相互转换。

## 指标展示
|Model| train | test | mAP | FPS |
|-----|------|------|-----|-----|
|yolov1-tiny(paper) | 0712 |	2007_test |	52.7 |	155 |
|yolov1-tiny(retrain from darknet) | 0712 |	2007_test |	51.2 |	155 |
|**yolov1-tiny(ours)** | 0712 |	2007_test |	**52.1** |	**155** |

## 效果展示
<img src="assets/result1.jpg" width="400" height="260"/>   <img src="assets/result2.jpg" width="400" height="260"/>   
<img src="assets/result3.jpg" width="400" height="260"/>   <img src="assets/result4.jpg" width="400" height="260"/>   

## 使用说明
### 要求
> Python 3.6 \
> PyTorch >= 1.4
### 数据集下载
```shell script
cd <path-to-voc>/
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```
### 数据生成
```shell script
cd data/voc0712
python voc_label.py
cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt
```
### 预训练模型下载
```shell script
cd pretrain
wget https://pjreddie.com/media/files/darknet.weights
```
### 训练和测试
```shell script
python train.py
```
已训练好的模型：[百度云(提取码:8888)](https://pan.baidu.com/s/1xDWUi5Vwiwnf3VMFjpla_g)
```shell script
python detect.py
```
### 计算mAP
模型转换至darknet
```shell script
python cvt2darknet.py
```
编译原始版本[darknet](https://github.com/pjreddie/darknet)
```shell script
./darknet yolo valid cfg/yolov1-tiny.cfg weights/yolov1-tiny-final.weights
```
将生成的`results`文件夹移入`eval`文件夹
```shell script
python voc_eval.py
```

## 不同之处


## 参考
