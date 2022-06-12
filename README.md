# STAS_detect
2022年AICUP肺腺癌病理切片影像之腫瘤氣道擴散偵測競賽 I：運用物體偵測作法於找尋STAS - 第TEAM_1355組

## 安裝設定與環境版本 (linux)
* python == 3.8
* yapf==0.32.0
* timm==0.5.4
* opencv-python==4.5.5.64
* bbox-visualizer==0.1.0
* addict==2.4.0

### 安裝MMCV
```
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu114/torch1.10.0/index.html 
```

## 使用方式
### 資料集儲存路徑
```
STAS_detect
|___ STAS_DATASET
         train0.txt
         train1.txt
         train2.txt
         train3.txt
         train4.txt
         valid0.txt
         valid1.txt
         valid2.txt
         valid3.txt
         valid4.txt
         |___training
             |___Train_Annotations
                 |(放所有的偵測框label .txt檔)
             |___Train_Images
                 |(放所有的訓練圖片檔)
         |___Test_Images
             |(放所有的測試圖片檔)
         
```
### 開始訓練
```
python train_swin.py
```
* `--id` (int): 選擇使用哪一個子資料集當成驗證資料集(預設為0)
* `--save_dir` (str): 輸入儲存weight和訓練過程的資料夾(預設為"stas_swin")
* `--resize` (str): 選擇resize大小，只有900、850兩種選擇(預設為"850")


         
     
