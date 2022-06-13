# STAS_detect
2022年AICUP肺腺癌病理切片影像之腫瘤氣道擴散偵測競賽 I：運用物體偵測作法於找尋STAS - 第TEAM_1355組

## 安裝設定與環境版本 (Linux)
### 環境
* python == 3.8
* pytorch container version == 21.08
* CUDA == 11.4

### 套件
```
pip install -r requirements.txt
```

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
* `--epochs` (int): 選擇epoch的大小(預設為40)

### 測試並產生json檔結果
```
python test_swin.py
```
* `--resize` (str): 選擇resize大小，只有900、850兩種選擇(預設為"850")
* `--batch_size` (str): 選擇batch size的大小(預設為32)
* `--checkpoint` (str): 輸入checkpoint路徑位置(checkpoint即為訓練時擁有最高AP50的權重)
* `--save_vis` (str): 輸入資料夾名稱以儲存畫出bounding box的圖片結果(預設為"result")
* `--output` (str): 輸出json檔的檔名(預設為"result.json")

<b>亦可直接下載我們Private Leaderboard最高分數的權重(https://drive.google.com/drive/folders/1Px6YLB6oK-8q0MnemUeMzCWT_AUPG6wI?usp=sharing) 作為checkpoint，以重現結果</b>

### Ensemble
```
python wbf.py
```
* `--json_dir` (str): 選擇原本5次使用不同validation set得到的json測試結果的資料夾 (預設為"./label")
* `--output` (str): 輸出ensemble結果的json檔名 (預設為"ensemble.json")


### 參考資料
* 程式碼修改自 https://github.com/SwinTransformer/Swin-Transformer-Object-Detection 、 https://github.com/cv516Buaa/tph-yolov5 
     
