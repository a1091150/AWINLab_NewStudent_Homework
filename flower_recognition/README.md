
# Flowers Recognition
## 說明
[Flowers Recognition](https://www.kaggle.com/alxmamaev/flowers-recognition) 為辨識花朵，這裡用來繳交 AWINLab 的作業要求
1. 使用 Keras 的方法建立模型
2. 使用 CNN or DNN
3. 訓練集(Training Set)訓練完後，請針對測試集(Testing set)透過一些性能指標衡量模型的好壞，並印出測試集隨機十筆預測後的結果和正解進行比較

前兩項要求實作於 flowers-recognition.ipynb，其完全照抄自 [Aniruddha Bhosale](https://www.kaggle.com/code/aniruddha00707/simple-conv2d-model-tensorflow-14042022) 的上傳

第三項要求實作於 flowers-tentest.ipynb

## 簡單紀錄

按照 tensorflow 官方安裝說明，即可使用筆記型電腦中的 GPU: GTX 3060 6 GB，下列 terminal 擷取的輸出為正在訓練模型時的內容，可見 python 佔用 4995 MiB
```
➜  ~ nvidia-smi
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1123      G   /usr/lib/xorg/Xorg                 35MiB |
|    0   N/A  N/A      1663      G   /usr/lib/xorg/Xorg                125MiB |
|    0   N/A  N/A      1790      G   /usr/bin/gnome-shell              115MiB |
|    0   N/A  N/A      2383      G   ...RendererForSitePerProcess       50MiB |
|    0   N/A  N/A      2582      G   ...005511058114334029,131072      145MiB |
|    0   N/A  N/A      3073      C   ...dokipb/otensor/bin/python     4995MiB |
+-----------------------------------------------------------------------------+
```

### 模型層
```python
data_augmentation = Sequential([
    layers.RandomFlip(mode="horizontal",
                     input_shape=(img_height,img_width,3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
],name="data_augmentation_layer")
num_classes = len(class_names)

model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255,name="rescaling_layer"),
    layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(rate=0.2),
    layers.Flatten(),
    layers.Dense(units=128, activation="relu"),
    layers.Dense(units=128, activation="relu"),
    layers.Dense(units=128, activation="relu"),
    layers.Dense(units=64, activation="relu"),
    layers.Dense(units=32, activation="relu"),
    layers.Dense(units=num_classes)
])
```
用 GPU 訓練花了兩分鐘，也是挺久的。

## 過程與心得

之前，我有額外花時間去嘗試寫 Stanford 大學的 2017 cs231n 課程作業，課程中間進度使用 pytorch 建立 CNN, DNN，但是對我來說很困難。
在以前我的 iOS App 的功能中，我曾使用 keras 撰寫 CoreML model，並使用擷圖來作為參數的輸入。

在這次作業中，我以為要自己處理圖片，這些圖片有大有小，長寬不一處理起來很麻煩。
但是意外的在 ```Code``` 標籤頁中找到使用 tensorflow 與 keras 撰寫的程式碼，所以我直接照抄一遍程式碼～