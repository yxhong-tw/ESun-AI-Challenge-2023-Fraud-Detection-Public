# ESun-AI-Challenge-2023-Fraud-Detection-Public

本專案為 [AI CUP 2023 玉山人工智慧公開挑戰賽－信用卡冒用偵測 ](https://tbrain.trendmicro.com.tw/Competitions/Details/31) 的參賽原始碼。

## 專案資訊
### 檔案架構
```
.
├ configs
│ ├ params
│ │ ├ CatBoostClassifier.json
│ │ └ README.md
│ ├ configs.ini
│ └ README.md
├ data
│ └ README.md
├ logs
│ ├ FY-2.log
│ ├ FY-4.log
│ ├ HAHAHAA-9.log
│ ├ HAHAHAA-10.log
│ └ README.md
├ Model
│ └ README.md
├ outputs
│ ├ checkpoints
│ │ └ README.md
│ ├ EDAs
│ │ └ README.md
│ ├ openfes
│ │ └ README.md
│ ├ predictions
│ │ ├ HAHAHAA-10.csv
│ │ └ README.md
│ ├ studies
│ │ └ README.md
│ └ README.md
├ Preprocess
│ └ README.md
├ .gitignore
├ LICENSE
├ evalute.py
├ gadget.py
├ inference.py
├ initialize.py
├ main.py
├ my_openfe.py
├ README.md
├ requirements.txt
├ save.py
├ train.py
└ utils.py
```
> 關於 `Model`、`Preprocess` 資料夾為空之問題，我們隊伍的解釋如下：
> 1. 由於我們並沒有自行撰寫新模型，僅套用現成模型（CatBoost），因此我們不認為有程式應被歸於 `Model` 之下，故留空。
> 2. `Preprocess` 部份，在我們的專案設計中，我們將資料前處理歸於初始化步驟（initialize），經團隊討論後認為，我們若強硬地將資料前處理從原先的初始化部份拆分出來，會使專案結構變得有些奇怪，因此也留空。
>
> 以上解釋，供主辦單位參考，希望能體諒我們在專案架構設計上的理念，非常感謝！

### 檔案說明
- `configs/`: 存放所有參數設定檔的資料夾。
- `configs/configs.ini`: 程式執行的參數設定檔。
- `configs/params/`: 存放所有模型參數設定檔的資料夾。
- `configs/params/CatBoostClassifier.json`: CatBoostClassifier 模型的參數設定檔。
- `data/`: 存放用於訓練、驗證、推理的檔案（包含主辦單位提供之原始檔案，以及我們隊伍通過特徵工程生成的檔案）的資料夾。
- `logs/`: 存放程式執行日誌的資料夾。
- `Model/`: 主辦單位指定要存在的資料夾，但我們沒有使用。
- `outputs/`: 存放程式執行結果的資料夾。
- `outputs/checkpoints/`: 存放模型訓練檢查點的資料夾。
- `outputs/EDAs/`: 存放探索性資料分析結果的資料夾。
- `outputs/openfes/`: 存放 OpenFE 運算中間產物及結果的資料夾。
- `outputs/predictions/`: 存放推理結果的資料夾。
- `outputs/studies/`: 存放 Optuna 運算中間產物的資料夾。
- `Preprocess/`: 主辦單位指定要存在的資料夾，但我們沒有使用。
- `.gitignore`: Git 版本控制忽略檔案。
- `LICENSE`: 宣告本專案權利檔案。
- `evalute.py`: 評估模型效能的程式。
- `gadget.py`: 一些小工具。
- `inference.py`: 推理的程式。
- `initialize.py`: 初始化程式。
- `main.py`: 主程式。
- `my_openfe.py`: OpenFE 的程式。
- `README.md`: 專案說明文件。
- `requirements.txt`: 套件需求檔案。
- `save.py`: 儲存程式執行結果的程式。
- `train.py`: 訓練模型的程式。
- `utils.py`: 一些工具。

## 環境資訊
### 套件
```
catboost==1.2.2
numpy==1.24.4
openfe==0.0.12
optuna==3.4.0
pandas==2.0.3
tables==3.9.1
ydata-profiling==4.6.1
```
> 請使用 `pip3 install -r requirements.txt` 安裝所需套件。

### 超參數設定
```
- CatBoostClassifier
  - 'random_strength': 0.1275436492418589
  - 'bootstrap_type': 'Bayesian'
  - 'boosting_type': 'Ordered'
  - 'iterations': 544
  - 'task_type': 'GPU'
  - 'devices': '0'
  - 'nan_mode': 'Forbidden'
  - 'bagging_temperature': 0.65
  - 'loss_function': 'Logloss'
  - 'grow_policy': 'SymmetricTree'
  - 'l2_leaf_reg': 9.78874631581315
  - 'depth': 6
  - 'feature_border_type': 'GreedyLogSum'
  - 'learning_rate': 0.2218586703700778
  - 'random_seed': 48763
  - 'eval_metric': 'F1'
```
> 注意：以上參數設定僅適用由 OpenFE 經特徵工程所產生之訓練資料集，並非主辦單位提供之原始訓練資料集。

### 執行資源
```
- Ubuntu 20.04.6 LTS
- Python == 3.10.12
- Intel(R) Xeon(R) Silver 4216
- NVIDIA GeForce RTX 3090 (1 ~ 4 張)
  - Driver Version: 535.54.03
  - CUDA Version: 12.2
- 252 GB Main Memory + 477 GB Swap Memory
```
> 注意：
>   - 若以我們的參數設置，使用 `openfe-train` 模式執行程式，主記憶體與交換記憶體總和需大於 500 GB。
>   - 若以 `openfe-inference` 模式執行程式，則主記憶體與交換記憶體總和至少需大於等於 128 GB。

## 使用方法
本專案的 `main.py` 整合了七種模式（cv_train_with_optuna, evalute, inference, train, train_with_optuna, openfe-inference, openfe-train），可透過 `--mode` 參數指定要執行的模式，詳細用法可以通過 `python3 main.py -h` 查看，如下所示：

```
// Input
> python3 main.py -h

// Output
usage: main.py [-h] [-c CONFIG] -m \
{cv_train_with_optuna,evalute,inference,train, \
train_with_optuna,openfe-inference,openfe-train} [-uofed]
options:
  -h, --help
        show this help message and exit
  -c CONFIG, --config CONFIG
        Path to the configs file.
  -m {cv_train_with_optuna,evalute,inference,train, \
        train_with_optuna,openfe-inference,openfe-train}, \
    --mode {cv_train_with_optuna,evalute,inference,train, \
        train_with_optuna,openfe-inference,openfe-train}
        Mode to run the program.
        Notice that:
        1. "evalute", "inference" and "openfe-inference" modes must load a checkpoint.
        2. Checkpoints only can be used in "evalute", "inference" and "openfe-inference" modes. \
        There may be some unexpected errors if you use checkpoints in other modes.
  -uofed, --use_openfe_data
        Add this argument if you want to use the training data generated by OpenFE.
```

## 執行流程
```
# 安裝所需套件
> pip3 install -r requirements.txt

# 執行程式（根據需求，選擇相應的模式；若想使用 OpenFE 產生的資料，則加上 `-uofed` 參數）
# MODE = {cv_train_with_optuna, evalute, inference, train, \
#         train_with_optuna, openfe-inference, openfe-train}
# 注意：
#   - 如果不使用 OpenFE，需要把 `utils.py` 的第 128 行至 132 行取消註解（For not OpenFE data），\
#     第 134 行至 153 行則需註解。
#   - 若使用 OpenFE，則要把 `utils.py` 的第 128 行至 132 行註解，第 134 行至 153 行則視需求取消註解。
> python3 main.py -m MODE [-uofed]
```
