# 匯入所需的函式庫
import pandas as pd  # 用於資料處理，主要是 DataFrame 結構
import numpy as np  # 用於數值運算
import tensorflow as tf  # 匯入 TensorFlow，主要用於深度學習模型
from sklearn.preprocessing import StandardScaler  # 用於特徵標準化
from sklearn.model_selection import StratifiedKFold  # 用於分層 K 折交叉驗證
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix  # 用於模型性能評估的函式
from tensorflow.keras.models import Sequential  # 用於建立 Keras 模型
from tensorflow.keras.layers import Dense, LSTM, Dropout  # 用於建立神經網路層：全連接層、LSTM 層、Dropout 層
from tensorflow.keras.utils import to_categorical  # 用於將標籤轉換為 one-hot 編碼
import matplotlib.pyplot as plt  # 用於繪製圖表
import seaborn as sns  # 用於繪製精美的圖表，尤其是熱力圖
from matplotlib import font_manager  # 用於管理字型
import os  # 用於處理檔案路徑等操作
import joblib  # 用於儲存和載入標準化器

# 使用 pd (pandas) 來處理資料集，進行資料讀取與預處理
# 使用 numpy 來進行數學運算，特別是在資料轉換或數據操作時
# 使用 TensorFlow 與 Keras 來建立與訓練深度學習模型，特別是 LSTM 模型
# 使用 scikit-learn 中的工具來進行資料標準化，模型評估以及交叉驗證

# 以下是每個模組的詳細功能：
# 1. pandas 是資料處理工具，可以將資料讀取進來並轉換成 DataFrame 結構，便於進行數據操作。
# 2. numpy 用於高效的數值運算，特別是在大規模資料處理與向量化計算中常用。
# 3. tensorflow 提供了強大的深度學習框架，包含了多種神經網絡層，LSTM 就是其中之一。
# 4. sklearn 提供了標準化與交叉驗證的工具，還有多種模型評估的指標，像是混淆矩陣、ROC 曲線等。
# 5. matplotlib 和 seaborn 是用來繪製資料視覺化圖表的，對於結果呈現和報告非常有幫助。
# 6. font_manager 是用來管理字型的，當在繪圖中使用中文或其他特殊字型時非常有用。
# 7. os 用於文件和路徑處理，讓你能夠操作檔案系統，特別是進行資料儲存與載入操作。



# 設定中文字型
plt.rcParams['font.family'] = 'Microsoft YaHei'

# 設置隨機種子
np.random.seed(42)
tf.random.set_seed(42)

def plot_training_history(history, fold_num=None):
    """繪製訓練歷史圖表"""
    
    # 設定子圖的大小，2行2列的子圖佈局
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 損失函數圖表
    axes[0, 0].plot(history.history['loss'], label='訓練損失', linewidth=2)  # 繪製訓練損失
    if 'val_loss' in history.history:
        axes[0, 0].plot(history.history['val_loss'], label='驗證損失', linewidth=2)  # 繪製驗證損失
    
    # 標題設定
    title = '模型損失'
    if fold_num is not None:
        title += f' (第 {fold_num} 折)'  # 如果有交叉驗證折數，加入折數於標題中
    
    # 設定圖表標題和座標軸標籤
    axes[0, 0].set_title(title, fontsize=14, pad=15)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)  # x軸為訓練回合數
    axes[0, 0].set_ylabel('Loss', fontsize=12)  # y軸為損失值
    axes[0, 0].legend(fontsize=10)  # 顯示圖例
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)  # 加入網格線，並調整透明度
    
    # 2. 準確率圖表
    axes[0, 1].plot(history.history['accuracy'], label='訓練準確率', linewidth=2)  # 繪製訓練準確率
    if 'val_accuracy' in history.history:
        axes[0, 1].plot(history.history['val_accuracy'], label='驗證準確率', linewidth=2)  # 繪製驗證準確率
    
    # 標題設定
    title = '模型準確率'
    if fold_num is not None:
        title += f' (第 {fold_num} 折)'  # 如果有交叉驗證折數，加入折數於標題中
    
    # 設定圖表標題和座標軸標籤
    axes[0, 1].set_title(title, fontsize=14, pad=15)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)  # x軸為訓練回合數
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)  # y軸為準確率
    axes[0, 1].legend(fontsize=10)  # 顯示圖例
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)  # 加入網格線，並調整透明度
    
    # 3. AUC圖表
    if 'auc' in history.history and 'val_auc' in history.history:
        axes[1, 0].plot(history.history['auc'], label='訓練 AUC', linewidth=2)  # 繪製訓練AUC
        axes[1, 0].plot(history.history['val_auc'], label='驗證 AUC', linewidth=2)  # 繪製驗證AUC
        
        # 標題設定
        title = '模型 AUC'
        if fold_num is not None:
            title += f' (第 {fold_num} 折)'  # 如果有交叉驗證折數，加入折數於標題中
        
        # 設定圖表標題和座標軸標籤
        axes[1, 0].set_title(title, fontsize=14, pad=15)
        axes[1, 0].set_title(title, fontsize=14, pad=15)
        axes[1, 0].title.set_position([0, 1])  # 將標題的水平位置設為接近左邊

        axes[1, 0].set_xlabel('Epoch', fontsize=12)  # x軸為訓練回合數
        axes[1, 0].set_ylabel('AUC', fontsize=12)  # y軸為AUC值
        axes[1, 0].legend(fontsize=10)  # 顯示圖例
        axes[1, 0].grid(True, linestyle='--', alpha=0.7)  # 加入網格線，並調整透明度

    # 4. 學習率圖表
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], label='學習率', linewidth=2, color='green')  # 繪製學習率變化
        
        # 標題設定
        title = '學習率變化'
        if fold_num is not None:
            title += f' (第 {fold_num} 折)'  # 如果有交叉驗證折數，加入折數於標題中
        
        # 設定圖表標題和座標軸標籤
        axes[1, 1].set_title(title, fontsize=14, pad=15)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)  # x軸為訓練回合數
        axes[1, 1].set_ylabel('Learning Rate', fontsize=12)  # y軸為學習率
        axes[1, 1].legend(fontsize=10)  # 顯示圖例
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)  # 加入網格線，並調整透明度
        axes[1, 1].set_yscale('log')  # 設定y軸為對數尺度

    # 調整子圖之間的間距，避免重疊
    plt.tight_layout()
    # 顯示圖表
    plt.show()


def 載入與預處理數據(文件路徑):
    # 讀取資料
    數據框 = pd.read_csv(文件路徑)
    
# 定義特徵預測名稱
    特徵列 = [
        'age',        # 年齡
        'sex',        # 性別 (0: 女性, 1: 男性)
        'cp',         # 胸痛類型 (0: 無胸痛, 1: 頸部, 2: 心絞痛, 3: 非心絞痛)
        'trestbps',   # 靜態血壓 (mm Hg)
        'chol',       # 胆固醇 (mg/dl)
        'fbs',        # 空腹血糖 (1: 大於 120 mg/dl, 0: 小於等於 120 mg/dl)
        'restecg',    # 靜態心電圖結果 (0: 正常, 1: ST-T 波異常, 2: 左心室肥大)
        'thalach',    # 最大心跳速率 (bpm)
        'exang',      # 運動引起的心絞痛 (1: 有, 0: 無)
        'oldpeak',    # 由於運動引起的 ST 降低 (以毫米為單位)
        'slope',      # ST 段的坡度 (0: 上升, 1: 平坦, 2: 下降)
        'ca',         # 主要血管數量 (0 到 3)
        'thal'        # 地中海貧血 (3: 正常, 6: 固定缺陷, 7: 可逆缺陷)
    ]

    
    # 分割特徵 X 和目標標籤 y
    X = 數據框[特徵列].values
    y = 數據框['target'].values
    
    return X, y, 特徵列

# 定義 LSTM 模型
def 建立_LSTM模型(輸入維度):
    # 建立一個順序模型
    模型 = Sequential([
        # LSTM 層：128 個單元，輸入維度為 (輸入維度, 1)，使用 tanh 激活函數，return_sequences=False 代表只輸出最後一個時間步的結果
        LSTM(128, input_shape=(輸入維度, 1), activation='tanh', return_sequences=False),  
        
        # Dropout 層：以 0.3 的比例隨機丟棄部分神經元，防止過擬合
        Dropout(0.3),
        
        # Dense 層：全連接層，包含 64 個神經元，使用 ReLU 激活函數
        Dense(64, activation='relu'),
        
        # Dropout 層：以 0.2 的比例隨機丟棄部分神經元，防止過擬合
        Dropout(0.2),
        
        # Dense 層：輸出層，包含 2 個神經元，使用 Softmax 激活函數，輸出為分類結果
        Dense(2, activation='softmax')
    ])
    
    模型.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    return 模型

def 繪製混淆矩陣(真實標籤, 預測標籤, 折數=None):
    類別名稱 = ['無心臟病', '有心臟病']
    混淆矩陣 = confusion_matrix(真實標籤, 預測標籤)
    sns.heatmap(混淆矩陣, annot=True, fmt='d', cmap='Blues', xticklabels=類別名稱, yticklabels=類別名稱)
    plt.title(f'混淆矩陣 (第{折數}折)' if 折數 else '混淆矩陣', fontsize=14)
    plt.xlabel('預測標籤', fontsize=12)
    plt.ylabel('真實標籤', fontsize=12)
    plt.show()

# 更新訓練過程中的標準化器處理，儲存和載入 scaler.pkl
def 訓練_LSTM模型(X, y):
    # 初始化
    k折 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    平均準確率, 平均AUC = [], []
    
    # 儲存模型的路徑
    模型儲存路徑 = r"E:\replit2\heart_disease_model.h5"   # 儲存在當前目錄
    scaler儲存路徑 = r"E:\replit2\scaler.pkl"  # 儲存標準化器的路徑

    # 檢查儲存目錄是否存在，若不存在則創建
    if not os.path.exists(os.path.dirname(模型儲存路徑)):
        os.makedirs(os.path.dirname(模型儲存路徑))
    
    if not os.path.exists(os.path.dirname(scaler儲存路徑)):
        os.makedirs(os.path.dirname(scaler儲存路徑))
    
    for 折數, (訓練索引, 驗證索引) in enumerate(k折.split(X, y), 1):
        # 分割數據
        X_訓練, X_驗證 = X[訓練索引], X[驗證索引]
        y_訓練, y_驗證 = y[訓練索引], y[驗證索引]
        
        # 標準化
        標準化器 = StandardScaler()
        X_訓練 = 標準化器.fit_transform(X_訓練).reshape(-1, X.shape[1], 1)
        X_驗證 = 標準化器.transform(X_驗證).reshape(-1, X.shape[1], 1)
        
        # 儲存標準化器（如果是第一次訓練的話）
        if 折數 == 1:  # 在第一次訓練的折數中儲存標準化器
            joblib.dump(標準化器, scaler儲存路徑)
            print(f"標準化器已儲存至 {scaler儲存路徑}")
        
        # 標籤轉為類別型
        y_訓練 = to_categorical(y_訓練, num_classes=2)
        y_驗證 = to_categorical(y_驗證, num_classes=2)
        
        # 建立模型
        模型 = 建立_LSTM模型(X_訓練.shape[1])
        
        # 訓練模型
        history = 模型.fit(
            X_訓練, y_訓練,  # 訓練資料 (X_訓練: 特徵資料, y_訓練: 標籤資料)
            validation_data=(X_驗證, y_驗證),  # 驗證資料 (X_驗證: 驗證特徵資料, y_驗證: 驗證標籤資料)
            epochs=50,  # 訓練的回合數 (500個訓練回合)
            batch_size=32,  # 每個批次的樣本數量 (每次從資料中抽取32個樣本進行訓練)
            verbose=1  # 訓練過程中的顯示模式，1表示顯示進度條和訓練信息
        )

        # 繪製訓練歷史圖表
        plot_training_history(history, 折數)
        
        # 預測
        y_預測 = np.argmax(模型.predict(X_驗證), axis=1)
        y_驗證真值 = np.argmax(y_驗證, axis=1)
        
        # 評估
        準確率 = (y_預測 == y_驗證真值).mean()
        平均準確率.append(準確率)
        
        # AUC
        fpr, tpr, _ = roc_curve(y_驗證真值, y_預測)
        auc值 = auc(fpr, tpr)
        平均AUC.append(auc值)
        
        print(f"Fold {折數} - 準確率: {準確率:.4f}, AUC: {auc值:.4f}")
        
        # 混淆矩陣
        繪製混淆矩陣(y_驗證真值, y_預測, 折數)
    
    # 儲存最終模型（這裡使用最後一個訓練的模型）
    模型.save(模型儲存路徑)
    print(f"\n模型已儲存至 {模型儲存路徑}")
    
    print(f"\n平均準確率: {np.mean(平均準確率):.4f}, 平均AUC: {np.mean(平均AUC):.4f}")
    
# 主程序
if __name__ == "__main__":
    文件路徑 = r"E:\心臟病\heart.csv"  # 資料集路徑
    X, y, 特徵列 = 載入與預處理數據(文件路徑)  # 載入與預處理資料
    
    # 新增這兩行
    print("原始資料類別分佈:")
    print(pd.Series(y).value_counts())
    
    訓練_LSTM模型(X, y)  # 訓練模型




# k折交叉驗證 (StratifiedKFold) 設定為 5 折（n_splits=5），這意味著資料集會被分成 5 個子集
# ，並且會進行 5 次訓練，每一次使用 4/5 的資料作為訓練集，剩下 1/5 的資料作為驗證集。因此，每次訓練時，訓練集佔 80%，驗證集佔 20%。





# 提高模型的準確性可以從以下幾個方面進行調整：

# 數據預處理：

# 特徵標準化：神經網絡模型通常對特徵的範圍和分佈較為敏感，因此對特徵進行標準化或正規化可以提高模型的效果。你可以使用 StandardScaler 或 MinMaxScaler 來標準化特徵數據。
# 處理缺失值：如果數據中有缺失值，應該進行處理。可以選擇丟棄缺失值行，或用均值/中位數等進行填補。
# 模型架構調整：

# 增加層數或神經元：如果模型容量過小，可能無法學到足夠的特徵。你可以嘗試增加神經網絡的層數或每層的神經元數量。
# 改變激活函數：ReLU 是一個很常見的選擇，但有時候可以考慮其他激活函數，如 Leaky ReLU 或 ELU，這些可以避免 ReLU 的「死神經元」問題。
# 使用 Dropout：Dropout 是一種正則化技術，能幫助防止過擬合。你可以在某些層之間加入 Dropout 層。
# 訓練過程調整：

# 增加訓練輪次：2000 輪的訓練可能過多或過少，應該根據損失和準確率的趨勢進行調整。你可以觀察訓練過程中的損失和準確率變化，並進行早停（Early Stopping）來避免過擬合。
# 調整學習率：Adam 優化器的學習率可能需要調整。你可以嘗試不同的學習率，看看哪一個效果最好。
# 批次大小調整：批次大小對訓練速度和結果有影響。你可以試著調整批次大小（例如，32、128）來獲得更好的結果。
# 使用更多的訓練數據：

# 增強數據集：如果數據量不夠多，可以嘗試進行數據增強，例如使用交叉驗證來減少模型的偏差，或者收集更多樣本。
# 合成數據：可以考慮生成合成數據（例如 SMOTE）來平衡樣本數量，尤其是當某些類別的樣本過少時。
