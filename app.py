import os  # 載入操作系統功能模組，主要用於處理檔案和目錄
import pandas as pd  # 載入 pandas 库，用於資料處理和分析
import requests  # 載入 requests 库，用於發送 HTTP 請求
from io import StringIO  # 載入 StringIO，用於處理字符串流
import numpy as np  # 載入 numpy 库，用於數據處理與運算
import chardet  # 載入 chardet 库，用於檢測字元編碼
import joblib  # 載入 joblib 库，用於儲存和載入模型
from tensorflow import keras  # 載入 TensorFlow 的 Keras API 用於神經網絡建模
from flask import Flask, render_template, jsonify, request  # 載入 Flask 庫及其用於網頁應用的功能
import logging  # 載入 logging 库，用於記錄日誌信息


# 設置日誌級別為 DEBUG
logging.basicConfig(level=logging.DEBUG)

# 初始化 Flask 應用
app = Flask(__name__, template_folder='templates')
basedir = os.path.abspath(os.path.dirname(__file__))  # 獲取當前文件夾的絕對路徑

# 加載模型和標準化器的路徑
model_path = os.path.join(basedir, 'heart_disease_model.h5')
scaler_path = os.path.join(basedir, 'scaler.pkl')

try:
    # 嘗試加載深度學習模型和標準化器
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    logging.info("模型和標準化器成功加載")
except Exception as e:
    # 如果加載失敗，記錄錯誤信息
    logging.error(f"加載模型或標準化器時發生錯誤: {str(e)}")
    model = None
    scaler = None



# 預測心臟病
def predict_heart_disease(features):
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)
    return 1 if prediction[0] > 0.5 else 0

# 嘗試載入模型與縮放器
def focal_loss_fixed(y_true, y_pred, gamma=2.0, alpha=0.25):
    """自定義 focal loss 函數，用於處理不平衡資料集"""
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    fl = -alpha_t * tf.pow((1 - p_t), gamma) * tf.math.log(tf.clip_by_value(p_t, 1e-8, 1.0))
    return tf.reduce_mean(fl)

# 註冊 focal loss 函數，讓 Keras 可以識別
keras.utils.get_custom_objects()['focal_loss_fixed'] = focal_loss_fixed





# 定義特徵預測名稱
features = [
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



# 取得CSV資料的函式
def get_csv_data(url):
    """根據 URL 獲取並處理 CSV 數據"""
    try:
        response = requests.get(url)  # 發送 HTTP 請求
        response.raise_for_status()  # 若請求不成功，則拋出異常

        # 偵測編碼並解碼內容
        encoding = chardet.detect(response.content)['encoding']
        csv_content = response.content.decode(encoding)

        # 將 CSV 內容載入到 DataFrame
        df = pd.read_csv(StringIO(csv_content))
        df.columns = df.columns.str.strip()  # 去除欄位名稱的空格
        df.replace({np.nan: None}, inplace=True)  # 替換 NaN 為 None
        return df
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error: {req_err}")
    except pd.errors.EmptyDataError:
        logging.warning("No data returned from the URL.")
    except Exception as e:
        logging.error(f"Error processing CSV data: {e}")

    return pd.DataFrame()  # 若發生錯誤，返回空的 DataFrame


# 取得最新雲端資料
def get_latest_heart_rate():
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSULVwFdSh9_HuIJe1dWPae-jzcQYsyYb5DuRfXHtDenUlr1oSYTRr-AQ-aMthcCsNRTcVIbvmt_7qJ/pub?output=csv"
    df = get_csv_data(url)  # 獲取心跳數據的 CSV 內容
    if not df.empty:  # 如果 DataFrame 不為空
        latest = df.iloc[-1].to_dict()  # 獲取最後一行數據並轉為字典
        if '心跳_正常_異常' in latest:  # 如果字典中包含 '心跳_正常_異常' 鍵
            latest['心跳_正常_異常'] = '正常' if latest['心跳_正常_異常'] in [0, '0'] else '異常'  # 判斷心跳狀況
        return latest  # 返回最新的心跳數據
    return {}  # 若 DataFrame 為空，返回空字典

def get_latest_dht11():
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS5C_o47POhPXTZEgq40budOJB1ygTTZx9D_086I-ZbHfApFPZB_Ra5Xi09Qu6hxzk9_QXJ-7-QFoKD/pub?output=csv"
    df = get_csv_data(url)  # 獲取 DHT11 數據的 CSV 內容
    if not df.empty:  # 如果 DataFrame 不為空
        latest = df.iloc[-1].to_dict()  # 獲取最後一行數據並轉為字典
        for key in ['溫度_正常_異常', '濕度_正常_異常', '體溫_正常_異常']:  # 遍歷每個狀況鍵
            if key in latest:  # 如果字典中包含該鍵
                latest[key] = '正常' if latest[key] in [0, '0'] else '異常'  # 判斷狀況
        return latest  # 返回最新的 DHT11 數據
    return {}  # 若 DataFrame 為空，返回空字典

def get_latest_bmi():
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTbj3f0rhEu2aCljm1AgkPiaqU7XLGfLUfmL_3NVClYABWXmarViEg1RSE4Q9St0YG_rR74VZyNh7MF/pub?output=csv"
    df = get_csv_data(url)  # 獲取 BMI 數據的 CSV 內容
    if not df.empty:  # 如果 DataFrame 不為空
        latest = df.iloc[-1].to_dict()  # 獲取最後一行數據並轉為字典
        if '正常_異常' in latest:  # 如果字典中包含 '正常_異常' 鍵
            latest['正常_異常'] = '正常' if latest['正常_異常'] in [0, '0'] else '異常'  # 判斷 BMI 狀況
        return latest  # 返回最新的 BMI 數據
    return {}  # 若 DataFrame 為空，返回空字典

def get_historical_data():
    heart_rate_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSULVwFdSh9_HuIJe1dWPae-jzcQYsyYb5DuRfXHtDenUlr1oSYTRr-AQ-aMthcCsNRTcVIbvmt_7qJ/pub?output=csv"
    dht11_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS5C_o47POhPXTZEgq40budOJB1ygTTZx9D_086I-ZbHfApFPZB_Ra5Xi09Qu6hxzk9_QXJ-7-QFoKD/pub?output=csv"
    bmi_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTbj3f0rhEu2aCljm1AgkPiaqU7XLGfLUfmL_3NVClYABWXmarViEg1RSE4Q9St0YG_rR74VZyNh7MF/pub?output=csv"

    heart_rate_df = get_csv_data(heart_rate_url)  # 獲取心跳歷史數據的 DataFrame
    dht11_df = get_csv_data(dht11_url)  # 獲取 DHT11 歷史數據的 DataFrame
    bmi_df = get_csv_data(bmi_url)  # 獲取 BMI 歷史數據的 DataFrame

    if not heart_rate_df.empty and not dht11_df.empty and not bmi_df.empty:  # 如果所有 DataFrame 都不為空
        # 檢查並統一列名
        time_column_names = ['時間戳記', 'timestamp', 'time']

        for df in [heart_rate_df, dht11_df, bmi_df]:  # 遍歷每個 DataFrame
            for col in time_column_names:  # 遍歷時間列名
                if col in df.columns:  # 如果 DataFrame 包含該列名
                    df.rename(columns={col: '時間戳記'}, inplace=True)  # 將該列名統一為 '時間戳記'
                    break
            else:
                print(
                    f"Warning: No time column found in DataFrame with columns: {df.columns}"
                )  # 如果沒有找到時間列名，打印警告信息
                return []

        try:
            # 根據 '時間戳記' 合併 DataFrames
            merged_df = pd.merge(heart_rate_df,
                                 dht11_df,
                                 on='時間戳記',
                                 how='inner')
            merged_df = pd.merge(merged_df, bmi_df, on='時間戳記', how='inner')
        except KeyError as e:
            print(f"Error during merge: {e}")  # 合併過程中出現錯誤，打印錯誤信息
            return []

        # 選擇相關列並按時間戳記排序
        merged_df = merged_df[['時間戳記', '心跳', '溫度', '濕度', '體溫', 'BMI']]
        merged_df = merged_df.sort_values('時間戳記', ascending=False).head(10)  # 取最新的 10 條記錄
        return merged_df.to_dict('records')  # 將結果轉換為字典列表
    return []  # 若任一 DataFrame 為空，返回空列表




# 預測心臟病
def predict_heart_disease(features):
    """預測心臟病：標準化特徵並使用模型進行預測"""
    try:
        if model is None or scaler is None:
            raise Exception("模型或標準化器未加載成功")
        
        # 日誌：顯示輸入特徵
        logging.debug(f"輸入特徵: {features}")

        scaled_features = scaler.transform([features])  # 使用標準化器縮放特徵
        prediction = model.predict(scaled_features)  # 使用模型進行預測
        
        # 日誌：顯示預測結果
        logging.debug(f"預測結果: {prediction}")

        return 1 if prediction[0] > 0.5 else 0  # 0表示無心臟病，1表示有心臟病
    except Exception as e:
        logging.error(f"預測心臟病時發生錯誤: {str(e)}")
        return None


# 首頁路由
@app.route('/')
def index():
    latest_heart_rate = get_latest_heart_rate()  # 獲取最新的心跳數據
    latest_dht11 = get_latest_dht11()  # 獲取最新的 DHT11 數據
    latest_bmi = get_latest_bmi()  # 獲取最新的 BMI 數據
    historical_data = get_historical_data()  # 獲取歷史數據

    # 渲染 HTML 模板並傳遞數據
    return render_template('index.html',
                           heart_rate=latest_heart_rate,
                           dht11=latest_dht11,
                           bmi=latest_bmi,
                           historical_data=historical_data)

# 這裡是原先的 index 函數，應該刪除或重新命名
# @app.route('/')
# def index():
#     return render_template('index.html')


# 取得最新資料的路由


@app.route('/get_latest_data')
def get_latest_data():
    try:
        latest_heart_rate = get_latest_heart_rate()  # 獲取最新的心跳數據
        latest_dht11 = get_latest_dht11()  # 獲取最新的 DHT11 數據
        latest_bmi = get_latest_bmi()  # 獲取最新的 BMI 數據
        historical_data = get_historical_data()  # 獲取歷史數據

        return jsonify({
            'heart_rate': latest_heart_rate,  # 返回心跳數據
            'dht11': latest_dht11,  # 返回 DHT11 數據
            'bmi': latest_bmi,  # 返回 BMI 數據
            'historical_data': historical_data  # 返回歷史數據
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # 返回錯誤信息及 HTTP 500 狀態碼
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 將輸入特徵轉換為數值
        inputs = np.array([
            float(request.form['age']),
            int(request.form['sex']),
            int(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            int(request.form['fbs']),
            int(request.form['restecg']),
            float(request.form['thalach']),
            int(request.form['exang']),
            float(request.form['oldpeak']),
            int(request.form['slope']),
            int(request.form['ca']),
            int(request.form['thal'])
        ]).reshape(1, -1)

        # 使用標準化器縮放特徵
        scaled_inputs = scaler.transform(inputs)

        # 預測並獲取具體概率
        prediction_proba = model.predict(scaled_inputs)[0][0]

        # 風險分層
        def get_risk_level(proba):
            if proba < 0.3:
                return '低風險'
            elif 0.3 <= proba < 0.6:
                return '中等風險'
            else:
                return '高風險'

        # 風險建議
        def get_risk_advice(risk_level):
            advice_map = {
                '低風險': '建議定期體檢，保持健康生活方式',
                '中等風險': '建議進一步醫學評估，調整生活習慣',
                '高風險': '強烈建議立即就醫，進行專業心臟檢查'
            }
            return advice_map[risk_level]

        # 確定最終預測
        final_prediction = 1 if prediction_proba > 0.3 else 0
        risk_level = get_risk_level(prediction_proba)
        risk_advice = get_risk_advice(risk_level)

        return jsonify({
            'prediction': '檢測到心臟病，請諮詢醫生。' if final_prediction == 1 else '未檢測到心臟病，請保持健康。',
            'risk_score': int(final_prediction),
            'risk_probability': float(prediction_proba*100),
            'risk_level': risk_level,
            'risk_advice': risk_advice
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    logging.info("啟動Flask應用程式")
    app.run(debug=True, host='0.0.0.0', port=5000)
