# app.py
from flask import Flask, request, jsonify, render_template
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import io
import base64

# Flask 應用初始化
app = Flask(__name__)

# --- 四參數邏輯回歸模型 ---
# 這個函數定義了S形曲線的數學模型
def logistic_4param(x, top, bottom, ic50_log, hillslope):
    """
    x: log10(濃度) - 輸入參數必須是對數濃度
    top: 曲線頂部平台值 (e.g., 100% 活力)
    bottom: 曲線底部平台值 (e.g., 0% 活力)
    ic50_log: log10(IC50) - 半數抑制濃度的對數值
    hillslope: 希爾斜率 (Hill Slope) - 描述曲線陡峭程度
    """
    # 這是標準的四參數邏輯函數公式
    return bottom + (top - bottom) / (1 + (10**(x - ic50_log))**hillslope)

# --- 網頁路由與邏輯 ---

# 定義首頁路由，當使用者訪問網站根目錄時顯示 index.html
@app.route('/')
def index():
    return render_template('index.html')

# 定義 IC50 計算路由，當前端發送數據到這個路徑時執行計算
@app.route('/calculate_ic50', methods=['POST']) # 只接受 POST 請求
def calculate_ic50():
    try:
        # 從前端接收 JSON 格式的數據
        data = request.json

        # 檢查關鍵數據是否存在
        if not all(k in data for k in ['concentrations', 'raw_replicates', 'control_abs', 'background_abs']):
            raise ValueError("缺少必要的數據 (濃度、重複數據、對照組、背景值)。")

        # 提取並轉換數據類型
        concentrations = np.array(data['concentrations'], dtype=float)
        # raw_replicates 是一個包含多個子列表的列表，每個子列表代表一個濃度的所有重複值
        raw_replicates = [np.array(r, dtype=float) for r in data['raw_replicates']]
        control_abs = float(data['control_abs'])
        background_abs = float(data['background_abs'])

        # --- 1. 數據處理：計算平均值和正規化為細胞活力百分比 ---
        mean_viabilities = []
        stds = [] # 用於誤差棒的標準差
        for reps in raw_replicates:
            if len(reps) == 0:
                # 處理沒有重複數據的情況，可以給一個默認值或跳過
                mean_viabilities.append(np.nan) # 使用 NaN 表示無效數據
                stds.append(np.nan)
                continue

            mean_abs = np.mean(reps) # 計算重複的平均吸光值
            std_abs = np.std(reps)   # 計算重複的標準差

            # 正規化為百分比活力
            if (control_abs - background_abs) == 0:
                # 避免除以零的錯誤
                raise ValueError("對照組和背景值不能相同，無法進行正規化。")
            
            normalized_viability = ((mean_abs - background_abs) / (control_abs - background_abs)) * 100
            mean_viabilities.append(normalized_viability)
            
            # 轉換標準差到正規化後的尺度
            std_viability = (std_abs / (control_abs - background_abs)) * 100
            stds.append(std_viability)
        
        # 過濾掉 NaN 值，只使用有效數據進行擬合和繪圖
        valid_indices = ~np.isnan(mean_viabilities)
        valid_concentrations = concentrations[valid_indices]
        valid_mean_viabilities = np.array(mean_viabilities)[valid_indices]
        valid_stds = np.array(stds)[valid_indices]

        if len(valid_concentrations) < 4: # 4參數模型至少需要4個點
             raise ValueError("數據點不足，無法進行擬合 (至少需要4個有效濃度點)。")

        log_concentrations = np.log10(valid_concentrations)

        # --- 2. 擬合 4PL 模型 ---
        # 初始參數估計：這對擬合成功很重要，可以根據經驗設置
        # top: 曲線頂部，通常接近 100 (無抑制時的活力)
        # bottom: 曲線底部，通常接近 0 (最大抑制時的活力)
        # ic50_log: IC50 的對數值，初估可以取濃度範圍的中間值對數
        # hillslope: 希爾斜率，通常在 0.5 到 2 之間，初估為 1
        
        # 更好的初始估計：
        # top 設為數據中的最大活力值
        # bottom 設為數據中的最小活力值
        # ic50_log 設為 (min_log_conc + max_log_conc) / 2
        
        # 確保初始估計值合理
        initial_top = np.max(valid_mean_viabilities) if np.max(valid_mean_viabilities) > 0 else 100
        initial_bottom = np.min(valid_mean_viabilities) if np.min(valid_mean_viabilities) < 100 else 0
        initial_ic50_log = np.mean(log_concentrations) # 簡單取平均
        
        initial_guess = [initial_top, initial_bottom, initial_ic50_log, 1.0]

        # 執行曲線擬合
        # bounds 參數可以限制參數的範圍，有助於擬合穩定性
        # 例如：top 0-200, bottom 0-100, ic50_log 任意, hillslope 0.1-5.0
        bounds = ([0, 0, -np.inf, 0.1], [200, 100, np.inf, 5.0])
        
        params, covariance = curve_fit(logistic_4param, log_concentrations, valid_mean_viabilities,
                                       p0=initial_guess, bounds=bounds, maxfev=5000) # 增加最大迭代次數

        # 提取擬合參數
        top_fit, bottom_fit, ic50_log_fit, hillslope_fit = params

        # 計算 IC50 (將 log10(IC50) 轉換回來)
        ic50_calculated = 10**ic50_log_fit

        # --- 3. 生成圖表 ---
        plt.figure(figsize=(10, 6)) # 設定圖片大小

        # 繪製實驗數據點 (帶有誤差棒)
        plt.errorbar(log_concentrations, valid_mean_viabilities, yerr=valid_stds, fmt='o', capsize=4, label='(Mean ± SD)')

        # 生成平滑的擬合曲線數據點
        x_fit = np.linspace(min(log_concentrations) - 0.5, max(log_concentrations) + 0.5, 200) # 擴大範圍讓曲線更完整
        y_fit = logistic_4param(x_fit, *params)
        plt.plot(x_fit, y_fit, color='red', label='4PL Fitting curve')

        # 標記 IC50 點
        plt.axvline(ic50_log_fit, color='green', linestyle='--', linewidth=1, label=f'IC50 ({ic50_calculated:.2f})')
        plt.axhline((top_fit + bottom_fit) / 2, color='blue', linestyle=':', linewidth=1, label='50% Response')

        # 設定圖表標籤和標題
        plt.title('IC50')
        plt.xlabel('Log$_{10}$ ')
        plt.ylabel('Viability (%)')
        plt.grid(True, which="both", ls="--", c='0.7') # 顯示網格線
        plt.legend() # 顯示圖例
        plt.tight_layout() # 自動調整佈局，防止標籤重疊

        # 將 Matplotlib 圖表轉換為 Base64 編碼的 PNG 圖片，以便在網頁上顯示
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0) # 將緩衝區指針移回開頭
        img_b64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        plt.close() # 關閉圖表以釋放記憶體

        # 返回 JSON 格式的結果給前端
        return jsonify({
            'ic50': f'{ic50_calculated:.3f}', # 保留三位小數
            'top': f'{top_fit:.2f}',
            'bottom': f'{bottom_fit:.2f}',
            'hillslope': f'{hillslope_fit:.2f}',
            'plot_image': f'data:image/png;base64,{img_b64}' # 圖片數據
        })

    except ValueError as ve:
        # 捕獲自定義的數據錯誤
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        # 捕獲其他所有未預期的錯誤
        print(f"發生未預期錯誤: {e}") # 在伺服器端打印錯誤日誌
        return jsonify({'error': f'計算發生錯誤：{str(e)}'}), 500

# 運行 Flask 應用
if __name__ == '__main__':
    app.run(debug=True) # debug=True 模式會讓伺服器在程式碼變動時自動重載，適合開發