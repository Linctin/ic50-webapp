# app.py
from flask import Flask, request, jsonify, render_template
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import io
import base64

# --- Matplotlib 中文顯示設定 ---
# 使用支援中文的字體，例如 'Microsoft JhengHei' for Windows
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
# 解決負號顯示問題
plt.rcParams['axes.unicode_minus'] = False

# Flask 應用初始化
app = Flask(__name__)

# --- 四參數邏輯回歸模型 ---
def logistic_4param(x, top, bottom, ic50_log, hillslope):
    return bottom + (top - bottom) / (1 + (10**(x - ic50_log))**hillslope)

# --- 網頁路由與邏輯 ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate_ic50', methods=['POST'])
def calculate_ic50_combined():
    try:
        all_experiments_data = request.json.get('experiments', [])
        if not all_experiments_data:
            raise ValueError("沒有提供任何實驗數據。")

        results = []
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_experiments_data)))

        for i, experiment in enumerate(all_experiments_data):
            experiment_name = experiment.get('name', f'Experiment {i+1}')
            
            if not all(k in experiment for k in ['concentrations', 'raw_replicates', 'control_abs', 'background_abs']):
                raise ValueError(f"實驗 '{experiment_name}' 缺少必要的數據。")

            concentrations = np.array(experiment['concentrations'], dtype=float)
            raw_replicates = [np.array([val for val in r if val is not None], dtype=float) for r in experiment['raw_replicates']]
            control_abs = float(experiment['control_abs'])
            background_abs = float(experiment['background_abs'])

            if (control_abs - background_abs) == 0:
                raise ValueError(f"實驗 '{experiment_name}' 的對照組和背景值相同，無法正規化。")

            mean_viabilities = []
            stds = []
            for reps in raw_replicates:
                if reps.size == 0:
                    mean_viabilities.append(np.nan)
                    stds.append(np.nan)
                    continue

                mean_abs = np.mean(reps)
                std_abs = np.std(reps)
                
                normalized_viability = ((mean_abs - background_abs) / (control_abs - background_abs)) * 100
                std_viability = (std_abs / (control_abs - background_abs)) * 100
                
                mean_viabilities.append(normalized_viability)
                stds.append(std_viability)

            valid_indices = ~np.isnan(mean_viabilities)
            if np.sum(valid_indices) < 4:
                raise ValueError(f"實驗 '{experiment_name}' 的有效數據點不足4個，無法擬合。")

            valid_concentrations = concentrations[valid_indices]
            valid_mean_viabilities = np.array(mean_viabilities)[valid_indices]
            valid_stds = np.array(stds)[valid_indices]
            
            log_concentrations = np.log10(valid_concentrations)

            initial_top = np.max(valid_mean_viabilities) if np.any(valid_mean_viabilities) else 100
            initial_bottom = np.min(valid_mean_viabilities) if np.any(valid_mean_viabilities) else 0
            initial_ic50_log = np.mean(log_concentrations)
            initial_guess = [initial_top, initial_bottom, initial_ic50_log, 1.0]
            
            bounds = ([0, -np.inf, -np.inf, 0.1], [np.inf, np.inf, np.inf, 5.0])

            try:
                params, _ = curve_fit(logistic_4param, log_concentrations, valid_mean_viabilities,
                                      p0=initial_guess, bounds=bounds, maxfev=10000)
            except RuntimeError:
                raise ValueError(f"實驗 '{experiment_name}' 無法找到最佳擬合曲線，請檢查數據。")

            top_fit, bottom_fit, ic50_log_fit, hillslope_fit = params
            ic50_calculated = 10**ic50_log_fit

            results.append({
                'name': experiment_name,
                'ic50': f'{ic50_calculated:.3f}',
                'top': f'{top_fit:.2f}',
                'bottom': f'{bottom_fit:.2f}',
                'hillslope': f'{hillslope_fit:.2f}'
            })

            color = colors[i]
            plt.errorbar(log_concentrations, valid_mean_viabilities, yerr=valid_stds, fmt='o', capsize=4, color=color)
            
            x_fit = np.linspace(min(log_concentrations) - 0.5, max(log_concentrations) + 0.5, 200)
            y_fit = logistic_4param(x_fit, *params)
            plt.plot(x_fit, y_fit, color=color, linestyle='-', label=experiment_name)

        # --- 設定組合圖表 ---
        plt.title('IC50 劑量反應曲線')
        plt.xlabel('Log$_{10}$ (藥物濃度)')
        plt.ylabel('Viability (%)')
        plt.ylim(0, 150)
        plt.legend(loc='best', fontsize='small')
        plt.tight_layout()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        plt.close()
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        return jsonify({
            'results': results,
            'plot_image': f'data:image/png;base64,{img_b64}'
        })

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"發生未預期錯誤: {e}")
        return jsonify({'error': f'計算發生錯誤：{str(e)}'}), 500

# 運行 Flask 應用
if __name__ == '__main__':
    app.run(debug=True)
