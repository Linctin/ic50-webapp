# app.py
from flask import Flask, request, jsonify, render_template
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import io
import base64
import psycopg2
import os

# --- Matplotlib 中文顯示設定 ---
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

# Flask 應用初始化
app = Flask(__name__)

# --- Database Connection ---
def get_db_connection():
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST", "dpg-d253m93e5dus73f7i4eg-a.oregon-postgres.render.com"),
        database=os.environ.get("DB_NAME", "ic50_5fu_iri_oxa"),
        user=os.environ.get("DB_USER", "ic50_5fu_iri_oxa_user"),
        password=os.environ.get("DB_PASSWORD", "Hnwas25AX7N9JR8p4eeuAX1ewrx9zT80")
    )
    return conn

# --- 四參數邏輯回歸模型 ---
def logistic_4param(x, top, bottom, ic50_log, hillslope):
    return bottom + (top - bottom) / (1 + (10**(x - ic50_log))**hillslope)

# --- 主頁和儀表板路由 ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# --- API Endpoints ---
@app.route('/api/dashboard_data', methods=['GET'])
def get_dashboard_data():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT organoid_id, ic50_5fu, z_score_5fu, ic50_irinotecan, z_score_irinotecan, ic50_oxaliplatin, z_score_oxaliplatin, composite_score FROM organoid_results ORDER BY composite_score DESC")
        data = cur.fetchall()
        cur.close()
        conn.close()
        
        results = []
        for row in data:
            results.append({
                "organoid_id": row[0],
                "ic50_5fu": f"{row[1]:.2f}",
                "z_score_5fu": f"{row[2]:.2f}",
                "ic50_irinotecan": f"{row[3]:.2f}",
                "z_score_irinotecan": f"{row[4]:.2f}",
                "ic50_oxaliplatin": f"{row[5]:.2f}",
                "z_score_oxaliplatin": f"{row[6]:.2f}",
                "composite_score": f"{row[7]:.2f}"
            })
        return jsonify(results)
    except Exception as e:
        print(f"Error fetching dashboard data: {e}")
        return jsonify({"error": "無法獲取儀表板數據。"}), 500

@app.route('/calculate_and_store', methods=['POST'])
def calculate_and_store():
    data = request.json
    organoid_id = data.get('organoid_id')
    experiments = data.get('experiments', [])
    
    if not organoid_id:
        return jsonify({"error": "必須提供 Organoid ID。"}), 400
        
    required_drugs = {"5FU", "Irinotecan", "Oxaliplatin"}
    provided_drugs = {exp['name'] for exp in experiments}
    if not required_drugs.issubset(provided_drugs):
        return jsonify({"error": f"缺少必要的藥物數據，需要: {', '.join(required_drugs)}"}), 400

    calculated_results = {}

    # 1. Calculate IC50 for each drug
    for exp in experiments:
        try:
            concentrations = np.array(exp['concentrations'], dtype=float)
            raw_replicates = [np.array([val for val in r if val is not None], dtype=float) for r in exp['raw_replicates']]
            control_abs = float(exp['control_abs'])
            background_abs = float(exp['background_abs'])

            if (control_abs - background_abs) == 0:
                raise ValueError("對照組和背景值相同。")

            mean_viabilities = [((np.mean(reps) - background_abs) / (control_abs - background_abs)) * 100 for reps in raw_replicates if reps.size > 0]
            
            valid_indices = ~np.isnan(mean_viabilities)
            if np.sum(valid_indices) < 4:
                raise ValueError("有效數據點不足4個。")

            valid_concentrations = concentrations[valid_indices]
            valid_mean_viabilities = np.array(mean_viabilities)[valid_indices]
            log_concentrations = np.log10(valid_concentrations)

            initial_guess = [100, 0, np.mean(log_concentrations), 1.0]
            bounds = ([0, -np.inf, -np.inf, 0.1], [np.inf, np.inf, np.inf, 5.0])
            
            params, _ = curve_fit(logistic_4param, log_concentrations, valid_mean_viabilities, p0=initial_guess, bounds=bounds, maxfev=10000)
            ic50_calculated = 10**params[2]
            calculated_results[exp['name']] = ic50_calculated

        except (ValueError, RuntimeError) as e:
            return jsonify({"error": f"計算 {exp['name']} IC50 時出錯: {e}"}), 400

    # 2. Calculate Z-scores and save to DB
    z_scores = {}
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        for drug_name, ic50_value in calculated_results.items():
            # Save to history
            cur.execute("INSERT INTO ic50_history (drug_name, ic50_value) VALUES (%s, %s)", (drug_name, ic50_value))
            
            # Get historical data for Z-score calculation
            cur.execute("SELECT ic50_value FROM ic50_history WHERE drug_name = %s", (drug_name,))
            history = cur.fetchall()
            
            if len(history) > 1:
                values = [h[0] for h in history]
                mean_val = np.mean(values)
                std_dev = np.std(values)
                if std_dev == 0:
                    z_scores[drug_name] = 0.0
                else:
                    z_scores[drug_name] = (ic50_value - mean_val) / std_dev
            else:
                z_scores[drug_name] = 0.0 # Not enough data for Z-score

        # 3. Calculate composite score and save the final result
        composite_score = sum(z_scores.values())
        
        # Use INSERT ... ON CONFLICT to handle updates
        sql = """
            INSERT INTO organoid_results (organoid_id, ic50_5fu, z_score_5fu, ic50_irinotecan, z_score_irinotecan, ic50_oxaliplatin, z_score_oxaliplatin, composite_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (organoid_id) DO UPDATE SET
                ic50_5fu = EXCLUDED.ic50_5fu,
                z_score_5fu = EXCLUDED.z_score_5fu,
                ic50_irinotecan = EXCLUDED.ic50_irinotecan,
                z_score_irinotecan = EXCLUDED.z_score_irinotecan,
                ic50_oxaliplatin = EXCLUDED.ic50_oxaliplatin,
                z_score_oxaliplatin = EXCLUDED.z_score_oxaliplatin,
                composite_score = EXCLUDED.composite_score,
                created_at = CURRENT_TIMESTAMP;
        """
        cur.execute(sql, (
            organoid_id,
            calculated_results['5FU'], z_scores['5FU'],
            calculated_results['Irinotecan'], z_scores['Irinotecan'],
            calculated_results['Oxaliplatin'], z_scores['Oxaliplatin'],
            composite_score
        ))
        
        conn.commit()
        
        return jsonify({"message": "計算成功並已儲存。", "redirect_url": "/dashboard"})

    except Exception as e:
        conn.rollback()
        print(f"Database error: {e}")
        return jsonify({"error": "數據庫操作失敗。"}), 500
    finally:
        cur.close()
        conn.close()

# 運行 Flask 應用
if __name__ == '__main__':
    app.run(debug=True)