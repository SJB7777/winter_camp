# app.py
from flask import Flask, request, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "InviX Server is Running!"

@app.route('/upload_data', methods=['POST'])
def upload_data():
    try:
        # 1. n8n에서 보낸 파일 받기
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['file']
        
        # 2. 데이터 처리 (로그 스케일 그래프 그리기)
        df = pd.read_csv(file, sep='\s+', comment='#', header=None)
        
        plt.figure(figsize=(6, 6), dpi=100)
        plt.semilogy(df.iloc[:, 0], df.iloc[:, 1], 'b-', label='Data') # Semilogy 필수
        plt.legend()
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return jsonify({
            "status": "success",
            "image_base64": image_base64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)