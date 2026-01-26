from flask import Flask, request, jsonify
import matplotlib
matplotlib.use('Agg') # [중요] 서버에서 GUI 창 끄기 (반드시 최상단)
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
import os

app = Flask(__name__)

def clean_and_parse_numeric(value):
    """
    문자열에서 괄호 '(', ')'를 제거하고 실수형(float)으로 변환하는 함수
    예: "(123.45)" -> 123.45
    """
    try:
        if isinstance(value, str):
            # 괄호 제거
            clean_str = value.replace('(', '').replace(')', '')
            return float(clean_str)
        return float(value)
    except (ValueError, TypeError):
        return None # 변환 불가능한 경우 (나중에 dropna로 제거)

@app.route('/')
def home():
    return "InviX Server is Running (Robust Mode)!"

@app.route('/upload_data', methods=['POST'])
def upload_data():
    try:
        # 1. 파일 수신 확인
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['file']
        
        # 2. 데이터 읽기 (Pandas)
        # dtype=str: 일단 모든 데이터를 문자로 읽어서 괄호 처리를 직접 함 (안전성 확보)
        # sep=r'\s+': Raw String을 써서 정규식 이스케이프 문제 원천 차단
        df = pd.read_csv(file, sep=r'\s+', comment='#', header=None, dtype=str)
        
        # 3. 데이터 전처리 (괄호 제거 및 숫자 변환)
        # 데이터프레임의 모든 요소에 대해 clean 함수 적용
        for col in df.columns:
            df[col] = df[col].apply(clean_and_parse_numeric)
            
        # 숫자로 변환 못 한 행(None/NaN)이 있으면 제거 (데이터 오염 방지)
        df = df.dropna()

        # 데이터가 비었는지 확인
        if df.empty or df.shape[1] < 2:
            return jsonify({"error": "Invalid data format: Need at least 2 numeric columns"}), 400

        # 4. 그래프 그리기 (XRR 로그 스케일)
        plt.figure(figsize=(6, 6), dpi=100)
        
        # X축: 0번 컬럼, Y축: 1번 컬럼 (보통 XRR 포맷)
        plt.semilogy(df.iloc[:, 0], df.iloc[:, 1], 'b-', linewidth=1.5, label='Experimental Data')
        
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend(loc='upper right')
        plt.xlabel("Angle / q")
        plt.ylabel("Intensity (log)")
        plt.tight_layout()
        
        # 5. 이미지 인코딩
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        # 메모리 정리 (중요: 메모리 누수 방지)
        plt.close('all')
        buf.close()
        
        return jsonify({
            "status": "success",
            "image_base64": image_base64
        })

    except Exception as e:
        # 에러 발생 시에도 메모리 정리 시도
        plt.close('all')
        return jsonify({"error": f"Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)