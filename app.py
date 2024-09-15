import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model.analyze_diary import analyze_diary

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False
app.json.ensure_ascii = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 환경변수 설정
PORT = int(os.getenv('PORT', 5001))
CHECKPOINT_PATH = os.getenv('CHECKPOINT_PATH', './checkpoints/checkpoint_epoch_8.pth')

# GPU/CPU 설정
device = torch.device("cpu")

# 모델 불러오기
model_name = "beomi/KcELECTRA-base-v2022"

# 모델 설정
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
tokenizer = AutoTokenizer.from_pretrained(model_name)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# 감정 분석 엔드포인트
@app.route('/predict', methods=['POST'])
def predict_emotion():
    try:
        # 클라이언트로부터 JSON 데이터 받기
        data = request.get_json()

        # 클라이언트가 보낸 일기 텍스트 추출
        diary_text = data.get('diary')

        if not diary_text:
            return jsonify({'error': 'No diary text provided'}), 400

        # 일기 텍스트 분석 (analyze_diary 함수 사용)
        top_3_emotions = analyze_diary(model, tokenizer, diary_text)

        logging.info(f"Processed diary. Length: {len(diary_text)}")

        # 결과 반환
        return jsonify({
            '분석 결과': [{emotion: round(prob, 2)} for emotion, prob in top_3_emotions]
        })

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 서버 실행
if __name__ == '__main__':
    logging.info(f"Starting server on port {PORT}")
    app.run(host='0.0.0.0', port=PORT)