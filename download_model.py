import torch  # 딥러닝 프레임워크 (Transformers 모델 구동용)
from transformers import pipeline  # Hugging Face 모델 다운로드 및 파이프라인 로드

print("모델 다운로드 중... 잠시만 기다려주세요.")

# ==========================================
# [스크립트 설명]
# 이 스크립트는 Hugging Face Hub에서 감성 분석 모델을 다운로드하여
# 로컬 디렉토리('./my_model')에 저장하는 역할을 합니다.
# Streamlit 앱 실행 시 매번 모델을 다운로드하지 않도록, 미리 한 번 실행해두어야 합니다.
# ==========================================

# 1. 모델을 인터넷에서 받아옵니다.
pipe = pipeline("text-classification", model="matthewburke/korean_sentiment")

# 2. 'my_model'이라는 폴더에 저장합니다.
pipe.save_pretrained("./my_model")

print("다운로드 완료! 'my_model' 폴더가 생성되었습니다.")