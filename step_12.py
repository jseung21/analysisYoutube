import streamlit as st  # 웹 애플리케이션 프레임워크 (UI 출력용)
import subprocess  # 외부 프로세스 실행 (OS 명령 등)
import os  # 파일 시스템 접근 및 환경변수 제어
import re  # 정규표현식 (자막 텍스트 정제용)
from openai import OpenAI  # OpenAI API (GPT 모델 사용)
from keybert import KeyBERT  # BERT 기반 키워드 추출 라이브러리
from dotenv import load_dotenv  # .env 파일에서 환경변수 로드
import yt_dlp  # YouTube 영상/자막 다운로드 라이브러리
import yt_dlp  # YouTube 영상/자막 다운로드 라이브러리

# .env 파일 로드
load_dotenv()


# 1. OpenAI Client
# 주의: API Key는 소스코드에 직접 노출하는 것보다 환경 변수(.env) 등으로 관리하는 것이 보안상 좋습니다.
# 현재는 코드 내에 하드코딩 되어 있습니다.
client = OpenAI(api_key="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")

# 2. 유튜브 자막 다운로드 함수
def download_subtitle(video_url):
    """
    [함수] download_subtitle
    yt-dlp 라이브러리를 사용하여 유튜브 영상의 '자동 생성 자막(Korean)'을 다운로드합니다.
    영상 파일은 다운로드하지 않고 자막 파일(.vtt)만 가져옵니다.
    """
    ydl_opts = {
        'writeautomaticsub': True, # 자동 생성된 자막 활성화
        'subtitleslangs': ['ko'],  # 다운로드할 언어 코드 (한국어)
        'skip_download': True,     # 비디오/오디오 파일은 다운로드 스킵
    }
    # yt_dlp 라이브러리 실행
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

# 3. 자막 텍스트 읽기
def load_latest_subtitle_file():
    """
    [함수] load_latest_subtitle_file
    현재 디렉토리에 있는 .vtt 파일 중 가장 최근에 수정된 파일을 찾아 내용을 읽습니다.
    여러 영상의 자막이 섞여 있을 경우를 대비해 가장 최신 파일을 대상으로 합니다.
    """
    vtt_files = [f for f in os.listdir() if f.endswith(".vtt")]

    if not vtt_files:
        return None, None   # 파일이 없을 경우 처리

    # 파일 수정 시간(getmtime)을 기준으로 정렬하여 최신 파일 선택
    latest_file = max(vtt_files, key=os.path.getmtime)

    with open(latest_file, "r", encoding="utf-8") as f:
        subtitle_text = f.read()

    return subtitle_text

#  4. VTT 자막 정제
def clean_vtt_text(vtt_text):
    """
    [함수] clean_vtt_text
    WebVTT 포맷의 자막 파일에는 타임스탬프, 스타일 태그 등이 포함되어 있습니다.
    순수한 대사 내용만 남기기 위해 정규표현식(Regex)을 이용해 불필요한 부분을 제거합니다.
    """
    # 타임스탬프 (00:00:00.000 --> ...) 제거
    text = re.sub(r"\d{2}:\d{2}:\d{2}\.\d+ --> .*", "", vtt_text)

    # VTT 헤더 및 정렬 태그 제거
    text = re.sub(r"webvtt|align:start|position:\d+%", "", text, flags=re.IGNORECASE)

    # HTML 스타일 태그(<c>, <b> 등) 제거
    text = re.sub(r"<.*?>", "", text)

    # 특수문자 제거 (한글, 영문, 숫자, 공백만 남김) - 분석 목적에 따라 조절 가능
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)

    # 중복 공백을 하나의 공백으로 치환
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

# [STEP 1] GPT 영상 요약
def summarize_with_gpt(text):
    """
    OpenAI GPT 모델을 사용하여 긴 자막 텍스트를 5줄로 요약합니다.
    """
    prompt = f"""
다음은 유튜브 영상 자막입니다.
핵심 내용만 5줄로 요약해 주세요.

자막:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# [STEP 2-1] GPT 키워드 추출
def extract_keywords_with_gpt(text):
    prompt = f"""
다음 영상 자막에서 핵심 키워드 5개만 추출해 주세요.
형식: 쉼표(,) 로 구분

자막:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# [STEP 2-2] KeyBERT 키워드 추출
def extract_keywords_with_keybert(text, top_n=5):
    """
    [함수] extract_keywords_with_keybert
    KeyBERT 모델을 사용하여 텍스트에서 키워드를 추출합니다.
    GPT와 달리 임베딩 기반의 통계적 추출 방식을 사용하므로 상호 보완적인 결과를 얻을 수 있습니다.
    
    Args:
        top_n: 추출할 키워드 개수
    """
    kw_model = KeyBERT(model="all-MiniLM-L6-v2") # 가볍고 성능이 좋은 기본 모델 사용

    # 정제된 텍스트 사용
    clean_text = clean_vtt_text(text)

    # keyphrase_ngram_range=(1, 2): 1단어 또는 2단어 조합까지 키워드로 추출
    keywords = kw_model.extract_keywords(
        clean_text,
        keyphrase_ngram_range=(1, 2),
        stop_words=None,
        top_n=top_n
    )

    return keywords

# [STEP 2-3] GPT 영상 주제 분류
def classify_topic_with_gpt(text):
    prompt = f"""
다음 유튜브 영상 자막을 분석해서 아래 형식으로 출력해 주세요.

[출력 형식]
- 대분류 카테고리: (기술 / 경제 / 투자 / 교육 / 사회 / 엔터테인먼트 중 하나)
- 세부 주제: (한 줄 핵심 주제)

자막:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# 5. 전체 실행 (STEP 1 ~ STEP 3)
def run_step12(video_url): 
    """
    [함수] run_step12
    위에서 정의한 모든 단계(자막 다운로드 -> 로드 -> 요약 -> 키워드 -> 주제 분류)를 
    순차적으로 실행하는 메인 파이프라인 함수입니다.
    """
    st.info("▶ 자막 다운로드 중...")
    download_subtitle(video_url)

    subtitle_text = load_latest_subtitle_file()

    if subtitle_text is None:
        st.error("❌ 자막 파일을 찾을 수 없습니다.")
    else:
        # STEP 1
        st.info("▶ [STEP 1] GPT 영상 요약 중...")
        summary = summarize_with_gpt(subtitle_text)

        # STEP 2-1
        st.info("▶ [STEP 2-1] GPT 키워드 추출 중...")
        gpt_keywords = extract_keywords_with_gpt(subtitle_text)

        # STEP 2-2
        st.info("▶ [STEP 2-2] KeyBERT 키워드 추출 중...")
        bert_keywords = extract_keywords_with_keybert(subtitle_text)

        # STEP 2-3
        st.info("▶ [STEP 2-3] GPT 영상 주제 분류 중...")
        topic = classify_topic_with_gpt(subtitle_text)

        # 결과 출력
        st.divider()
        st.subheader("[STEP 1] 영상 요약 결과")
        st.write(summary)

        st.divider()
        st.subheader("[STEP 2-1] GPT 키워드")
        st.write(gpt_keywords)

        st.divider()
        st.subheader("[STEP 2-2] KeyBERT 키워드")
        st.write(bert_keywords)

        st.divider()
        st.subheader("[STEP 2-3] GPT 영상 주제 분류")
        st.write(topic)

        return summary, gpt_keywords, bert_keywords, topic
