# 1. Konlpy Import First to avoid JVM/DLL conflicts
from konlpy.tag import Okt  # 한국어 형태소 분석기 (명사 추출용)
from collections import Counter  # 단어 빈도수 계산
from wordcloud import WordCloud  # 워드클라우드 시각화 생성
import torch  # 딥러닝 프레임워크 (Transformers 모델 구동용)
from transformers import pipeline  # Hugging Face의 NLP 파이프라인 (감성 분석 등)

import streamlit as st  # 웹 애플리케이션 프레임워크
import pandas as pd  # 데이터프레임 처리 및 조작
import matplotlib.pyplot as plt  # 데이터 시각화 라이브러리
import matplotlib
matplotlib.use('Agg') # Streamlit 환경에서 GUI 에러 방지 (백엔드 설정)
import seaborn as sns  # Matplotlib 기반의 통계적 시각화
from googleapiclient.discovery import build  # Google API 클라이언트 (YouTube Data API 연동)
import re  # 정규표현식 (텍스트 정제용)
import os  # 운영체제 상호작용 (파일 경로 확인 등)

from step_12 import run_step12

# ==========================================
# [개요]
# 이 파일은 Streamlit을 사용한 웹 애플리케이션의 메인 진입점입니다.
# 유튜브 비디오 ID를 입력받아 다음 단계들을 수행합니다:
# 1. step_12.py의 기능을 이용해 영상 요약, 키워드 추출, 주제 분류 (LLM 활용)
# 2. Google YouTube Data API를 통해 영상의 댓글을 수집
# 3. 수집된 댓글에 대해 감성 분석(Sentiment Analysis) 및 형태소 분석 수행
# 4. 분석 결과를 파이 차트, 워드 클라우드 등으로 시각화 및 데이터 다운로드 제공
# ==========================================


# 페이지 설정 (가장 먼저 호출되어야 함)
st.set_page_config(
    page_title="YouTube 댓글 분석기",
    page_icon="🎬",
    layout="wide"
)

# 한글 폰트 설정
import matplotlib.font_manager as fm
font_path = 'C:/Windows/Fonts/malgun.ttf'
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 맥용 한글 폰트
#import matplotlib.font_manager as fm
#font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
#font_name = fm.FontProperties(fname=font_path).get_name()
#plt.rc('font', family=font_name)
#plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 함수 정의 (캐싱 적용)
# ==========================================

@st.cache_resource
def load_sentiment_model():
    """
    [함수] load_sentiment_model
    로컬 디렉토리("./my_model")에 저장된 감성 분석 모델을 로드하여 Hugging Face 파이프라인을 생성합니다.
    @st.cache_resource를 사용하여 모델을 메모리에 한 번만 로드하고 세션 전체에서 공유합니다.
    
    Returns:
        pipeline: Hugging Face의 텍스트 분류(text-classification) 파이프라인 객체
    """
    # 로컬에 저장된 모델 디렉토리("./my_model")에서 로드 (download_model.py로 미리 다운로드 필요)
    return pipeline("text-classification", model="./my_model")


@st.cache_data
def get_video_comments(api_key, video_id, max_results=100):
    """
    [함수] get_video_comments
    YouTube Data API v3를 사용하여 특정 비디오의 댓글을 수집합니다.
    @st.cache_data: 동일한 입력(api_key, video_id 등)에 대해 결과를 캐싱하여API 호출 비용을 절약하고 응답 속도를 높입니다.
    
    Args:
        api_key (str): 구글 개발자 콘솔에서 발급받은 유튜브 API 키
        video_id (str): 댓글을 수집할 유튜브 영상 ID
        max_results (int): 수집할 최대 댓글 수 (기본값 100)
    
    Returns:
        pd.DataFrame: 수집된 댓글 데이터(날짜, 작성자, 내용)가 담긴 데이터프레임
    """

    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        comments = []
        
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_results, 100), # API 한 번 요청 최대 100개
            textFormat="plainText"
        )

        while request and len(comments) < max_results:
            response = request.execute()
            
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                author = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
                date = item['snippet']['topLevelComment']['snippet']['publishedAt']
                comments.append([date, author, comment])
                
            if 'nextPageToken' in response and len(comments) < max_results:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(max_results - len(comments), 100),
                    textFormat="plainText",
                    pageToken=response['nextPageToken']
                )
            else:
                break
        
        pd.DataFrame(comments, columns=['Date', 'Author', 'Comment']).to_csv('test.csv')
        return pd.DataFrame(comments, columns=['Date', 'Author', 'Comment'])
    except Exception as e:
        st.error(f"댓글 수집 중 오류 발생: {e}")
        return pd.DataFrame()

@st.cache_data
def analyze_comments(df):
    """
    [함수] analyze_comments
    수집된 댓글 데이터프레임을 받아 텍스트 전처리, 형태소 분석, 감성 분석을 수행합니다.
    @st.cache_data: 동일한 입력(df)에 대해 결과를 캐싱하여 응답 속도를 높입니다.

    Args:
        df (pd.DataFrame): 'Comment' 컬럼이 포함된 댓글 데이터프레임
        
    Returns:
        result_df (pd.DataFrame): 감성 분석 결과가 추가된 데이터프레임
        all_nouns (list): 워드클라우드 생성을 위한 추출된 모든 명사 리스트
    """
    # 형태소 분석기 초기화 (Konlpy의 Okt 사용)
    okt = Okt()
    
    valid_comments = []
    valid_dates = []
    valid_authors = []
    sentiments = []
    all_nouns = []

    # 진행률 표시줄
    progress_bar = st.progress(0)
    total_rows = len(df)

    for i, row in df.iterrows():
        comment = row['Comment']
        # 정규표현식을 사용하여 한글, 영문, 숫자, 공백을 제외한 특수문자 제거 (이모지 등)
        clean_text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', comment)
        
        if not clean_text.strip():
            continue
            
        # 1. 명사 추출 (워드클라우드 용)
        nouns = okt.nouns(clean_text)
        all_nouns.extend([n for n in nouns if len(n) > 1])

        # 2. AI 감성 분석
        # 모델은 입력 길이 제한(보통 512 토큰)이 있으므로, 안전하게 앞부분 512자만 잘라서 분석
        try:
            result = sentiment_classifier(clean_text[:512])[0] 
            label = result['label']
            score = result['score'] # 모델이 예측한 확률 값 (현재 로직에서는 라벨 결정에만 사용됨)

            if label == 'LABEL_1':
                sentiment = 'Positive'
            else:
                sentiment = 'Negative'
        except Exception:
            sentiment = 'Neutral' # 오류 시 중립 
        
        valid_comments.append(clean_text)
        valid_dates.append(row['Date'])
        valid_authors.append(row['Author'])
        sentiments.append(sentiment)
        
        # 진행률 업데이트
        if (i + 1) % 10 == 0 or (i + 1) == total_rows:
            progress_bar.progress((i + 1) / total_rows)

    progress_bar.empty() # 완료 후 제거

    result_df = pd.DataFrame({
        'Date': valid_dates,
        'Author': valid_authors,
        'Comment': valid_comments,
        'Sentiment': sentiments
    })
    
    return result_df, all_nouns


# 감성 분석 모델 로드 및 초기화
sentiment_classifier = load_sentiment_model()

# ==========================================
# 2. UI 구성
# ==========================================

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    api_key_input = st.text_input("YouTube API Key", value='AIzaSyDD4Kw6X4RlToeRp1YkwJG0LRW6izBr9JU', type="password")
    max_comments = st.slider("수집할 댓글 수", min_value=10, max_value=1000, value=200, step=10)
    st.info("API Key는 기본값이 입력되어 있습니다.")

# 메인 화면
st.title("🎬 AI 기반 YouTube 콘텐츠 자동 분석 시스템")
st.markdown("""
### 📌 프로젝트 개요

**STEP 1** : 유튜브 영상의 내용을 요약합니다.  

**STEP 2** : 요약된 내용을 바탕으로 유튜브 주제를 분류합니다.  

**STEP 3** : 유튜브 영상의 댓글을 수집하여  
- **긍정 / 부정 여론**을 분석하고  
- 주요 키워드를 **워드클라우드**로 시각화합니다.
""")

video_id_input = st.text_input("YouTube Video ID 또는 URL 입력", value="QkGkE9jRX_g")

# URL에서 ID 추출 로직
# URL 파싱 로직: 사용자가 전체 URL을 입력했든, 단축 URL(youtu.be)을 입력했든 ID만 추출
if "youtube.com" in video_id_input or "youtu.be" in video_id_input:
    if "v=" in video_id_input:
        video_id = video_id_input.split("v=")[1].split("&")[0]
    elif "youtu.be" in video_id_input:
        video_id = video_id_input.split("/")[-1]
    else:
        video_id = video_id_input
else:
    video_id = video_id_input

# 분석(step_12)을 위해 전체 URL 재구성
video_url = f'https://www.youtube.com/watch?v={video_id}'

if st.button("분석 시작 🚀", type="primary"):
    if not api_key_input:
        st.error("API Key를 입력해주세요.")
    elif not video_id:
        st.error("Video ID를 입력해주세요.")
    else:
        # 메인 작업 시작
        run_step12(video_url)
        with st.spinner(f"댓글을 수집하고 있습니다... (ID: {video_id})"):
            df = get_video_comments(api_key_input, video_id, max_comments)
        
        if not df.empty:
            st.divider()
            st.subheader("[STEP 3] 댓글 감정 분석")
            st.success(f"총 {len(df)}개의 댓글을 수집했습니다!")
            
            with st.spinner("텍스트 분석 중..."):
                result_df, nouns = analyze_comments(df)
            
            # 1. 감성 분석 결과
            st.divider()
            st.subheader("📊 감성 분석 결과")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                sentiment_counts = result_df['Sentiment'].value_counts()
                fig1, ax1 = plt.subplots()
                colors = ['#ff9999', '#66b3ff', '#99ff99']
                ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
                ax1.axis('equal')  # 원형 유지
                st.pyplot(fig1)
            
            with col2:
                st.write("#### 감성 요약")
                st.dataframe(sentiment_counts, use_container_width=True)
                st.metric("긍정 댓글 수", len(result_df[result_df['Sentiment'] == 'Positive']))
                st.metric("부정 댓글 수", len(result_df[result_df['Sentiment'] == 'Negative']))

            # 2. 워드 클라우드
            st.divider()
            st.subheader("☁️ 주요 키워드 (Word Cloud)")
            if nouns:
                count = Counter(nouns)
                tags = count.most_common(50)
                
                # 폰트 경로 확인 및 예외 처리
                if not os.path.exists(font_path):
                    st.warning(f"폰트 파일을 찾을 수 없습니다: {font_path}")
                    # 기본 폰트 사용 시도 (한글 깨질 수 있음)
                    font_path = None 

                wc = WordCloud(font_path=font_path,
                               background_color='white', 
                               width=800, height=600)
                cloud = wc.generate_from_frequencies(dict(tags))
                
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.imshow(cloud)
                ax2.axis('off')
                st.pyplot(fig2)
            else:
                st.warning("분석할 명사가 충분하지 않습니다.")

            # 3. 원본 데이터 (Expander 사용)
            st.divider()
            with st.expander("📝 수집된 댓글 데이터 보기"):
                st.dataframe(result_df)
                
                # CSV 다운로드 버튼
                csv = result_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="CSV로 다운로드",
                    data=csv,
                    file_name=f'youtube_comments_{video_id}.csv',
                    mime='text/csv',
                )

            # # 4. 시간별 차트
            # result_df['Date'] = pd.to_datetime(result_df['Date'])
            # # 일별(D) 또는 시간별(H)로 리샘플링하여 감성 점수 평균 내기
            # # (Positive=1, Negative=-1, Neutral=0 으로 매핑하여 평균 계산)
            # sentiment_map = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
            # result_df['Score'] = result_df['Sentiment'].map(sentiment_map)
            # daily_sentiment = result_df.set_index('Date').resample('D')['Score'].mean()
            # st.subheader("📈 시간대별 여론 변화")
            # st.line_chart(daily_sentiment)

        else:
            st.warning("댓글을 가져오지 못했습니다. Video ID나 API Key를 확인해주세요.")
