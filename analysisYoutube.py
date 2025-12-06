# -*- coding: utf-8 -*-
# 1. Konlpy Import First to avoid JVM/DLL conflicts
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Streamlit 환경에서 GUI 에러 방지
import seaborn as sns
from googleapiclient.discovery import build
import re
import os

# 페이지 설정 (가장 먼저 호출되어야 함)
st.set_page_config(
    page_title="YouTube 댓글 분석기",
    page_icon="🎬",
    layout="wide"
)

# 한글 폰트 설정
import matplotlib.font_manager as fm
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 함수 정의 (캐싱 적용)
# ==========================================

@st.cache_data
def get_video_comments(api_key, video_id, max_results=100):
    """YouTube API를 통해 댓글을 수집합니다."""
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
                
        return pd.DataFrame(comments, columns=['Date', 'Author', 'Comment'])
    except Exception as e:
        st.error(f"댓글 수집 중 오류 발생: {e}")
        return pd.DataFrame()

@st.cache_data
def analyze_comments(df):
    """수집된 댓글을 분석하여 감성 점수와 명사를 추출합니다."""
    okt = Okt()
    
    positive_keywords = ['좋다', '멋지다', '최고', '응원', '사랑', '재미', '감동', '꿀팁', '성공']
    negative_keywords = ['싫다', '최악', '노잼', '반대', '실망', '우려', '쓰레기', '별로', '화남']

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
        clean_text = re.sub(r'[^가-힣\s]', '', comment) 
        
        if not clean_text.strip():
            continue
            
        nouns = okt.nouns(clean_text)
        all_nouns.extend([n for n in nouns if len(n) > 1])
        
        score = 0
        for word in clean_text.split():
            if any(pos in word for pos in positive_keywords):
                score += 1
            elif any(neg in word for neg in negative_keywords):
                score -= 1
        
        if score > 0: sentiment = 'Positive'
        elif score < 0: sentiment = 'Negative'
        else: sentiment = 'Neutral'
        
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

# ==========================================
# 2. UI 구성
# ==========================================

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    api_key_input = st.text_input("YouTube API Key", value='AIzaSyDQsGvOtDZfe6nFDdjcxZkybcpKTJ9Z-BI', type="password")
    max_comments = st.slider("수집할 댓글 수", min_value=10, max_value=1000, value=200, step=10)
    st.info("API Key는 기본값이 입력되어 있습니다.")

# 메인 화면
st.title("🎬 YouTube 댓글 감성 분석기")
st.markdown("""
유튜브 영상의 댓글을 수집하여 **긍정/부정 여론**을 분석하고, 
주요 키워드를 **워드클라우드**로 시각화합니다.
""")

video_id_input = st.text_input("YouTube Video ID 또는 URL 입력", value="QkGkE9jRX_g")

# URL에서 ID 추출 로직
if "youtube.com" in video_id_input or "youtu.be" in video_id_input:
    if "v=" in video_id_input:
        video_id = video_id_input.split("v=")[1].split("&")[0]
    elif "youtu.be" in video_id_input:
        video_id = video_id_input.split("/")[-1]
    else:
        video_id = video_id_input
else:
    video_id = video_id_input

if st.button("분석 시작 🚀", type="primary"):
    if not api_key_input:
        st.error("API Key를 입력해주세요.")
    elif not video_id:
        st.error("Video ID를 입력해주세요.")
    else:
        with st.spinner(f"댓글을 수집하고 있습니다... (ID: {video_id})"):
            df = get_video_comments(api_key_input, video_id, max_comments)
        
        if not df.empty:
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
                font_path = 'C:/Windows/Fonts/malgun.ttf'
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
        else:
            st.warning("댓글을 가져오지 못했습니다. Video ID나 API Key를 확인해주세요.")
