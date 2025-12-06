# YouTube 댓글 감성 분석기 (YouTube Comment Sentiment Analyzer)

이 프로젝트는 **Streamlit**을 기반으로 구축된 웹 애플리케이션으로, YouTube 동영상의 댓글을 수집하고 자연어 처리(NLP)를 통해 감성 분석 및 주요 키워드를 시각화하여 보여줍니다.

## 📋 주요 기능

1.  **웹 인터페이스 (GUI)**: Streamlit을 사용하여 직관적인 웹 환경에서 분석을 수행할 수 있습니다.
2.  **실시간 댓글 수집**: YouTube Data API를 통해 사용자가 입력한 동영상 ID(또는 URL)의 댓글을 실시간으로 수집합니다.
3.  **감성 분석**: `Konlpy`의 `Okt` 형태소 분석기를 활용하여 댓글의 긍정/부정/중립 감성을 분석하고 파이 차트로 시각화합니다.
4.  **워드 클라우드**: 댓글에서 가장 많이 언급된 명사를 추출하여 워드 클라우드로 시각화합니다.
5.  **데이터 다운로드**: 분석된 결과 데이터를 CSV 파일로 다운로드할 수 있습니다.
6.  **사용자 설정**: 사이드바를 통해 API Key 및 수집할 댓글 수를 조절할 수 있습니다.

## 🛠️ 필요 환경 및 라이브러리

이 애플리케이션을 실행하기 위해서는 Python 3.x 환경과 다음 라이브러리들이 필요합니다. 또한 `Konlpy` 실행을 위해 **Java (JDK)** 가 설치되어 있어야 합니다.

### 필수 라이브러리 설치
```bash
pip install streamlit konlpy wordcloud pandas matplotlib seaborn google-api-python-client
```

### 시스템 요구사항
*   **Java (JDK)**: `Konlpy`는 Java 기반이므로 시스템에 Java가 설치되어 있고 환경 변수(`JAVA_HOME`)가 설정되어 있어야 합니다.
*   **한글 폰트**: 워드 클라우드 및 그래프 출력을 위해 시스템에 한글 폰트(예: 맑은 고딕 `Malgun Gothic`)가 필요합니다.

## 🚀 사용 방법

1.  **Google Cloud Console 설정**:
    *   Google Cloud Console에서 프로젝트를 생성하고 **YouTube Data API v3**를 활성화합니다.
    *   API Key를 발급받습니다. (기본값이 코드에 포함되어 있으나, 본인의 키를 사용하는 것을 권장합니다.)

2.  **애플리케이션 실행**:
    터미널에서 다음 명령어를 실행하여 Streamlit 서버를 시작합니다.

    ```bash
    streamlit run analysisYoutube.py
    ```

3.  **웹 브라우저 접속**:
    명령어 실행 후 자동으로 브라우저가 열리거나, 터미널에 표시된 로컬 주소(예: `http://localhost:8501`)로 접속합니다.

4.  **분석 수행**:
    *   **사이드바**: API Key를 입력하고(선택 사항), 수집할 댓글 수를 설정합니다.
    *   **메인 화면**: 분석하고 싶은 YouTube 영상의 URL 또는 ID를 입력하고 **"분석 시작 🚀"** 버튼을 클릭합니다.

## 📊 실행 결과 화면

*   **감성 분석 결과**: 긍정/부정/중립 비율을 파이 차트로 확인하고, 구체적인 댓글 수를 볼 수 있습니다.
*   **주요 키워드**: 워드 클라우드를 통해 영상에 대한 주요 반응 키워드를 한눈에 파악할 수 있습니다.
*   **데이터 보기**: 수집된 원본 댓글과 분석 결과를 표 형태로 확인하고 CSV로 저장할 수 있습니다.

## ⚠️ 주의사항

*   **API 할당량**: YouTube Data API는 하루 사용량 제한(Quota)이 있습니다. 과도한 요청 시 API 호출이 차단될 수 있습니다.
*   **형태소 분석 속도**: 댓글 수가 많을 경우 `Konlpy`의 분석 속도가 느려질 수 있습니다.
*   **Streamlit 실행**: 반드시 `python analysisYoutube.py`가 아닌 `streamlit run analysisYoutube.py`로 실행해야 합니다.

## 📝 파일 구조

*   `analysisYoutube.py`: Streamlit 웹 애플리케이션 소스 코드
*   `readme.md`: 프로젝트 설명 파일
