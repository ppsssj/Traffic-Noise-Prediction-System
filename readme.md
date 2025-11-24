# 🚦 교통 소음 예측 시스템 (Traffic Noise Prediction System)

본 프로젝트는 **도시 환경에서 수집된 비정형 교통 소음 메타데이터(JSON)**를 기반으로,
**CatBoost 회귀 모델**을 이용해 **교통수단별 소음 수준(dB)을 예측**하는 시스템입니다.

프론트엔드에서는 **React 기반 지도 UI**를 통해
위치·환경값을 입력하면 **24시간 소음 곡선·원인 기여도 표가 즉시 갱신**되도록 설계했습니다.

백엔드는 **Flask API + CatBoost 모델(.pkl)** 구조로 동작하며,
범주형/수치형 스키마를 고정하기 위해 **feature_list.json**, **cat_cols.json**을 함께 사용합니다.

---

## 🔍 주요 기능 요약 (Features)

### 1) **교통수단별 소음 예측 모델**

* 자동차 / 이륜자동차 / 열차에 대해 **개별 CatBoost 회귀 모델** 사용
* 시간·거리·위치·도시환경 요인을 통합하여 dB 예측
* 전처리 및 스키마가 완전 고정된 상태로 서빙 (`feature_list.json`)

### 2) **24시간 소음 프로파일 자동 생성**

* 사용자가 선택한 지점의 좌표(latitude, longitude)에 대해
  **0~23시 시간축 기준 예측값을 일괄 계산**
* 프론트엔드에서 곡선 그래프(Line Chart)로 시각화

### 3) **원인 기여도(Feature Importance) 표시**

* 모델이 판단한 주요 영향 요인을
  “거리·시간·도시환경·날씨·카테고리” 기준으로 재정렬하여 UI에 표시

### 4) **인터랙티브 지도 UI**

* 마커 이동/클릭 시 API를 호출해 결과를 즉시 갱신
* 좌표는 서울 지역 범위로 제한 (Validation 포함)

### 5) **완전 분리형 구조 (AI_Backend / frontend)**

* **백엔드만 단독 실행 가능**
* **프론트엔드만 단독 개발 가능**
* CORS 허용 및 환경변수 기반 API URL 설정

---

## 📁 프로젝트 구조

```
AI_Backend/
  ├─ noise_analysis.py        # CatBoost 모델 학습 및 저장
  ├─ app.py                   # Flask API 서버 (예측/스키마 로딩)
  ├─ requirements.txt         # Python 의존성 목록
  ├─ feature_list.json        # 모델 입력 컬럼 순서 스키마
  ├─ cat_cols.json            # 범주형 컬럼 정의
  ├─ noise_model_자동차.pkl
  ├─ noise_model_이륜자동차.pkl
  └─ noise_model_열차.pkl

frontend/
  ├─ package.json
  ├─ package-lock.json
  ├─ src/
  ├─ public/
  └─ .env.example             # API URL 템플릿
```

---

## ▶️ 백엔드 실행 방법

```bash
cd AI_Backend
pip install -r requirements.txt
python app.py
# 서버 URL: http://127.0.0.1:5001
```

※ CORS 기본 허용
※ 모델 재학습 필요 시 `noise_analysis.py` 실행 후 `.pkl` 교체

---

## ▶️ 프론트엔드 실행 방법

```bash
cd frontend
npm ci   # 또는 npm install
```

### 환경변수 생성

```
REACT_APP_API_BASE=http://127.0.0.1:5001
```

### 개발 서버 실행

```bash
npm start
# http://localhost:3000
```

---

## ✔️ 작동 확인 체크리스트

* 지도에서 마커를 움직이면 **24시간 예측 그래프가 즉시 갱신**
* 주요 기여 요인이 표 형태로 출력
* API 요청이 **서울 범위 좌표만 허용**되는지 확인
* 백엔드 콘솔에 요청 로그 정상 출력

---

## 📌 참고 및 개발 규칙

* `node_modules/`, `__pycache__/`, `.pkl` 등은 ZIP 제출 시 제외
* 모델 구조를 변경하면 반드시
  `feature_list.json`, `cat_cols.json`을 함께 갱신해야 함
* 프론트엔드는 API 응답 스키마에 의존하므로
  필드명 변경 시 반드시 동기화 필요

