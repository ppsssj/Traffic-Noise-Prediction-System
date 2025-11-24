import os, json, glob, re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def parse_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # annotations는 리스트 형태이므로 각 사운드 구간을 펼쳐야 함
    rows = []
    for ann in data.get("annotations", []):
        env = data.get("environment", {})
        gps = env.get("gps", {})
        audio = data.get("audio", {})
        cat = ann.get("categories", {})

        # 시간 → 시(hour)
        rec_time = env.get("recordingTime", "")
        hour = None
        if isinstance(rec_time, str):
            try:
                hour = int(rec_time.split(":")[0])
            except:
                hour = np.nan

        # 거리 숫자 추출
        dist = env.get("distance", "")
        dist_val = None
        if isinstance(dist, str):
            match = re.search(r"(\d+)", dist)
            if match:
                dist_val = int(match.group(1))

        # sampleRate 변환 (kHz 제거)
        sr = audio.get("sampleRate", "")
        sr_val = None
        if isinstance(sr, str):
            match = re.search(r"(\d+(\.\d+)?)", sr)
            if match:
                sr_val = float(match.group(1))

        rows.append({
            "hour": hour,
            "dayNight": env.get("dayNight"),
            "urban": env.get("urban"),
            "district": env.get("district"),
            "place": env.get("place"),
            "areaUse": env.get("areaUse"),
            "weather": env.get("weather"),
            "distance_m": dist_val,
            "obstacle": env.get("obstacle"),
            "latitude": gps.get("latitude"),
            "longitude": gps.get("longitude"),
            "duration": audio.get("duration"),
            "sampleRate_kHz": sr_val,
            "category_01": cat.get("category_01"),
            "category_02": cat.get("category_02"),
            "category_03": cat.get("category_03"),
            "subCategory": ann.get("subCategory"),
            "decibel": ann.get("decibel")
        })
    return rows

# 🔹 1. JSON 병합
folder = "./noise_data"
all_files = glob.glob(os.path.join(folder, "*.json"))

data = []
for f in all_files:
    data.extend(parse_json(f))

df = pd.DataFrame(data)
print(f"✅ JSON 병합 완료: {len(df)} 개 레코드")

# 🔹 2. 정제
df = df.dropna(subset=["decibel", "hour", "weather", "urban"])
print(f"✅ 유효 데이터: {len(df)}")

# 🔹 3. 피처 확장
df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
df["is_daytime"] = df["hour"].apply(lambda x: 1 if 6 <= x <= 20 else 0)

# 🔹 4. 인코딩
le = LabelEncoder()
cat_cols = [
    "dayNight", "urban", "district", "place", "areaUse",
    "weather", "obstacle", "category_01", "category_02",
    "category_03", "subCategory"
]
for col in cat_cols:
    df[col] = df[col].fillna("기타")
    df[col] = le.fit_transform(df[col])

# 🔹 5. 피처 선택
feature_cols = [
    "hour", "sin_hour", "cos_hour", "is_daytime",
    "dayNight", "urban", "district", "place", "areaUse",
    "weather", "obstacle",
    "distance_m", "duration", "sampleRate_kHz",
    "latitude", "longitude",
    "category_01", "category_02", "category_03", "subCategory"
]

X = df[feature_cols]
y = df["decibel"]

# 🔹 6. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=df["category_03"], random_state=42
)

# 🔹 7. 학습
model = RandomForestRegressor(n_estimators=400, random_state=42)
model.fit(X_train, y_train)

# 🔹 8. 평가
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print(f"✅ 평가 완료: MAE={mae:.2f}, R²={r2:.3f}")

# 🔹 9. 저장
joblib.dump(model, "noise_predictor.pkl")
print("💾 모델 저장 완료: noise_predictor.pkl")
