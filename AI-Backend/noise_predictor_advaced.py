# ============================================================
# noise_predictor_filtered.py (항공기/헬리콥터 제외 버전)
# ============================================================
import os, json, glob, re, joblib, warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor, Pool
warnings.filterwarnings("ignore")

# ============================================================
# 1) JSON 병합
# ============================================================
json_dir = "./noise_data"
all_json = glob.glob(os.path.join(json_dir, "*.json"))

def parse_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for ann in data.get("annotations", []):
        env = data.get("environment", {}) or {}
        gps = env.get("gps", {}) or {}
        audio = data.get("audio", {}) or {}
        cat = ann.get("categories", {}) or {}

        try:
            hour = int(str(env.get("recordingTime", "")).split(":")[0])
        except:
            hour = np.nan

        m = re.search(r"(\d+)", str(env.get("distance", "")))
        dist = int(m.group(1)) if m else np.nan

        rows.append({
            "hour": hour,
            "dayNight": env.get("dayNight"),
            "urban": env.get("urban"),
            "district": env.get("district"),
            "place": env.get("place"),
            "areaUse": env.get("areaUse"),
            "weather": env.get("weather"),
            "distance_m": dist,
            "obstacle": env.get("obstacle"),
            "latitude": gps.get("latitude"),
            "longitude": gps.get("longitude"),
            "duration": audio.get("duration"),
            "sampleRate_kHz": float(re.sub(r"[^\d.]", "", str(audio.get("sampleRate", "")) or "0") or 0),
            "category_01": cat.get("category_01"),
            "category_02": cat.get("category_02"),
            "category_03": cat.get("category_03"),
            "subCategory": ann.get("subCategory"),
            "decibel": ann.get("decibel")
        })
    return rows


# ============================================================
# 2) 데이터 병합 및 필터링
# ============================================================
records = []
for f in tqdm(all_json, desc="📂 JSON 병합 중"):
    records.extend(parse_json(f))
df = pd.DataFrame(records)
print(f"✅ JSON 병합 완료: {len(df)}")

req_cols = ["decibel","hour","weather","urban","category_02","category_03"]
df = df.dropna(subset=req_cols).copy()

# 항공기/헬리콥터 제외
exclude_list = ["항공기", "비행기", "헬리콥터"]
df = df[~df["category_02"].isin(exclude_list)].copy()
print("🚫 항공기/헬리콥터 제외 완료")
print("📊 남은 category_02 분포:")
print(df["category_02"].value_counts())

# ============================================================
# 3) Feature Engineering
# ============================================================
df["sin_hour"] = np.sin(2*np.pi*df["hour"]/24)
df["cos_hour"] = np.cos(2*np.pi*df["hour"]/24)
df["is_daytime"] = df["hour"].apply(lambda x: 1 if 6 <= x <= 20 else 0)
df["log_distance"] = np.log1p(df["distance_m"].fillna(10))
df["weather_daytime"] = df["weather"].astype(str) + "_" + df["is_daytime"].astype(str)
df["urban_x_weather"] = df["urban"].astype(str) + "_" + df["weather"].astype(str)
df["hour_group"] = pd.cut(df["hour"], bins=[-1,6,12,18,24], labels=["밤","오전","오후","야간"])
df["weather_simple"] = df["weather"].replace({"맑음":"좋음","흐림":"나쁨","비":"나쁨","눈":"나쁨"})

feature_cols = [
    "hour","sin_hour","cos_hour","is_daytime","log_distance",
    "latitude","longitude","duration","sampleRate_kHz",
    "dayNight","urban","district","place","areaUse","weather","obstacle",
    "category_01","category_02","category_03","subCategory",
    "weather_daytime","urban_x_weather","hour_group","weather_simple"
]
cat_cols = df[feature_cols].select_dtypes(include=["object","category"]).columns.tolist()
cat_idx = [feature_cols.index(c) for c in cat_cols]

# ============================================================
# 4) 학습 루프 (자동차 / 이륜자동차 / 열차)
# ============================================================
vehicle_groups = df["category_02"].unique().tolist()
print("🚗 학습 대상 종류:", vehicle_groups)
models = {}

for vtype in vehicle_groups:
    subset = df[df["category_02"] == vtype]
    if len(subset) < 50:
        print(f"⚠️ {vtype} 데이터 부족 ({len(subset)}) → 건너뜀")
        continue

    idx_train, idx_test = train_test_split(
        subset.index,
        stratify=subset["category_03"],
        test_size=0.2,
        random_state=42
    )
    train_df = subset.loc[idx_train].copy()
    test_df = subset.loc[idx_test].copy()

    for col in train_df.select_dtypes(include="object").columns:
        train_df[col] = train_df[col].fillna("None").astype(str)
    for col in test_df.select_dtypes(include="object").columns:
        test_df[col] = test_df[col].fillna("None").astype(str)

    X_train = train_df[feature_cols]
    y_train = train_df["decibel"].astype(float)
    X_test = test_df[feature_cols]
    y_test = test_df["decibel"].astype(float)

    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    test_pool = Pool(X_test, y_test, cat_features=cat_idx)

    model = CatBoostRegressor(
        loss_function="MAE",
        iterations=1000,
        depth=8,
        learning_rate=0.03,
        subsample=0.8,
        random_seed=42,
        eval_metric="MAE",
        od_type="Iter",
        od_wait=150,
        verbose=200
    )
    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    pred = model.predict(test_pool)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"✅ {vtype} 완료: MAE={mae:.2f}, R²={r2:.3f}")

    models[vtype] = model
    joblib.dump(model, f"noise_model_{vtype}.pkl")

# ============================================================
# 5) 피처 목록 저장
# ============================================================
with open("feature_list.json","w",encoding="utf-8") as f:
    json.dump(feature_cols, f, ensure_ascii=False, indent=2)
with open("cat_cols.json","w",encoding="utf-8") as f:
    json.dump(cat_cols, f, ensure_ascii=False, indent=2)
print("💾 모든 모델 저장 완료:", list(models.keys()))
