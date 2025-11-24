from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json
import joblib
import numpy as np
import pandas as pd
from catboost import Pool

app = Flask(__name__)
CORS(app)

# ---------------------------
# 서울 경계(대략적 bbox): 위도/경도 한정
# ---------------------------
SEOUL_BOUNDS = {
    "lat_min": 37.413294,  # 남단(과천/성남 경계 부근)
    "lat_max": 37.715,     # 북단(도봉/강북 상단 근처, 살짝 여유)
    "lon_min": 126.734086, # 서단(김포/강서 경계 부근)
    "lon_max": 127.269311, # 동단(구리/남양주 경계 부근)
}

def in_seoul(lat, lon) -> bool:
    try:
        lat = float(lat); lon = float(lon)
    except Exception:
        return False
    return (SEOUL_BOUNDS["lat_min"] <= lat <= SEOUL_BOUNDS["lat_max"] and
            SEOUL_BOUNDS["lon_min"] <= lon <= SEOUL_BOUNDS["lon_max"])

# ---------------------------
# 모델 및 학습 메타 로드
# ---------------------------
MODEL_PATHS = {
    "자동차": "noise_model_자동차.pkl",
    "이륜자동차": "noise_model_이륜자동차.pkl",
    "열차": "noise_model_열차.pkl",
}
models = {}
for k, p in MODEL_PATHS.items():
    if os.path.exists(p):
        models[k] = joblib.load(p)
        print(f"✅ {k} 모델 로드 완료")
    else:
        print(f"⚠️ {k} 모델 없음: {p}")

with open("feature_list.json", "r", encoding="utf-8") as f:
    FEATURE_COLS = json.load(f)
with open("cat_cols.json", "r", encoding="utf-8") as f:
    CAT_COLS = json.load(f)

# 서버 기본값(프론트에서 안 보낸 값은 여기서 채움)
DEFAULTS = {
    "urban": "urban",
    "district": "None",
    "place": "None",
    "areaUse": "None",
    "obstacle": "None",
    "distance_m": 20,
    "duration": 5,
    "sampleRate_kHz": 44.1,
    "category_01": "교통소음",
    "category_03": "일반",
    "subCategory": "None",
}

print("🚀 Noise Prediction API running on port 5001")

# ---------------------------
# 전처리: 최소 입력 → 학습 피처로 확장
# ---------------------------
def make_feature_frame(payload: dict) -> pd.DataFrame:
    x = {**DEFAULTS, **payload}  # 누락된 키를 기본값으로 보강
    df = pd.DataFrame([x])

    # 안전 캐스팅
    df["hour"] = pd.to_numeric(df.get("hour", np.nan), errors="coerce")
    df["distance_m"] = pd.to_numeric(df.get("distance_m", 20), errors="coerce").fillna(20)

    # 파생 피처(학습 코드와 동일 로직)
    df["sin_hour"] = np.sin(2*np.pi*df["hour"]/24)
    df["cos_hour"] = np.cos(2*np.pi*df["hour"]/24)
    df["is_daytime"] = df["hour"].apply(lambda h: 1 if pd.notna(h) and 6 <= float(h) <= 20 else 0)
    df["log_distance"] = np.log1p(df["distance_m"])
    weather_str = df.get("weather", "").astype(str)
    urban_str = df.get("urban", "").astype(str)
    df["weather_daytime"] = weather_str + "_" + df["is_daytime"].astype(str)
    df["urban_x_weather"] = urban_str + "_" + weather_str
    df["hour_group"] = pd.cut(df["hour"], bins=[-1,6,12,18,24], labels=["밤","오전","오후","야간"])
    if "weather" in df.columns:
        df["weather_simple"] = df["weather"].replace({"맑음":"좋음","흐림":"나쁨","비":"나쁨","눈":"나쁨"})
    else:
        df["weather_simple"] = "None"

    # 누락 컬럼 채우고 순서 정렬
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[FEATURE_COLS]

    # 범주형 NaN 금지
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(object).where(df[c].notna(), "None").astype(str)

    # 수치형 안전화
    for c in [k for k in FEATURE_COLS if k not in CAT_COLS]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def cat_idx(df: pd.DataFrame):
    return [df.columns.get_loc(c) for c in CAT_COLS if c in df.columns]

# ---------------------------
# 단일 예측(참고용, 유지)
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict_once():
    data = request.get_json(force=True)
    # ✅ 서울 좌표 검증
    if not in_seoul(data.get("latitude"), data.get("longitude")):
        return jsonify({"error": "위치가 서울 경계 밖입니다. 서울 내 좌표만 허용합니다.",
                        "bounds": SEOUL_BOUNDS}), 400

    vtype = (data.get("category_02", "자동차") or "").strip()
    if vtype not in models:
        return jsonify({"error": f"❌ '{vtype}' 모델이 없습니다."}), 400
    model = models[vtype]
    df = make_feature_frame(data)
    pool = Pool(df, cat_features=cat_idx(df))
    pred = float(model.predict(pool)[0])
    return jsonify({"pred_db": round(pred, 2)})

# ---------------------------
# 시간대별 예측 + 시간대별 원인 Top-K(ShapValues)
# ---------------------------
@app.route("/predict_series_explain", methods=["POST"])
def predict_series_explain():
    """
    입력: { latitude, longitude, weather, category_02 }
    출력: {
      hourly: [{hour, pred_db}],
      reasons: { "<hour>": [{feature, contribution, abs_contribution, rank}, ...TopK] }
    }
    """
    data = request.get_json(force=True)
    # ✅ 서울 좌표 검증
    if not in_seoul(data.get("latitude"), data.get("longitude")):
        return jsonify({"error": "위치가 서울 경계 밖입니다. 서울 내 좌표만 허용합니다.",
                        "bounds": SEOUL_BOUNDS}), 400

    vtype = (data.get("category_02", "자동차") or "").strip()
    if vtype not in models:
        return jsonify({"error": f"❌ '{vtype}' 모델이 없습니다."}), 400
    model = models[vtype]

    hourly = []
    reasons = {}
    TOPK = 5

    for h in range(24):
        temp = dict(data)
        temp["hour"] = h
        df = make_feature_frame(temp)
        pool = Pool(df, cat_features=cat_idx(df))

        # 예측값
        pred = float(model.predict(pool)[0])
        hourly.append({"hour": h, "pred_db": round(pred, 2)})

        # --- ShapValues 기여도 ---
        contrib_raw = model.get_feature_importance(pool, type="ShapValues")
        arr = np.array(contrib_raw, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if arr.ndim == 2:
            row = arr[0]
        elif arr.ndim == 1:
            row = arr
        else:
            row = np.zeros(len(df.columns) + 1, dtype=float)

        if row.size >= len(df.columns):
            feat_contrib = row[:len(df.columns)]   # 마지막(base value) 제외
        else:
            feat_contrib = np.zeros(len(df.columns), dtype=float)

        names = getattr(model, "feature_names_", None) or list(df.columns)
        n = min(len(names), len(feat_contrib))
        items = []
        for f, v in zip(names[:n], feat_contrib[:n]):
            items.append({
                "feature": f,
                "contribution": round(float(v), 4),
                "abs_contribution": round(float(abs(v)), 4),
            })
        items.sort(key=lambda x: -x["abs_contribution"])
        for i, it in enumerate(items):
            it["rank"] = i + 1
        reasons[str(h)] = items[:TOPK]

    return jsonify({"hourly": hourly, "reasons": reasons})

# ---------------------------
# 특정 시간대 상세 기여도(ShapValues)
# ---------------------------
@app.route("/explain_hour", methods=["POST"])
def explain_hour():
    data = request.get_json(force=True)
    # ✅ 서울 좌표 검증
    if not in_seoul(data.get("latitude"), data.get("longitude")):
        return jsonify({"error": "위치가 서울 경계 밖입니다. 서울 내 좌표만 허용합니다.",
                        "bounds": SEOUL_BOUNDS}), 400

    vtype = (data.get("category_02", "자동차") or "").strip()
    hour = int(data.get("hour", 12))
    if vtype not in models:
        return jsonify({"error": f"❌ '{vtype}' 모델이 없습니다."}), 400
    model = models[vtype]

    temp = dict(data)
    temp["hour"] = hour
    df = make_feature_frame(temp)
    pool = Pool(df, cat_features=cat_idx(df))

    contrib_raw = model.get_feature_importance(pool, type="ShapValues")
    arr = np.array(contrib_raw, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.ndim == 2:
        row = arr[0]
    elif arr.ndim == 1:
        row = arr
    else:
        row = np.zeros(len(df.columns) + 1, dtype=float)

    if row.size >= len(df.columns):
        feat = row[:len(df.columns)]
    else:
        feat = np.zeros(len(df.columns), dtype=float)

    names = getattr(model, "feature_names_", None) or list(df.columns)
    n = min(len(names), len(feat))
    out = []
    for f, v in zip(names[:n], feat[:n]):
        out.append({
            "feature": f,
            "contribution": round(float(v), 4),
            "abs_contribution": round(float(abs(v)), 4)
        })
    out.sort(key=lambda x: -x["abs_contribution"])
    return jsonify({"hour": hour, "contributions": out})

# ---------------------------
# 에러도 JSON으로 반환
# ---------------------------
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not Found", "detail": str(e)}), 404

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": "ServerError", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
