import React, { useEffect, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import "./App.css";

/* ===== Leaflet default marker ===== */
let DefaultIcon = L.icon({
  iconUrl: require("leaflet/dist/images/marker-icon.png"),
  shadowUrl: require("leaflet/dist/images/marker-shadow.png"),
});
L.Marker.prototype.options.icon = DefaultIcon;

/* ===== Constants ===== */
const SEOUL_BOUNDS = [
  [37.413294, 126.734086], // SW
  [37.715000, 127.269311], // NE
];
const CENTER = [37.52, 126.91];
const API = "http://127.0.0.1:5001";

/* ===== Utils ===== */
function clamp(v, min, max) { return Math.min(Math.max(v, min), max); }
function inSeoul(lat, lon) {
  return (lat >= SEOUL_BOUNDS[0][0] && lat <= SEOUL_BOUNDS[1][0] &&
          lon >= SEOUL_BOUNDS[0][1] && lon <= SEOUL_BOUNDS[1][1]);
}

function MapClick({ onPick }) {
  useMapEvents({
    click(e) {
      const { lat, lng } = e.latlng;
      if (!inSeoul(lat, lng)) {
        alert("서울 지역 밖은 선택할 수 없습니다.");
        return;
      }
      onPick(lat, lng);
    },
  });
  return null;
}

async function safeJsonFetch(url, options) {
  const r = await fetch(url, options);
  const text = await r.text();
  const ct = r.headers.get("content-type") || "";
  if (!r.ok) {
    try {
      const j = ct.includes("application/json") ? JSON.parse(text) : null;
      const msg = j?.error || j?.detail || text.slice(0, 300);
      throw new Error(`HTTP ${r.status} ${r.statusText}: ${msg}`);
    } catch {
      throw new Error(`HTTP ${r.status} ${r.statusText}: ${text.slice(0, 300)}`);
    }
  }
  if (!ct.includes("application/json")) {
    throw new Error(`Non-JSON response: ${text.slice(0, 300)}`);
  }
  return JSON.parse(text);
}

/* ===== App ===== */
export default function App() {
  // Inputs
  const [lat, setLat] = useState(CENTER[0]);
  const [lon, setLon] = useState(CENTER[1]);
  const [weather, setWeather] = useState("맑음");
  const [category, setCategory] = useState("자동차");

  // Outputs
  const [hourly, setHourly] = useState([]);
  const [reasons, setReasons] = useState({});
  const [selHour, setSelHour] = useState(12);
  const [loading, setLoading] = useState(false);

  // Fetch
  async function run(nextLat = lat, nextLon = lon, nextWeather = weather, nextCategory = category) {
    setLoading(true);
    try {
      const payload = { latitude: nextLat, longitude: nextLon, weather: nextWeather, category_02: nextCategory };
      const d = await safeJsonFetch(`${API}/predict_series_explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      setHourly(d.hourly || []);
      setReasons(d.reasons || {});
    } catch (e) {
      console.error(e);
      alert(e.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { run(); }, []); // initial

  function onChartClick(state) {
    const p = state && state.activePayload && state.activePayload[0];
    if (p && typeof p.payload?.hour === "number") setSelHour(p.payload.hour);
  }

  const selTopK = reasons[String(selHour)] || [];

  function handleLatChange(val) {
    const num = parseFloat(val);
    if (Number.isNaN(num)) return;
    setLat(clamp(num, SEOUL_BOUNDS[0][0], SEOUL_BOUNDS[1][0]));
  }
  function handleLonChange(val) {
    const num = parseFloat(val);
    if (Number.isNaN(num)) return;
    setLon(clamp(num, SEOUL_BOUNDS[0][1], SEOUL_BOUNDS[1][1]));
  }

  return (
    <div className="app">
      {/* Map side */}
      <section className="map-wrap">
        <div className="map-header">
          <span>서울 경계 내에서만 조회 가능</span>
        </div>
        <MapContainer
          className="map-canvas"
          center={CENTER}
          zoom={12}
          maxBounds={SEOUL_BOUNDS}
          maxBoundsViscosity={1.0}
          minZoom={11}
        >
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

          {/* (옵션) 지도 클릭 이동 */}
          <MapClick
            onPick={(lt, ln) => {
              setLat(lt);
              setLon(ln);
            }}
          />

          {/* Draggable marker → dragend 시 즉시 호출 */}
          <Marker
            position={[lat, lon]}
            draggable={true}
            eventHandlers={{
              dragend: (e) => {
                const p = e.target.getLatLng();
                if (!inSeoul(p.lat, p.lng)) {
                  alert("서울 지역 밖은 불가");
                  e.target.setLatLng({ lat, lng: lon });
                  return;
                }
                setLat(p.lat);
                setLon(p.lng);
                run(p.lat, p.lng, weather, category);
              },
            }}
          >
            <Popup>예측 위치<br />({lat.toFixed(4)}, {lon.toFixed(4)})</Popup>
          </Marker>
        </MapContainer>
      </section>

      {/* Side panel */}
      <aside className="side">
        <div className="side__head">
          <h2 className="side__title">시간대별 소음 예측 + 원인 분석</h2>
          <span className="badge">서울 한정</span>
        </div>
        <div className="side__sub">좌표·날씨·소음원만 입력하면 24시간 예측과 Top-5 기여도를 제공합니다.</div>

        <div className="side__scroll">
          {/* Form */}
          <div className="form-grid">
            <div className="form-item">
              <label>소음 종류</label>
              <select className="select" value={category}
                onChange={(e) => {
                  const v = e.target.value;
                  setCategory(v);
                }}>
                <option>자동차</option>
                <option>이륜자동차</option>
                <option>열차</option>
              </select>
            </div>
            <div className="form-item">
              <label>날씨</label>
              <select className="select" value={weather}
                onChange={(e) => {
                  const v = e.target.value;
                  setWeather(v);
                }}>
                <option>맑음</option>
                <option>흐림</option>
                <option>비</option>
                <option>눈</option>
              </select>
            </div>
            <div className="form-item">
              <label>위도 (서울만)</label>
              <input className="input" value={lat} step="0.0001"
                     onChange={(e) => handleLatChange(e.target.value)}
                     onBlur={() => run(lat, lon, weather, category)} />
            </div>
            <div className="form-item">
              <label>경도 (서울만)</label>
              <input className="input" value={lon} step="0.0001"
                     onChange={(e) => handleLonChange(e.target.value)}
                     onBlur={() => run(lat, lon, weather, category)} />
            </div>
          </div>

          <button className="btn" onClick={() => run()} disabled={loading}>
            {loading ? "계산 중..." : "예측 실행"}
          </button>

          {/* Chart */}
          <div className="card">
            <div className="card__row">
              <h3 className="card__title">시간대별 예측(dB)</h3>
              <span className="badge">라인 차트</span>
            </div>
            <div className="chart">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={hourly} onClick={onChartClick}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hour" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="pred_db" stroke="#60a5fa" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Reasons */}
          <div className="card">
            <div className="card__row">
              <h3 className="card__title">원인 분석 (Top-5)</h3>
              <div style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                <span className="badge" title="시간 선택">HOUR</span>
                <select className="select" style={{ width: 120 }}
                        value={selHour} onChange={(e) => setSelHour(parseInt(e.target.value, 10))}>
                  {Array.from({ length: 24 }, (_, i) => <option key={i} value={i}>{i}:00</option>)}
                </select>
              </div>
            </div>
            <table className="table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Feature</th>
                  <th>기여도</th>
                </tr>
              </thead>
              <tbody>
                {selTopK.length > 0 ? selTopK.map((r) => (
                  <tr key={r.feature}>
                    <td>{r.rank}</td>
                    <td>{r.feature}</td>
                    <td className={r.contribution >= 0 ? "positive" : "negative"}>
                      {r.contribution >= 0 ? "+" : ""}{r.contribution} (|{r.abs_contribution}|)
                    </td>
                  </tr>
                )) : (
                  <tr><td colSpan={3} style={{ color: "var(--muted)", paddingTop: 6 }}>
                    해당 시간대 원인 데이터가 없습니다.
                  </td></tr>
                )}
              </tbody>
            </table>
          </div>

        </div>
      </aside>
    </div>
  );
}
