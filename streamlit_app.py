import json
import math
import os
import re
import shutil
import tempfile
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import requests
import streamlit as st
import toml

APP_VERSION = "0.6.0-online"
TMP_ROOT = Path(tempfile.gettempdir()) / "physiosentinel_gait_online" / "sessions"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

HALPE26 = {
    "Nose": 0,
    "LShoulder": 5, "RShoulder": 6,
    "LElbow": 7, "RElbow": 8,
    "LWrist": 9, "RWrist": 10,
    "LHip": 11, "RHip": 12,
    "LKnee": 13, "RKnee": 14,
    "LAnkle": 15, "RAnkle": 16,
    "Head": 17, "Neck": 18, "Hip": 19,
    "LBigToe": 20, "RBigToe": 21,
    "LSmallToe": 22, "RSmallToe": 23,
    "LHeel": 24, "RHeel": 25,
}
LOWER_BODY = ["LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle", "LHeel", "RHeel", "LBigToe", "RBigToe"]
FOOT_POINTS = ["LAnkle", "RAnkle", "LHeel", "RHeel", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe"]
UPPER_BODY = ["LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist"]
ASSISTIVE_OPTIONS = ["Sin ayuda", "Bastón", "1 muleta", "2 muletas", "Caminador", "Rollator", "Otra"]
SKELETON = [
    ("LShoulder", "RShoulder"), ("LShoulder", "LHip"), ("RShoulder", "RHip"), ("LHip", "RHip"),
    ("LShoulder", "LElbow"), ("LElbow", "LWrist"), ("RShoulder", "RElbow"), ("RElbow", "RWrist"),
    ("LHip", "LKnee"), ("LKnee", "LAnkle"), ("LAnkle", "LHeel"), ("LAnkle", "LBigToe"),
    ("RHip", "RKnee"), ("RKnee", "RAnkle"), ("RAnkle", "RHeel"), ("RAnkle", "RBigToe"),
    ("Neck", "Hip"), ("Nose", "Neck"),
]

st.set_page_config(page_title="PhysioSentinel Gait", page_icon="🚶", layout="wide")

# ------------------------- seguridad / secretos -------------------------
def secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

SUPABASE_URL = (secret("SUPABASE_URL", "") or "").rstrip("/")
SUPABASE_KEY = secret("SUPABASE_SERVICE_ROLE_KEY", "") or ""
APP_PASSWORD = secret("GAIT_APP_PASSWORD", "") or ""


def require_password():
    if not APP_PASSWORD:
        st.warning("⚠️ GAIT_APP_PASSWORD no está configurada. Modo de prueba sin control de acceso.")
        return True
    if st.session_state.get("authenticated"):
        return True
    st.title("PhysioSentinel Gait")
    st.caption("Acceso protegido")
    pwd = st.text_input("Contraseña", type="password")
    if st.button("Entrar", type="primary"):
        if pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    return False

if not require_password():
    st.stop()

# ------------------------- Supabase REST -------------------------
def sb_ready():
    return bool(SUPABASE_URL and SUPABASE_KEY)


def sb_headers(extra_prefer=None):
    h = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if extra_prefer:
        h["Prefer"] = extra_prefer
    return h


def sb_request(method, table, params=None, payload=None, prefer=None, timeout=30):
    if not sb_ready():
        raise RuntimeError("Supabase no está configurado en Streamlit Secrets.")
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    r = requests.request(method, url, headers=sb_headers(prefer), params=params, json=payload, timeout=timeout)
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase {r.status_code}: {r.text[:800]}")
    if not r.text.strip():
        return None
    try:
        return r.json()
    except Exception:
        return r.text


def sb_upsert_patient(code):
    data = sb_request(
        "POST", "gait_patients",
        params={"on_conflict": "code"},
        payload={"code": code},
        prefer="resolution=merge-duplicates,return=representation",
    )
    if not data:
        data = sb_request("GET", "gait_patients", params={"code": f"eq.{code}", "select": "id,code"})
    return data[0]["id"]


def sb_create_session(code, record_name, mode, view, meta, assistive_device="Sin ayuda", frontal_orientation="No especificada"):
    patient_id = sb_upsert_patient(code)
    session_id = str(uuid.uuid4())
    payload = {
        "id": session_id,
        "patient_id": patient_id,
        "record_name": record_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "view": view or "",
        "assistive_device": assistive_device or "Sin ayuda",
        "assisted_gait": bool((assistive_device or "Sin ayuda") != "Sin ayuda"),
        "frontal_orientation": frontal_orientation or "No especificada",
        "fps": float(meta.get("fps", 0)) if meta else None,
        "frames": int(meta.get("frames", 0)) if meta else None,
        "duration_s": float(meta.get("duration", 0)) if meta else None,
        "video_persisted": False,
        "app_version": APP_VERSION,
    }
    sb_request("POST", "gait_sessions", payload=payload, prefer="return=minimal")
    return session_id


def sb_save_metrics(session_id, metrics, start_s, end_s):
    payload = []
    for m in metrics:
        v = m.get("value")
        if v is not None:
            try:
                v = float(v)
                if not np.isfinite(v):
                    v = None
            except Exception:
                v = None
        payload.append({
            "session_id": session_id,
            "metric_key": m["key"],
            "metric_label": m["label"],
            "value": v,
            "unit": m.get("unit", ""),
            "quality": m.get("quality", ""),
            "notes": m.get("notes", ""),
        })
    sb_request(
        "POST", "gait_metrics",
        params={"on_conflict": "session_id,metric_key"},
        payload=payload,
        prefer="resolution=merge-duplicates,return=minimal",
        timeout=60,
    )
    sb_request(
        "PATCH", "gait_sessions",
        params={"id": f"eq.{session_id}"},
        payload={"segment_start_s": float(start_s), "segment_end_s": float(end_s), "analysis_status": "completed"},
        prefer="return=minimal",
    )


def sb_list_patients():
    if not sb_ready():
        return pd.DataFrame()
    data = sb_request("GET", "gait_patients", params={"select": "id,code,created_at", "order": "code.asc"}) or []
    return pd.DataFrame(data)


def sb_patient_history(code):
    if not sb_ready():
        return pd.DataFrame()
    p = sb_request("GET", "gait_patients", params={"code": f"eq.{code}", "select": "id,code", "limit": 1}) or []
    if not p:
        return pd.DataFrame()
    pid = p[0]["id"]
    sessions = sb_request(
        "GET", "gait_sessions",
        params={"patient_id": f"eq.{pid}", "select": "id,created_at,record_name,mode,view,assistive_device,assisted_gait,frontal_orientation,fps,frames,duration_s,segment_start_s,segment_end_s,analysis_status", "order": "created_at.asc"},
    ) or []
    if not sessions:
        return pd.DataFrame()
    ids = ",".join(s["id"] for s in sessions)
    metrics = sb_request(
        "GET", "gait_metrics",
        params={"session_id": f"in.({ids})", "select": "session_id,metric_key,metric_label,value,unit,quality,notes"},
    ) or []
    sdf = pd.DataFrame(sessions)
    mdf = pd.DataFrame(metrics)
    if mdf.empty:
        return pd.DataFrame()
    return sdf.merge(mdf, left_on="id", right_on="session_id", how="inner")

# ------------------------- utilidades vídeo -------------------------
def safe_name(text):
    text = (text or "sesion").strip()
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text)[:60] or "sesion"


def video_metadata(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps, "frames": frames, "width": width, "height": height,
        "duration": frames / fps if fps > 0 else 0,
        "orientation": "Vertical" if height > width else "Horizontal",
    }


def save_upload(uploaded, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())


def create_temp_session(patient, record):
    old = st.session_state.get("session_dir")
    if old:
        try:
            shutil.rmtree(old, ignore_errors=True)
        except Exception:
            pass
    folder = TMP_ROOT / f"{safe_name(patient)}_{safe_name(record)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    (folder / "videos").mkdir(parents=True, exist_ok=True)
    return folder


def prepare_config(session_dir):
    cfg = {
        "project": {
            "project_dir": str(session_dir),
            "multi_person": False,
            "participant_height": "auto",
            "participant_mass": 70,
            "frame_rate": "auto",
            "frame_range": "auto",
        },
        "pose": {
            "pose_model": "Body_with_feet",
            "mode": "balanced",
            "det_frequency": 4,
            "device": "auto",
            "backend": "auto",
            "display_detection": False,
            "overwrite_pose": True,
            "save_video": "to_video",
            "output_format": "openpose",
            "tracking_mode": "sports2d",
        },
    }
    path = session_dir / "Config.toml"
    with open(path, "w", encoding="utf-8") as f:
        toml.dump(cfg, f)
    return path


def run_pose2sim(config_path):
    from Pose2Sim import Pose2Sim
    Pose2Sim.poseEstimation(str(config_path))


def find_pose_json_dir(session_dir, cam="cam01"):
    pose_dir = session_dir / "pose"
    if not pose_dir.exists():
        return None
    preferred = sorted([p for p in pose_dir.rglob(f"{cam}*_json") if p.is_dir()])
    if preferred:
        return preferred[0]
    candidates = sorted([p for p in pose_dir.rglob("*_json") if p.is_dir()])
    return candidates[0] if candidates else None


def parse_frame_number(path, fallback):
    m = re.search(r"(\d+)(?=\.json$)", path.name)
    return int(m.group(1)) if m else fallback


def load_pose_dataframe(json_dir):
    rows = []
    for i, path in enumerate(sorted(json_dir.glob("*.json"))):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            people = data.get("people", [])
            if not people:
                continue
            pts = people[0].get("pose_keypoints_2d", [])
            if len(pts) < 26 * 3:
                continue
            row = {"frame": parse_frame_number(path, i)}
            for name, idx in HALPE26.items():
                base = idx * 3
                row[f"{name}_x"] = float(pts[base])
                row[f"{name}_y"] = float(pts[base + 1])
                row[f"{name}_score"] = float(pts[base + 2])
            rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("frame").reset_index(drop=True) if rows else pd.DataFrame()


def point_angle(ax, ay, bx, by, cx, cy):
    ba = np.array([ax - bx, ay - by], dtype=float)
    bc = np.array([cx - bx, cy - by], dtype=float)
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba == 0 or nbc == 0:
        return np.nan
    c = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def robust_rom(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 95) - np.percentile(x, 5)) if len(x) >= 5 else np.nan


def rolling_smooth(arr, window=7):
    return pd.Series(arr).rolling(window, center=True, min_periods=1).mean().to_numpy()


def zero_crossings(signal):
    s = np.asarray(signal, dtype=float)
    out = []
    for i in range(1, len(s)):
        if not (np.isfinite(s[i-1]) and np.isfinite(s[i])):
            continue
        if (s[i-1] <= 0 < s[i]) or (s[i-1] >= 0 > s[i]):
            out.append(i)
    return np.asarray(out, dtype=int)


def quality_label(score):
    return "Alta" if score >= 0.80 else ("Moderada" if score >= 0.65 else "Baja")


def add_angle_columns(seg):
    seg = seg.copy()
    for side in ("L", "R"):
        knee, hip, ankle, shoulder = [], [], [], []
        for _, r in seg.iterrows():
            ka = point_angle(r[f"{side}Hip_x"], r[f"{side}Hip_y"], r[f"{side}Knee_x"], r[f"{side}Knee_y"], r[f"{side}Ankle_x"], r[f"{side}Ankle_y"])
            ha = point_angle(r[f"{side}Shoulder_x"], r[f"{side}Shoulder_y"], r[f"{side}Hip_x"], r[f"{side}Hip_y"], r[f"{side}Knee_x"], r[f"{side}Knee_y"])
            aa = point_angle(r[f"{side}Knee_x"], r[f"{side}Knee_y"], r[f"{side}Ankle_x"], r[f"{side}Ankle_y"], r[f"{side}BigToe_x"], r[f"{side}BigToe_y"])
            sa = point_angle(r[f"{side}Hip_x"], r[f"{side}Hip_y"], r[f"{side}Shoulder_x"], r[f"{side}Shoulder_y"], r[f"{side}Elbow_x"], r[f"{side}Elbow_y"])
            knee.append(180.0 - ka if np.isfinite(ka) else np.nan)
            hip.append(180.0 - ha if np.isfinite(ha) else np.nan)
            ankle.append(aa)
            shoulder.append(sa)
        seg[f"{side}_knee_flex"] = knee
        seg[f"{side}_hip_flex"] = hip
        seg[f"{side}_ankle_angle"] = ankle
        seg[f"{side}_shoulder_elev"] = shoulder
    return seg


def axis_angle_to_vertical(x1, y1, x2, y2):
    """Ángulo firmado de un eje 2D respecto a la vertical de la imagen, plegado a [-90, 90]."""
    dx, dy = float(x2-x1), float(y2-y1)
    if not np.isfinite(dx) or not np.isfinite(dy) or (abs(dx)+abs(dy) == 0):
        return np.nan
    a = float(np.degrees(np.arctan2(dx, -dy)))
    while a > 90: a -= 180
    while a < -90: a += 180
    return a


def axis_angle_to_horizontal(x1, y1, x2, y2):
    dx, dy = float(x2-x1), float(y2-y1)
    if not np.isfinite(dx) or not np.isfinite(dy) or (abs(dx)+abs(dy) == 0):
        return np.nan
    a = float(np.degrees(np.arctan2(dy, dx)))
    while a > 90: a -= 180
    while a < -90: a += 180
    return a


def add_frontal_columns(seg):
    """Métricas proyectadas para vista frontal/posterior. No equivalen a rotaciones 3D ni a pronación clínica."""
    seg = seg.copy()
    for side in ("L", "R"):
        knee_dev, foot_prog, rearfoot = [], [], []
        for _, r in seg.iterrows():
            ka = point_angle(r[f"{side}Hip_x"], r[f"{side}Hip_y"], r[f"{side}Knee_x"], r[f"{side}Knee_y"], r[f"{side}Ankle_x"], r[f"{side}Ankle_y"])
            knee_dev.append(abs(180.0-ka) if np.isfinite(ka) else np.nan)
            foot_prog.append(axis_angle_to_vertical(r[f"{side}Heel_x"], r[f"{side}Heel_y"], r[f"{side}BigToe_x"], r[f"{side}BigToe_y"]))
            rearfoot.append(axis_angle_to_vertical(r[f"{side}Ankle_x"], r[f"{side}Ankle_y"], r[f"{side}Heel_x"], r[f"{side}Heel_y"]))
        seg[f"{side}_frontal_knee_dev"] = knee_dev
        seg[f"{side}_foot_progress_proj"] = foot_prog
        seg[f"{side}_rearfoot_tilt_proj"] = rearfoot
    seg["pelvis_obliquity"] = [axis_angle_to_horizontal(r.LHip_x, r.LHip_y, r.RHip_x, r.RHip_y) for _, r in seg.iterrows()]
    seg["shoulder_obliquity"] = [axis_angle_to_horizontal(r.LShoulder_x, r.LShoulder_y, r.RShoulder_x, r.RShoulder_y) for _, r in seg.iterrows()]
    pelvis_w = np.abs(seg.RHip_x.to_numpy(float) - seg.LHip_x.to_numpy(float))
    ankle_w = np.abs(seg.RAnkle_x.to_numpy(float) - seg.LAnkle_x.to_numpy(float))
    seg["base_width_relative"] = np.divide(ankle_w, pelvis_w, out=np.full_like(ankle_w, np.nan), where=pelvis_w>1e-6)
    return seg


def visibility_pct(seg, names, threshold=0.5):
    cols = [f"{n}_score" for n in names if f"{n}_score" in seg.columns]
    if not cols:
        return np.nan
    return float((seg[cols].min(axis=1) >= threshold).mean() * 100)


def compute_metrics(df, fps, start_frame, end_frame, view, assistive_device="Sin ayuda"):
    seg = df[(df.frame >= start_frame) & (df.frame <= end_frame)].copy()
    if len(seg) < max(30, int(fps * 2)):
        raise ValueError("El segmento seleccionado es demasiado corto.")
    seg = add_angle_columns(seg)
    is_frontal = "Frontal" in (view or "")
    if is_frontal:
        seg = add_frontal_columns(seg)

    score_cols = [f"{n}_score" for n in LOWER_BODY]
    mean_tracking = float(seg[score_cols].mean(axis=1).mean())
    good_frames = float((seg[score_cols].min(axis=1) >= 0.5).mean() * 100)
    foot_visible = visibility_pct(seg, FOOT_POINTS, 0.5)
    upper_visible = visibility_pct(seg, UPPER_BODY, 0.5)
    q = quality_label(mean_tracking)
    assisted = assistive_device != "Sin ayuda"

    ly = (seg.LAnkle_y.to_numpy() + seg.LHeel_y.to_numpy() + seg.LBigToe_y.to_numpy()) / 3.0
    ry = (seg.RAnkle_y.to_numpy() + seg.RHeel_y.to_numpy() + seg.RBigToe_y.to_numpy()) / 3.0
    diff = rolling_smooth(ry - ly, 7)
    crossings = zero_crossings(diff)
    if len(crossings) > 1:
        kept = [crossings[0]]
        min_gap = max(1, int(round(0.25 * fps)))
        for c in crossings[1:]:
            if c - kept[-1] >= min_gap:
                kept.append(c)
        crossings = np.asarray(kept, dtype=int)
    intervals = np.diff(crossings) / fps if len(crossings) >= 2 else np.asarray([])
    cadence = mean_alt = cv_alt = asym = np.nan
    if len(intervals) >= 3:
        mean_alt = float(np.mean(intervals)); cadence = 60.0 / mean_alt
        cv_alt = float(np.std(intervals, ddof=1) / np.mean(intervals) * 100)
        a, b = intervals[0::2], intervals[1::2]
        if len(a) >= 2 and len(b) >= 2:
            ma, mb = float(np.mean(a)), float(np.mean(b))
            asym = abs(ma - mb) / ((ma + mb) / 2.0) * 100.0

    aid_note = f" Marcha con ayuda técnica: {assistive_device}; revisar oclusiones." if assisted else ""
    metrics = [
        {"key":"tracking_mean","label":"Confianza media del tracking","value":mean_tracking,"unit":"","quality":q,"notes":"Media HALPE26 del tren inferior."+aid_note},
        {"key":"good_frames_pct","label":"Frames con tren inferior visible ≥0,50","value":good_frames,"unit":"%","quality":q,"notes":"Puntos principales del tren inferior ≥0,50."+aid_note},
        {"key":"foot_visibility_pct","label":"Visibilidad de pie/tobillo","value":foot_visible,"unit":"%","quality":quality_label((foot_visible or 0)/100),"notes":"Tobillo, talón y antepié. Fundamental para métricas del pie."+aid_note},
        {"key":"upper_visibility_pct","label":"Visibilidad del tren superior","value":upper_visible,"unit":"%","quality":quality_label((upper_visible or 0)/100),"notes":"Hombros, codos y muñecas; puede disminuir con muletas/caminador."+aid_note},
        {"key":"cadence_exp","label":"Cadencia estimada","value":cadence,"unit":"pasos/min","quality":"Experimental" if np.isfinite(cadence) else "No calculable","notes":"Alternancia distal D/I; no equivale todavía a heel-strike validado."},
        {"key":"alternation_interval","label":"Intervalo medio de alternancia","value":mean_alt,"unit":"s","quality":"Experimental" if np.isfinite(mean_alt) else "No calculable","notes":"Cruces de la señal distal D-I."},
        {"key":"regularity_cv","label":"Variabilidad temporal de alternancia","value":cv_alt,"unit":"%","quality":"Experimental" if np.isfinite(cv_alt) else "No calculable","notes":"CV de intervalos de alternancia."},
        {"key":"temporal_asymmetry_exp","label":"Asimetría temporal experimental","value":asym,"unit":"%","quality":"Experimental" if np.isfinite(asym) else "No calculable","notes":"Alternancias impares vs pares; aún no etiquetadas anatómicamente como paso D/I."},
    ]

    if not is_frontal:
        for key_base, label_base, col_base in [
            ("knee_flex", "Flexión rodilla", "knee_flex"),
            ("hip_flex", "Flexión cadera", "hip_flex"),
            ("ankle_angle", "Ángulo tobillo-pie", "ankle_angle"),
            ("shoulder_elev", "Elevación hombro", "shoulder_elev"),
        ]:
            vals = {}
            for side, side_name in [("L","izquierda"),("R","derecha")]:
                arr = seg[f"{side}_{col_base}"].to_numpy(dtype=float)
                p95 = float(np.nanpercentile(arr, 95)); rom = robust_rom(arr); vals[side] = p95
                quality = "Condicionada por ayuda técnica" if assisted and key_base=="shoulder_elev" else q
                metrics += [
                    {"key":f"{key_base}_{side.lower()}_p95","label":f"{label_base} {side_name} 2D (P95)","value":p95,"unit":"°","quality":quality,"notes":f"Ángulo 2D proyectado en vista {view}."+aid_note},
                    {"key":f"{key_base}_{side.lower()}_rom","label":f"ROM {label_base.lower()} {side_name} 2D","value":rom,"unit":"°","quality":quality,"notes":"ROM robusto P95-P5; 2D proyectado."+aid_note},
                ]
            metrics.append({"key":f"{key_base}_diff_p95","label":f"Diferencia D/I {label_base.lower()} 2D","value":abs(vals["L"]-vals["R"]),"unit":"°","quality":q,"notes":"Diferencia absoluta P95 D/I; 2D proyectado."})
    else:
        vals = {}
        for side, side_name in [("L","izquierda"),("R","derecha")]:
            kd = seg[f"{side}_frontal_knee_dev"].to_numpy(float)
            fp = seg[f"{side}_foot_progress_proj"].to_numpy(float)
            rf = seg[f"{side}_rearfoot_tilt_proj"].to_numpy(float)
            vals[side] = {
                "knee": float(np.nanpercentile(kd,95)),
                "foot": float(np.nanmedian(fp)),
                "rear": float(np.nanpercentile(np.abs(rf),95)),
            }
            foot_q = q if foot_visible >= 80 else "Baja/condicionada"
            metrics += [
                {"key":f"frontal_knee_dev_{side.lower()}_p95","label":f"Desviación frontal rodilla {side_name} (P95)","value":vals[side]["knee"],"unit":"°","quality":q,"notes":"Magnitud proyectada del eje cadera-rodilla-tobillo. No diagnostica valgo/varo 3D."},
                {"key":f"foot_progress_{side.lower()}_median","label":f"Orientación del pie {side_name} proyectada (mediana)","value":vals[side]["foot"],"unit":"°","quality":foot_q,"notes":"Proxy distal de orientación/rotación en la imagen. No equivale a rotación axial de cadera."},
                {"key":f"rearfoot_tilt_{side.lower()}_p95","label":f"Inclinación retropié {side_name} proyectada (P95 abs.)","value":vals[side]["rear"],"unit":"°","quality":foot_q,"notes":"Eje tobillo-talón proyectado. Puede sugerir cambios de eversión/inversión, pero no mide pronación 3D."},
            ]
        metrics += [
            {"key":"frontal_knee_dev_diff","label":"Diferencia D/I desviación frontal de rodilla","value":abs(vals['L']['knee']-vals['R']['knee']),"unit":"°","quality":q,"notes":"Comparación 2D proyectada."},
            {"key":"foot_progress_diff","label":"Diferencia D/I orientación del pie proyectada","value":abs(vals['L']['foot']-vals['R']['foot']),"unit":"°","quality":q,"notes":"Proxy distal; no atribuir directamente a rotación de cadera."},
            {"key":"rearfoot_tilt_diff","label":"Diferencia D/I inclinación del retropié proyectada","value":abs(vals['L']['rear']-vals['R']['rear']),"unit":"°","quality":q,"notes":"No equivale a pronación clínica."},
            {"key":"pelvis_obliquity_rom","label":"ROM oblicuidad pélvica proyectada","value":robust_rom(seg.pelvis_obliquity),"unit":"°","quality":q,"notes":"P95-P5 de la línea inter-caderas en el plano de imagen."},
            {"key":"base_width_relative_median","label":"Anchura de base relativa (tobillos/pelvis)","value":float(np.nanmedian(seg.base_width_relative)),"unit":"ratio","quality":q,"notes":"Anchura proyectada normalizada por anchura pélvica; no es distancia métrica sin calibración."},
        ]

    chart_data = {
        "frame": seg.frame.to_numpy(), "time_s": seg.frame.to_numpy()/fps,
        "Alternancia D-I": diff,
    }
    if is_frontal:
        chart_data.update({
            "Rodilla frontal izquierda": seg.L_frontal_knee_dev,
            "Rodilla frontal derecha": seg.R_frontal_knee_dev,
            "Orientación pie izquierda": seg.L_foot_progress_proj,
            "Orientación pie derecha": seg.R_foot_progress_proj,
            "Retropié izquierda": seg.L_rearfoot_tilt_proj,
            "Retropié derecha": seg.R_rearfoot_tilt_proj,
            "Oblicuidad pélvica": seg.pelvis_obliquity,
        })
    else:
        chart_data.update({
            "Rodilla izquierda": seg.L_knee_flex, "Rodilla derecha": seg.R_knee_flex,
            "Cadera izquierda": seg.L_hip_flex, "Cadera derecha": seg.R_hip_flex,
            "Tobillo izquierda": seg.L_ankle_angle, "Tobillo derecha": seg.R_ankle_angle,
            "Hombro izquierda": seg.L_shoulder_elev, "Hombro derecha": seg.R_shoulder_elev,
        })
    return metrics, pd.DataFrame(chart_data), seg

def metric_value(metrics, key):
    for m in metrics:
        if m["key"] == key:
            return m.get("value")
    return None


def fmt(v, n=1):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{n}f}"


def get_point(row, name, min_score=0.25):
    try:
        if row[f"{name}_score"] < min_score:
            return None
        return int(round(row[f"{name}_x"])), int(round(row[f"{name}_y"]))
    except Exception:
        return None


def render_angle_video(video_path, full_df, out_path, view="Lateral", assistive_device="Sin ayuda"):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw = out_path.with_name(out_path.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    enriched = add_angle_columns(full_df)
    if "Frontal" in (view or ""):
        enriched = add_frontal_columns(enriched)
    indexed = enriched.set_index("frame")
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_no in indexed.index:
            row = indexed.loc[frame_no]
            if isinstance(row, pd.DataFrame): row = row.iloc[0]
            for a,b in SKELETON:
                pa, pb = get_point(row,a), get_point(row,b)
                if pa and pb: cv2.line(frame, pa, pb, (40,220,80), 2, cv2.LINE_AA)
            for name in HALPE26:
                p = get_point(row,name)
                if p: cv2.circle(frame,p,3,(0,190,255),-1,cv2.LINE_AA)
            if "Frontal" in (view or ""):
                lines = [
                    f"Rod frontal I/D {row['L_frontal_knee_dev']:.0f}/{row['R_frontal_knee_dev']:.0f} deg",
                    f"Pie orient. I/D {row['L_foot_progress_proj']:.0f}/{row['R_foot_progress_proj']:.0f} deg",
                    f"Retropie I/D {row['L_rearfoot_tilt_proj']:.0f}/{row['R_rearfoot_tilt_proj']:.0f} deg",
                    f"Pelvis {row['pelvis_obliquity']:.0f} deg",
                    "2D proyectado · no rotacion/pronacion 3D",
                ]
            else:
                lines = [
                    f"Cad I/D {row['L_hip_flex']:.0f}/{row['R_hip_flex']:.0f} deg",
                    f"Rod I/D {row['L_knee_flex']:.0f}/{row['R_knee_flex']:.0f} deg",
                    f"Tob I/D {row['L_ankle_angle']:.0f}/{row['R_ankle_angle']:.0f} deg",
                    f"Hom I/D {row['L_shoulder_elev']:.0f}/{row['R_shoulder_elev']:.0f} deg",
                    "2D proyectado",
                ]
            if assistive_device != "Sin ayuda":
                lines.append(f"Ayuda: {assistive_device}")
            y = 32
            for txt in lines:
                cv2.putText(frame, txt, (12,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
                y += 25
        writer.write(frame)
        frame_no += 1
    cap.release(); writer.release()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = f'"{ffmpeg}" -y -loglevel error -i "{raw}" -c:v libx264 -pix_fmt yuv420p -movflags +faststart -an "{out_path}"'
    rc = os.system(cmd)
    try: raw.unlink(missing_ok=True)
    except Exception: pass
    return out_path if rc == 0 and out_path.exists() else None


def cleanup_temp_session(session_dir):
    try:
        shutil.rmtree(session_dir, ignore_errors=True)
        return True
    except Exception:
        return False

# ------------------------- UI -------------------------
st.title("PhysioSentinel Gait")
st.caption(f"Versión {APP_VERSION} · Supabase + histórico persistente · biomecánica frontal proyectada · ayudas técnicas")

if sb_ready():
    st.success("☁️ Supabase conectado: pacientes, sesiones y métricas se guardan de forma persistente. Los vídeos NO se guardan.")
else:
    st.error("Supabase aún no está configurado. Añade SUPABASE_URL y SUPABASE_SERVICE_ROLE_KEY en Streamlit Secrets.")

with st.sidebar:
    st.header("Sesión")
    patient = st.text_input("Paciente / código", value="Prueba")
    record = st.text_input("Nombre del registro", value="Marcha")
    st.caption("Usa un código seudonimizado, no nombre y apellidos.")
    st.divider()
    mode = st.radio("Modo de análisis", ["1 cámara · 2D", "2 cámaras · frontal/posterior + lateral"], index=0)
    view = st.radio("Vista", ["Frontal/posterior", "Lateral"], index=0) if mode.startswith("1 cámara") else "Frontal+Lateral"
    frontal_orientation = st.selectbox("Sentido de la toma frontal/posterior", ["No especificada", "Frontal", "Posterior", "Mixta/ida-vuelta"], index=0) if "Frontal" in view else "No aplica"
    assistive_device = st.selectbox("Ayuda técnica", ASSISTIVE_OPTIONS, index=0)
    if assistive_device != "Sin ayuda":
        st.caption("La app medirá la visibilidad por regiones y marcará métricas condicionadas por posibles oclusiones.")
    st.divider()
    st.markdown("**Motor interno**")
    st.write("Pose2Sim + RTMPose")
    st.write("Body_with_feet / HALPE26")
    st.caption("Vídeo → /tmp → análisis → resultados → eliminación")

# mantener estados
for k, v in {"pose_done":False, "metrics_done":False, "temp_deleted":False}.items():
    if k not in st.session_state: st.session_state[k] = v

tabs = st.tabs(["1 · Vídeos", "2 · Calidad", "3 · Analizar marcha", "4 · Resultados 2D", "5 · Pacientes / Evolución", "6 · 3D futuro"])

with tabs[0]:
    st.subheader("Carga temporal de vídeo")
    st.info("El vídeo se usa únicamente para este análisis. No se sube a Supabase ni queda guardado en el histórico.")
    if mode.startswith("1 cámara"):
        up1 = st.file_uploader(f"Vídeo {view.lower()}", type=["mp4","mov","avi","mkv"], key="video1")
        if up1: st.video(up1)
        up2 = None
    else:
        c1,c2 = st.columns(2)
        with c1:
            up1 = st.file_uploader("Vídeo frontal/posterior", type=["mp4","mov","avi","mkv"], key="front")
            if up1: st.video(up1)
        with c2:
            up2 = st.file_uploader("Vídeo lateral", type=["mp4","mov","avi","mkv"], key="side")
            if up2: st.video(up2)
        st.caption("v0.5 procesa ambas poses; las métricas integradas 3D todavía no se calculan.")

    if st.button("Crear sesión temporal", type="primary", use_container_width=True):
        if up1 is None or (mode.startswith("2 cámaras") and up2 is None):
            st.error("Selecciona el/los vídeo(s) necesarios.")
        else:
            try:
                folder = create_temp_session(patient, record)
                p1 = folder / "videos" / f"cam01{Path(up1.name).suffix.lower() or '.mp4'}"
                save_upload(up1, p1); meta1 = video_metadata(p1)
                if not meta1: raise RuntimeError("No puedo leer el vídeo de cámara 1.")
                p2 = None; meta2 = None
                if up2 is not None:
                    p2 = folder / "videos" / f"cam02{Path(up2.name).suffix.lower() or '.mp4'}"
                    save_upload(up2, p2); meta2 = video_metadata(p2)
                st.session_state.update({
                    "session_dir":str(folder), "video1":str(p1), "video2":str(p2) if p2 else None,
                    "meta1":meta1, "meta2":meta2, "mode":mode, "view":view,
                    "pose_done":False, "metrics_done":False, "temp_deleted":False,
                    "analysis_df":None, "annotated_video_bytes":None,
                    "patient_code":patient.strip(), "record_name":record.strip(),
                    "assistive_device":assistive_device, "frontal_orientation":frontal_orientation,
                })
                if sb_ready():
                    sid = sb_create_session(patient.strip(), record.strip(), mode, view, meta1, assistive_device, frontal_orientation)
                    st.session_state["cloud_session_id"] = sid
                st.success("Sesión temporal creada. El histórico ya tiene la sesión, pero no el vídeo.")
            except Exception as e:
                st.error(str(e))

with tabs[1]:
    st.subheader("Control de calidad")
    meta = st.session_state.get("meta1")
    if not meta:
        st.info("Crea primero una sesión temporal.")
    else:
        a,b,c,d = st.columns(4)
        a.metric("FPS", f"{meta['fps']:.1f}"); b.metric("Duración", f"{meta['duration']:.1f} s")
        c.metric("Resolución", f"{meta['width']} × {meta['height']}"); d.metric("Orientación", meta['orientation'])
        if meta["fps"] >= 29: st.success("Frecuencia de imagen adecuada para esta fase.")
        else: st.warning("FPS bajo: interpretar temporización con cautela.")
        if st.session_state.get("assistive_device", "Sin ayuda") != "Sin ayuda":
            st.info(f"Marcha con ayuda técnica: **{st.session_state.get('assistive_device')}**. La fiabilidad final se calculará según la visibilidad real de pie, tren inferior y tren superior.")

with tabs[2]:
    st.subheader("Analizar marcha")
    if not st.session_state.get("session_dir"):
        st.info("Crea primero una sesión temporal.")
    elif st.session_state.get("temp_deleted"):
        st.info("Los archivos temporales ya fueron eliminados después de guardar los resultados.")
    else:
        st.write("Motor: **Pose2Sim + RTMPose · Body_with_feet (HALPE26)**")
        if st.button("▶ Analizar marcha", type="primary", use_container_width=True):
            try:
                session_dir = Path(st.session_state.session_dir)
                with st.spinner("Detectando pose con Pose2Sim/RTMPose. La primera ejecución puede tardar más..."):
                    cfg = prepare_config(session_dir)
                    run_pose2sim(cfg)
                json_dir = find_pose_json_dir(session_dir, "cam01")
                if not json_dir: raise RuntimeError("Pose2Sim terminó pero no encuentro los JSON de cam01.")
                df = load_pose_dataframe(json_dir)
                if df.empty: raise RuntimeError("No se pudieron leer keypoints HALPE26.")
                st.session_state.analysis_df = df
                st.session_state.pose_done = True
                st.success(f"Pose completada: {len(df)} frames útiles.")
                st.info("Abre **4 · Resultados 2D**, selecciona el tramo válido y calcula. Después se eliminarán los vídeos temporales.")
            except Exception as e:
                st.error(f"Error durante el análisis: {e}")
                with st.expander("Detalles técnicos"):
                    st.code(traceback.format_exc())

with tabs[3]:
    st.subheader("Resultados 2D")
    df = st.session_state.get("analysis_df")
    meta = st.session_state.get("meta1")
    if df is None or not st.session_state.get("pose_done") or not meta:
        if st.session_state.get("metrics_done"):
            pass
        else:
            st.info("Ejecuta primero **Analizar marcha**.")
    if df is not None and meta:
        fps = float(meta["fps"]); duration = (int(df.frame.max()) + 1) / fps
        start_s, end_s = st.slider("Intervalo válido (segundos)", 0.0, float(round(duration,2)), (0.0,float(round(duration,2))), step=max(0.01, round(1/fps,2)))
        if st.button("Calcular, guardar histórico y eliminar vídeo", type="primary", use_container_width=True):
            try:
                start_frame = int(round(start_s*fps)); end_frame = int(round(end_s*fps))
                metrics, chart, seg = compute_metrics(df, fps, start_frame, end_frame, st.session_state.get("view",""), st.session_state.get("assistive_device","Sin ayuda"))
                st.session_state.metrics = metrics; st.session_state.chart = chart; st.session_state.metrics_done = True
                if sb_ready() and st.session_state.get("cloud_session_id"):
                    sb_save_metrics(st.session_state.cloud_session_id, metrics, start_s, end_s)
                # vídeo anotado solo en memoria de la sesión Streamlit; no persiste en Supabase
                session_dir = Path(st.session_state.session_dir)
                try:
                    out = session_dir / "gait_angles_web.mp4"
                    made = render_angle_video(Path(st.session_state.video1), df, out, st.session_state.get("view",""), st.session_state.get("assistive_device","Sin ayuda"))
                    if made and made.exists():
                        st.session_state.annotated_video_bytes = made.read_bytes()
                except Exception:
                    st.session_state.annotated_video_bytes = None
                cleanup_temp_session(session_dir)
                st.session_state.temp_deleted = True
                st.session_state.analysis_df = None
                st.success("✅ Resultados guardados en Supabase. ✅ Vídeo y archivos Pose2Sim eliminados del servidor temporal.")
            except Exception as e:
                st.error(str(e))
                with st.expander("Detalles técnicos"): st.code(traceback.format_exc())

    if st.session_state.get("metrics_done"):
        metrics = st.session_state.metrics; chart = st.session_state.chart
        st.markdown("### Resumen clínico 2D")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Cadencia estimada", fmt(metric_value(metrics,"cadence_exp"),1)+" pasos/min")
        c2.metric("Regularidad temporal", fmt(metric_value(metrics,"regularity_cv"),1)+" % CV")
        c3.metric("Tracking válido", fmt(metric_value(metrics,"good_frames_pct"),1)+" %")
        c4.metric("Asimetría temporal", fmt(metric_value(metrics,"temporal_asymmetry_exp"),1)+" %")
        st.caption("Cadencia, regularidad y asimetría temporal son experimentales hasta validar eventos de contacto del pie.")

        if "Frontal" in st.session_state.get("view", ""):
            st.markdown("### Biomecánica frontal/posterior 2D proyectada")
            st.warning("La rotación axial de cadera y la pronación son movimientos 3D. Aquí se muestran proxies 2D: eje frontal de rodilla, orientación distal del pie e inclinación tobillo-talón. No deben etiquetarse como rotación de cadera o pronación clínica aisladas.")
            for title, cols in [
                ("Eje frontal de rodilla", ["Rodilla frontal izquierda","Rodilla frontal derecha"]),
                ("Orientación distal del pie", ["Orientación pie izquierda","Orientación pie derecha"]),
                ("Inclinación del retropié", ["Retropié izquierda","Retropié derecha"]),
                ("Oblicuidad pélvica", ["Oblicuidad pélvica"]),
            ]:
                st.markdown(f"#### {title}")
                st.line_chart(chart.set_index("time_s")[cols], x_label="Tiempo (s)", y_label="Ángulo proyectado (°)")
        else:
            st.markdown("### Cinemática sagital 2D proyectada")
            for title, cols in [
                ("Rodillas", ["Rodilla izquierda","Rodilla derecha"]),
                ("Caderas", ["Cadera izquierda","Cadera derecha"]),
                ("Tobillos", ["Tobillo izquierda","Tobillo derecha"]),
                ("Hombros", ["Hombro izquierda","Hombro derecha"]),
            ]:
                st.markdown(f"#### {title}")
                st.line_chart(chart.set_index("time_s")[cols], x_label="Tiempo (s)", y_label="Ángulo 2D (°)")
        st.markdown("### Alternancia distal D/I")
        st.line_chart(chart.set_index("time_s")[["Alternancia D-I"]], x_label="Tiempo (s)", y_label="Diferencia vertical relativa (px)")
        with st.expander("Todas las métricas y calidad"):
            st.dataframe(pd.DataFrame(metrics)[["label","value","unit","quality","notes"]], use_container_width=True, hide_index=True)
        if st.session_state.get("annotated_video_bytes"):
            st.markdown("### Vídeo con esqueleto + ángulos")
            st.video(st.session_state.annotated_video_bytes)
            st.caption("El vídeo está únicamente en memoria temporal de esta sesión; no se guarda en Supabase.")

with tabs[4]:
    st.subheader("Pacientes / Evolución")
    if not sb_ready():
        st.info("Configura Supabase para activar el histórico persistente.")
    else:
        try:
            pats = sb_list_patients()
            if pats.empty:
                st.info("Todavía no hay pacientes en el histórico.")
            else:
                codes = pats.code.tolist(); default = codes.index(patient) if patient in codes else 0
                selected = st.selectbox("Paciente / código", codes, index=default)
                hist = sb_patient_history(selected)
                same_aid = st.checkbox("Comparar solo sesiones con la misma ayuda técnica", value=True)
                if same_aid and not hist.empty:
                    current_aid = st.session_state.get("assistive_device", assistive_device)
                    if "assistive_device" in hist.columns:
                        hist = hist[hist.assistive_device.fillna("Sin ayuda") == current_aid].copy()
                if hist.empty:
                    st.info("Este paciente todavía no tiene métricas guardadas.")
                else:
                    st.metric("Sesiones con resultados", hist["session_id"].nunique())
                    labels = hist[["metric_key","metric_label","unit"]].drop_duplicates().sort_values("metric_label")
                    label = st.selectbox("Parámetro para evolución", labels.metric_label.tolist())
                    key = labels.loc[labels.metric_label==label,"metric_key"].iloc[0]
                    h = hist[hist.metric_key==key].copy().sort_values("created_at")
                    h["fecha"] = pd.to_datetime(h.created_at, utc=True).dt.strftime("%d/%m/%Y %H:%M")
                    unit = h.unit.iloc[0] if len(h) else ""
                    st.line_chart(h.set_index("fecha")[["value"]].rename(columns={"value":label}), x_label="Registro", y_label=f"{label} ({unit})" if unit else label)
                    if len(h)>=2 and pd.notna(h.value.iloc[0]) and pd.notna(h.value.iloc[-1]):
                        first,last = float(h.value.iloc[0]),float(h.value.iloc[-1]); delta=last-first
                        a,b,c=st.columns(3); a.metric("Primer registro",f"{first:.2f} {unit}"); b.metric("Último registro",f"{last:.2f} {unit}",delta=f"{delta:+.2f}")
                        c.metric("Cambio vs basal",f"{(delta/first*100):+.1f} %" if first!=0 else "—")
                    st.markdown("### Sesiones")
                    cols = ["id","created_at","record_name","mode","view","assistive_device","frontal_orientation","duration_s","segment_start_s","segment_end_s","analysis_status"]
                    st.dataframe(hist[cols].drop_duplicates().sort_values("created_at",ascending=False), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"No se pudo leer el histórico: {e}")

with tabs[5]:
    st.subheader("3D futuro")
    st.write("La arquitectura queda preparada para calibración, sincronización, triangulación Pose2Sim y cinemática OpenSim.")
    st.warning("Los ángulos actuales son 2D proyectados. No deben interpretarse como ángulos anatómicos 3D.")

st.divider()
st.caption("PhysioSentinel Gait v0.6 · Supabase persistente · vídeos temporales · biomecánica frontal proyectada · ayudas técnicas")
