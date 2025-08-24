# server.py
import random
import numpy as np
import cv2
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import math
from fastapi import Response   # <-- 이거 없으면 /rps_points가 500난다

app = FastAPI()  # <- uvicorn이 찾을 app 객체

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 배포 시 특정 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

FINGERS_TIP = [8, 12, 16, 20] #검지/중지/약지/소지의 끝 마디(TIP) 인덱스
FINGERS_PIP = [6, 10, 14, 18] #PIP(두 번째 마디) 인덱스를 짝지어 둔 것.
FINGERS_MCP = [5, 9, 13, 17]   # 각 손가락의 MCP(뿌리) 인덱스

RPS = ["Rock", "Paper", "Scissors"]

app.state.computer = random.choice(RPS)

def angle_at(p_mcp, p_pip, p_tip):
    #하나의 손가락에는 MCP(뿌리) → PIP(두 번째 마디) → TIP(끝) 세 점이 있음.
    # 벡터: MCP->PIP, TIP->PIP
    v1 = (p_mcp.x - p_pip.x, p_mcp.y - p_pip.y)
    v2 = (p_tip.x - p_pip.x, p_tip.y - p_pip.y)
    n1 = math.hypot(*v1); n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos = (v1[0]*v2[0] + v1[1]*v2[1]) / (n1*n2)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))  # 0~180°

def rps_from_landmarks(lm, w, h):
    if lm is None:
        return "No Hand", [False, False, False, False]

    # 각도 기준: 160° 이상이면 '펴짐'으로 보자 (160~170 사이 튜닝)
    TH = 160.0
    up = []
    for mcp, pip, tip in zip(FINGERS_MCP, FINGERS_PIP, FINGERS_TIP):
        ang = angle_at(lm[mcp], lm[pip], lm[tip])
        up.append(ang >= TH)

    idx, mid, ring, pinky = up
    cnt = idx + mid + ring + pinky

    if cnt == 0:
        return "Rock", up
    if idx and mid and not ring and not pinky:
        return "Scissors", up
    if cnt == 4:
        return "Paper", up
    return "Unknown", up
def draw_mcp_pip_tip(bgr, lm, w, h, draw_angles=True):
    finger_names = ["Index", "Middle", "Ring", "Pinky"]
    for fname, mcp_i, pip_i, tip_i in zip(finger_names, FINGERS_MCP, FINGERS_PIP, FINGERS_TIP):
        mcp = lm[mcp_i]; pip = lm[pip_i]; tip = lm[tip_i]
        mx, my = int(mcp.x * w), int(mcp.y * h)
        px, py = int(pip.x * w), int(pip.y * h)
        tx, ty = int(tip.x * w), int(tip.y * h)

        # 선(뼈대)
        cv2.line(bgr, (mx, my), (px, py), (255, 255, 0), 2)  # MCP-PIP
        cv2.line(bgr, (px, py), (tx, ty), (255, 255, 0), 2)  # PIP-TIP

        # 점(색상: MCP-파랑, PIP-노랑, TIP-빨강)
        cv2.circle(bgr, (mx, my), 6, (255,   0,   0), -1)  # MCP
        cv2.circle(bgr, (px, py), 6, (  0, 255, 255), -1)  # PIP
        cv2.circle(bgr, (tx, ty), 6, (  0,   0, 255), -1)  # TIP

        # 라벨
        cv2.putText(bgr, f"{fname[0]}-MCP", (mx+6, my-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0),   1, cv2.LINE_AA)
        cv2.putText(bgr, f"{fname[0]}-PIP", (px+6, py-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bgr, f"{fname[0]}-TIP", (tx+6, ty-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255),   1, cv2.LINE_AA)

        # 각도(선택)
        if draw_angles:
            ang = angle_at(mcp, pip, tip)
            cv2.putText(bgr, f"{int(round(ang))}°", (px+6, py+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2, cv2.LINE_AA)

@app.post("/rps_points")
async def rps_points(file: UploadFile = File(...)):
    data = await file.read()
    arr  = np.frombuffer(data, np.uint8)
    bgr  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return JSONResponse({"error": "decode_failed"}, status_code=400)

    h, w = bgr.shape[:2]
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res  = hands.process(rgb)

    annotated = bgr.copy()
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        # MCP/PIP/TIP 점 + 선 + 각도 그리기
        draw_mcp_pip_tip(annotated, lm, w, h, draw_angles=True)

    ok, buf = cv2.imencode(".jpg", annotated)
    if not ok:
        return JSONResponse({"error": "encode_failed"}, status_code=500)
    return Response(content=buf.tobytes(), media_type="image/jpeg")

# def rps_from_landmarks(landmarks, w, h):
#     if landmarks is None:
#         return "No Hand", [False, False, False, False]
#     up = []
#     for tip, pip in zip(FINGERS_TIP, FINGERS_PIP):
#         tip_y = landmarks[tip].y * h
#         pip_y = landmarks[pip].y * h
#         up.append(tip_y < pip_y)
#     cnt = sum(up)
#     idx, mid, ring, pinky = up #검지, 중지, 약지, 소지
#     if idx and mid and not ring and not pinky:
#         return "Scissors", up
#     if not idx and not mid and ring and pinky:
#         return "Scissors",up
#     if cnt == 0:
#         return "Rock", up
#     if cnt == 4:
#         return "Paper", up
#     if not idx and not mid and not ring and not pinky:
#         return "Paper", up
#     return "Unknown", up

def decide(user, computer):
    if user in ("No Hand", "Unknown"): return ""
    if user == computer: return "draw"
    if (computer == "Rock" and user == "Scissors") or \
       (computer == "Scissors" and user == "Paper") or \
       (computer == "Paper" and user == "Rock"):
        return "lose"
    return "win"

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/start_round")
def start_round():
    app.state.computer = random.choice(RPS)
    return {"computer": app.state.computer}

@app.post("/rps")
async def rps(file: UploadFile = File(...)):
    data = await file.read()
    arr  = np.frombuffer(data, np.uint8)
    bgr  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return JSONResponse({"error": "decode_failed"}, status_code=400)

    h, w = bgr.shape[:2]
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res  = hands.process(rgb)

    user, up = "No Hand", [False, False, False, False]
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        user, up = rps_from_landmarks(lm, w, h)

    computer = app.state.computer
    result   = decide(user, computer)
    return {"gesture": user, "fingers_up": up, "computer": computer, "result": result}

# 정적 파일 mount (web 폴더 전체 제공)
app.mount("/web", StaticFiles(directory="web"), name="web")

# 루트("/") 접근 시 index.html 반환
@app.get("/")
def read_root():
    return FileResponse(os.path.join("web", "index.html"))