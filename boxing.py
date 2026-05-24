import cv2
import mediapipe as mp
import streamlit as st
import math
import time
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 计算角度
def calc_angle(a,b,c):
    a=[a.x,a.y]
    b=[b.x,b.y]
    c=[c.x,c.y]
    rad=math.atan2(c[1]-b[1],c[0]-b[0])-math.atan2(a[1]-b[1],a[0]-b[0])
    ang=abs(rad*180/math.pi)
    return min(ang,360-ang)

# 计算两点位移
def get_move_dist(p1,p2):
    return math.hypot(p1.x-p2.x, p1.y-p2.y)

st.set_page_config(page_title="🥊拳击AI精准动作纠正",layout="wide")
st.title("🥊拳击AI动作实时纠正系统")
c1,c2=st.columns([2,1])
frame=c1.empty()
info=c2.empty()

# 全局变量
cnt = 0
last_t = time.time()
cool_down = 0.6
base_right_shoulder = None
base_left_shoulder = None
base_right_wrist = None
base_left_wrist = None
punch_state = 0
standard_action = False
encourage_text = ""

cap=cv2.VideoCapture(0)
while cap.isOpened():
    ret,f=cap.read()
    if not ret:break
    h,w=f.shape[:2]
    f=cv2.flip(f,1)
    rgb=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
    res=pose.process(rgb)
    act="静止"
    ra=0;la=0
    tip_text=""

    if res.pose_landmarks:
        lm=res.pose_landmarks
        rs=lm.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        re=lm.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        rw=lm.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        ls=lm.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        le=lm.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        lw=lm.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

        ra=calc_angle(rs,re,rw)
        la=calc_angle(ls,le,lw)

        if base_right_shoulder is None:
            base_right_shoulder = rs
            base_left_shoulder = ls
            base_right_wrist = rw
            base_left_wrist = lw

        shoulder_move = max(get_move_dist(rs,base_right_shoulder), get_move_dist(ls,base_left_shoulder))
        fist_move = max(get_move_dist(rw,base_right_wrist), get_move_dist(lw,base_left_wrist))
        is_big_move = (shoulder_move > 0.08) and (fist_move > 0.22)

        if is_big_move:
            if ra>165 or la>165:
                act="直拳"
                if abs(rs.y-re.y)<0.1 and get_move_dist(rw,rs)>0.3:
                    standard_action=True
                    encourage_text="✅直拳动作标准！手臂打直，发力到位！继续保持！"
                else:
                    standard_action=False
                    tip_text="⚠️直拳：手肘抬高/出拳距离不足，手臂尽量打直！"
            elif ra<85 or la<85:
                act="勾拳"
                if abs(re.x-rs.x)>0.1 and get_move_dist(rw,rs)>0.25:
                    standard_action=True
                    encourage_text="✅勾拳动作标准！抬肘到位，发力充分！太专业了！"
                else:
                    standard_action=False
                    tip_text="⚠️勾拳：抬肘幅度不足，手肘向外打开！"
            elif 105<ra<155 or 105<la<155:
                act="摆拳"
                if get_move_dist(rs,ls)>0.15 and get_move_dist(rw,rs)>0.28:
                    standard_action=True
                    encourage_text="✅摆拳动作标准！转肩充分，发力流畅！很棒！"
                else:
                    standard_action=False
                    tip_text="⚠️摆拳：转肩幅度不够，加大转体发力！"

        now=time.time()
        if punch_state == 0 and is_big_move:
            punch_state = 1
        elif punch_state == 1 and not is_big_move:
            punch_state = 2
            if now - last_t > cool_down:
                cnt +=1
                last_t = now
                punch_state = 0
        elif punch_state ==2:
            punch_state=0

        mp_drawing.draw_landmarks(f,res.pose_landmarks,mp_pose.POSE_CONNECTIONS)

    feedback = encourage_text if standard_action else tip_text if tip_text else "✅姿态稳定，准备出拳！"
    info.markdown(f"""
### 📊实时运动分析
- 当前动作：**{act}**
- 累计出拳：**{cnt}次**（打出+收回完整计数）
- 右手肘：{int(ra)}° | 左手肘：{int(la)}°

### 🎯动作反馈
{feedback}
> 判定标准：肩膀大幅位移+拳头完整打出收回才算有效一拳
""")
    frame.image(cv2.cvtColor(f,cv2.COLOR_BGR2RGB),channels="RGB")
cap.release()