# Run streamlit
# streamlit run GradCero.py

# ========================
# Importing Fields
# ========================
import base64
import math
import time
import qrcode
import smtplib
from email.message import EmailMessage
import os
import cv2 as cv
import pandas as pd
import numpy as np
import face_recognition
from ultralytics import YOLO
from pyzbar.pyzbar import decode
import streamlit as st
from streamlit_option_menu import option_menu
from background import set_bg
from gtts import gTTS
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# ---- WebRTC camera setup ----
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class CameraProcessor(VideoProcessorBase):
    def __init__(self):
        self.latest_frame = None  # np.ndarray (BGR)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_frame = img
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def init_camera():
    # Create the WebRTC component exactly once and reuse the ctx
    return webrtc_streamer(
        key="camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=CameraProcessor,
    )

def get_frame(ctx):
    # Returns np.ndarray (BGR) or None if not ready yet
    if ctx and ctx.video_processor:
        return ctx.video_processor.latest_frame
    return None


# ==============================================================================
# macOS required only due to not self-located, comment section this on Windows
# ==============================================================================
# import ctypes

# zbar_path = "/opt/homebrew/lib/libzbar.dylib"
# if os.path.exists(zbar_path):
#     ctypes.cdll.LoadLibrary(zbar_path)
#     import pyzbar.zbar_library
#     pyzbar.zbar_library.load = lambda: (ctypes.cdll.LoadLibrary(zbar_path), [])

# ========================
# QR Generator - For all
# ========================

# SMTP Config

# sender_email = "jhlee-wm22@student.tarc.edu.my"
# sender_pass = "moseepjnzqxsdttn"


def generate_qr():
    sender_email = "ungms-wm22@student.tarc.edu.my"
    sender_pass = "dumzitmmuvknydhm"
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email, sender_pass)

    # Load Excel
    df = pd.read_excel("studentdb.xlsx")
    output_dir = "qr_codes"
    os.makedirs(output_dir, exist_ok=True)

    for _, row in df.iterrows():
        student_id = row['student_id']
        name = row['name']
        email = row['email']

        # Data to store in QR
        if pd.isna(email) or "@" not in str(email):
            st.warning(
                f"Skipping {name} (ID: {student_id}) — invalid or missing email")
            continue
        qr_data = f"{student_id}|{name}"

        # Create QR
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4
        )
        qr.add_data(qr_data)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")

        # Save QR at local
        qr_path = os.path.join(output_dir, f"{student_id}.png")
        img.save(os.path.join(output_dir, f"{student_id}.png"))

        # Email body
        msg = EmailMessage()
        msg["Subject"] = f"Your Student QR Code ({student_id})"
        msg["From"] = sender_email
        msg["To"] = email
        msg.set_content(
            f"Hello {name},\n\nAttached is your student QR code for graduation ceremony.\n\nBest regards, \n\n TARUMT"
        )

        with open(qr_path, "rb") as f:
            file_data = f.read()
            file_name = f"{student_id}.png"
            msg.add_attachment(file_data, maintype="image",
                               subtype="png", filename=file_name)

        # Send email
        server.send_message(msg)
        st.success(f"Sent QR to {name} ({email})")

    server.quit()
    st.info("QR codes generated for all students.")

# ========================
# QR Generator - For Ind
# ========================


def generate_ind_qr(student_id, name, email):

    sender_email = "ungms-wm22@student.tarc.edu.my"
    sender_pass = "dumzitmmuvknydhm"
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email, sender_pass)

    if pd.isna(email) or "@" not in str(email):
        st.warning(
            f"Skipping {name} (ID: {student_id}) — invalid or missing email")
        return
    qr_data = f"{student_id}|{name}"

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4
    )
    qr.add_data(qr_data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    output_dir = "qr_codes"
    os.makedirs(output_dir, exist_ok=True)
    qr_path = os.path.join(output_dir, f"{student_id}.png")
    img.save(qr_path)

    msg = EmailMessage()
    msg["Subject"] = f"Your Student QR Code ({student_id})"
    msg["From"] = sender_email
    msg["To"] = email
    msg.set_content(
        f"Hello {name},\n\nAttached is your student QR code for graduation ceremony.\n\nBest regards,\n\nTARUMT"
    )

    with open(qr_path, "rb") as f:
        file_data = f.read()
        msg.add_attachment(file_data, maintype="image",
                           subtype="png", filename=f"{student_id}.png")

    server.send_message(msg)

    server.quit()

# ========================
# QR Code Scan
# ========================


def scan_qr_and_get_student(ctx):
    df = pd.read_excel("studentdb.xlsx")
    st.info("Scanning for QR code...")
    st_frame = st.empty()

    student_id, name, course, image_bytes = None, None, None, None

    while True:
        frame = get_frame(ctx)
        if frame is None:
            time.sleep(0.05)
            continue

        # Decode QR on the BGR frame (pyzbar works fine with this)
        decoded_objs = decode(frame)
        if decoded_objs:
            for obj in decoded_objs:
                qr_data = obj.data.decode('utf-8').strip()
                parts = qr_data.split('|')
                if len(parts) < 2:
                    st.warning(f"QR code format invalid: '{qr_data}'")
                    continue

                student_id, name = parts[0].strip(), parts[1].strip()
                match = df[(df['student_id'] == student_id) & (df['name'] == name)]

                if not match.empty:
                    course = match.iloc[0]['course']
                    image_path = match.iloc[0]['image_path']

                    if not os.path.exists(image_path):
                        st.error(f"Image file not found: {image_path}")
                        return None, None, None, None

                    with open(image_path, "rb") as f:
                        image_bytes = f.read()

                    st.success("Match found in Excel.")
                    return student_id, name, course, image_bytes
                else:
                    st.error("No match found in Excel.")

        st_frame.image(frame, channels="BGR", caption="QR Scanner")


# ========================
# Face Recognition
# ========================

def face_match_with_qr(proper_name, ref_image_bytes, ctx):
    TOLERANCE = 0.50

    def ui_conf(distance, thresh=TOLERANCE):
        return 100.0 / (1.0 + math.exp(8 * (float(distance) - float(thresh))))

    # Prepare reference encoding from bytes (unchanged)
    nparr = np.frombuffer(ref_image_bytes, np.uint8)
    ref_image = cv.imdecode(nparr, cv.IMREAD_COLOR)
    ref_rgb = cv.cvtColor(ref_image, cv.COLOR_BGR2RGB)
    ref_encodings = face_recognition.face_encodings(ref_rgb)
    if not ref_encodings:
        st.error("No face found in reference image.")
        return None
    ref_encoding = ref_encodings[0]

    yolo_model = YOLO("yolov8n-face.pt")
    matched = None
    countdown = 3
    st.info("Adjust your face... capturing in 3 seconds")
    st_frame = st.empty()

    # Countdown with live frames
    while countdown > 0:
        frame = get_frame(ctx)
        if frame is not None:
            temp = frame.copy()
            cv.putText(temp, f"Capturing in {countdown}s", (50, 70),
                       cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 127), 3)
            st_frame.image(temp, channels="BGR", caption="Face Recognition")
        time.sleep(1)
        countdown -= 1

    # Final capture
    frame = get_frame(ctx)
    if frame is None:
        st.error("No camera frame received.")
        return None

    raw_frame = frame.copy()
    results = yolo_model(frame, verbose=False)
    largest_area, best_box = 0, None

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            area = (x2 - x1) * (y2 - y1)
            if area > largest_area:
                largest_area, best_box = area, (x1, y1, x2, y2)

    display_name, label_extra, color = "Unknown", "", (0, 0, 255)

    if best_box:
        x1, y1, x2, y2 = best_box
        face_roi = raw_frame[y1:y2, x1:x2]
        rgb_face = cv.cvtColor(face_roi, cv.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_face)

        if encodings:
            face_distances = float(face_recognition.face_distance([ref_encoding], encodings[0])[0])
            conf = ui_conf(face_distances, TOLERANCE)
            is_match = face_distances <= TOLERANCE

            if is_match:
                display_name, color, matched = proper_name, (34, 139, 34), True
                st.success(f"Face matched: {display_name} (dist = {face_distances:.3f} | conf ~ {conf:.0f}%)")
                announce_name(display_name)
            else:
                st.error(f"Face does not match reference (dist = {face_distances:.3f} | conf ~ {conf:.0f}%)")
                matched = False

            label_extra = f" | dist = {face_distances:.3f} | conf ~ {conf:.0f}%"
        else:
            st.warning("No face encoding from camera frame. Try better lighting / frontal pose.")
            matched = None

        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv.putText(frame, f"{display_name}{label_extra}", (x1, max(20, y1 - 10)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        st_frame.image(frame, channels="BGR", caption="Face Recognition")
    else:
        st.warning("No face detected. Please move closer and face the camera.")
        st_frame.image(frame, channels="BGR", caption="Face Recognition")
        matched = None

    return matched

# ========================
# Text to Speech
# ========================


def announce_name(student_name):
    text = f"Now presenting, {student_name}"
    tts = gTTS(text=text, lang="en", tld="co.uk")

    mp3_buffer = BytesIO()
    tts.write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)

    b64 = base64.b64encode(mp3_buffer.read()).decode()
    autoplay_audio = f"""
        <audio autoplay controls>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """

    st.markdown(autoplay_audio, unsafe_allow_html=True)
    mp3_buffer.seek(0)
    st.audio(mp3_buffer, format="audio/mp3")

# ========================
# Main UI
# ========================


st.set_page_config(page_title="Graduation Ceremony System",
                   page_icon="🎓", layout="wide")

with st.sidebar:
    menu = option_menu(
        menu_title=None,
        options=["Dashboard (Scan & Verification)",
                 "QR Generator", "Admin View"],
        icons=["house", "qr-code", "people-fill"],
        default_index=0
    )

# =================================================
# Successful Result Pop Up After Facial Recognition
# =================================================

if "show_result" not in st.session_state:
    st.session_state["show_result"] = False
if "result_show_time" not in st.session_state:
    st.session_state["result_show_time"] = None


@st.dialog("Verification Result")
def show_success_result(student_id, name, course, image_path):

    if st.session_state["result_show_time"] is None:
        st.session_state["result_show_time"] = time.time()

    elapsed_time = time.time() - st.session_state["result_show_time"]
    remaining_time = max(0, 3 - elapsed_time)

    if remaining_time > 0:
        st.markdown(
            "<p style='text-align:center; font-size:1.1rem; "
            "background:#dcfce7; color:#166534; padding:8px 12px; "
            "border-radius:8px; font-weight:700; margin:10px;'>Successful</p>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div style='display: flex; justify-content: center; margin: 20px 0;'>"
            f"<img src='data:image/png;base64,{base64.b64encode(image_path).decode()}' "
            f"style='width: 600px; max-width: 200px; max-height: 200px; object-fit:contain; border-radius: 8px;'>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<p style='text-align:center; font-size:1.1rem;'><b>Student ID:</b> {student_id}</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align:center'><b>Name:</b> {name}</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align:center'><b>Course:</b> {course}</p>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align:center; color:#16a34a; font-weight:700'>Status: Verified</p>", unsafe_allow_html=True)

        st.markdown(
            f"<p style='text-align:center; color:#666; margin-top:20px;'>Auto-closing in {remaining_time:.1f} seconds...</p>", unsafe_allow_html=True)
        time.sleep(3)
        st.rerun()
    else:
        st.session_state["result_show_time"] = None
        st.session_state["student_id"] = None
        st.session_state["name"] = None
        st.session_state["image_path"] = None
        st.session_state["face_verified"] = None
        st.session_state["scanning_started"] = True
        st.session_state["show_result"] = False
        st.rerun()

# =================================================
# Failed Result Pop Up After Facial Recognition
# =================================================


if "show_result" not in st.session_state:
    st.session_state["show_result"] = False
if "result_show_time" not in st.session_state:
    st.session_state["result_show_time"] = None


@st.dialog("Verification Result")
def show_failed_result():

    if st.session_state["result_show_time"] is None:
        st.session_state["result_show_time"] = time.time()

    elapsed_time = time.time() - st.session_state["result_show_time"]
    remaining_time = max(0, 3 - elapsed_time)

    if remaining_time > 0:
        st.markdown(
            "<p style='text-align:center; font-size:1.1rem; "
            "background:#fcdcdc; color:#651616; padding:8px 12px; "
            "border-radius:8px; font-weight:700; margin:10px;'>Failed</p>",
            unsafe_allow_html=True
        )

        with open("facecross.png", "rb") as f:
            cross_icon = base64.b64encode(f.read()).decode()

        st.markdown(
            f"""
            <div style='display:flex; justify-content:center; align-items:center;
                        border:2px dashed #991b1b; border-radius:10px;
                        width:430px; height:200px; margin: 20px auto;
                        flex-direction:column; background:#fff5f5;'>
                <div style='font-size:80px; color:#dc2626;'>
                    <img src="data:image/png;base64,{cross_icon}" style="width:100px;height:100px;display:block;margin-top:-15px;"/>
                </div>
                <div style='font-size:20px; color:#991b1b; font-weight:600;margin-top:5px;'>
                    Failed to Verify
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "<p style='text-align:center; color:#a31616; font-weight:700'>Status: Not Verified</p>", unsafe_allow_html=True)

        st.markdown(
            f"<p style='text-align:center; color:#666; margin-top:20px;'>Auto-closing in {remaining_time:.1f} seconds...</p>", unsafe_allow_html=True)
        time.sleep(3)
        st.rerun()
    else:
        st.session_state["result_show_time"] = None
        st.session_state["student_id"] = None
        st.session_state["name"] = None
        st.session_state["image_path"] = None
        st.session_state["face_verified"] = None
        st.session_state["scanning_started"] = True
        st.session_state["show_result"] = False
        st.rerun()

# =======================================================
# No Face Detected Result Pop Up After Facial Recognition
# =======================================================


if "show_result" not in st.session_state:
    st.session_state["show_result"] = False
if "result_show_time" not in st.session_state:
    st.session_state["result_show_time"] = None


@st.dialog("Verification Result")
def show_noDetect_result():

    if st.session_state["result_show_time"] is None:
        st.session_state["result_show_time"] = time.time()

    elapsed_time = time.time() - st.session_state["result_show_time"]
    remaining_time = max(0, 3 - elapsed_time)

    st_frame = st.empty()

    if remaining_time > 0:
        st.markdown(
            "<p style='text-align:center; font-size:1.1rem; "
            "background:#fcebdc; color:#654016; padding:8px 12px; "
            "border-radius:8px; font-weight:700; margin:10px;'>No Face Detection</p>",
            unsafe_allow_html=True
        )

        with open("facewarn.png", "rb") as f:
            warn_icon = base64.b64encode(f.read()).decode()

        st.markdown(
            f"""
            <div style='display:flex; justify-content:center; align-items:center;
                        border:2px dashed #e3740b; border-radius:10px;
                        width:430px; height:200px; margin: 20px auto;
                        flex-direction:column; background:#fffbf5;'>
                <div style='font-size:80px; color:#dc8426;'>
                    <img src="data:image/png;base64,{warn_icon}" style="width:100px;height:100px;display:block;margin-top:-15px;"/>
                </div>
                <div style='font-size:20px; color:#e3740b; font-weight:600;margin-top:5px;'>
                    No Face Detected, Failed to Verify
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "<p style='text-align:center; color:#d98211; font-weight:700'>Status: No Face Verified</p>", unsafe_allow_html=True)

        st.markdown(
            f"<p style='text-align:center; color:#666; margin-top:20px;'>Auto-closing in {remaining_time:.1f} seconds...</p>", unsafe_allow_html=True)
        time.sleep(3)
        st.rerun()
    else:
        st.session_state["result_show_time"] = None
        st.session_state["student_id"] = None
        st.session_state["name"] = None
        st.session_state["image_path"] = None
        st.session_state["face_verified"] = None
        st.session_state["scanning_started"] = True
        st.session_state["show_result"] = False
        st.rerun()

# ================================================
# Dashboard (QR Scan + Face Recognition)
# ================================================


if menu == "Dashboard (Scan & Verification)":

    set_bg("bckground/graduation_bg.jpg")

    # 🔑 Create WebRTC camera once here
    camera_ctx = webrtc_streamer(
        key="camera",
        mode="sendrecv",
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=CameraProcessor,
    )

    st.markdown(
        """
        <h2 style='text-align: center; padding:0 0 30px 0;'>Scan & Verify Students</h1><hr style='margin:0 0 40px 0;'>""",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        div.stButton > button {
            display: block;
            margin: 0 auto;
            border: 1px solid #57ffe0 !important;
            color: #57ffe0 !important;
        }
        div.stButton > button:hover {
            border: 1px solid #fae100 !important;
            color: #fae100 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # Session state, avoid redundant result for different person
    st.session_state.setdefault("scanning_started", False)
    st.session_state.setdefault("student_id", None)
    st.session_state.setdefault("face_verified", None)
    st.session_state.setdefault("show_result", False)

    with col1:
        header_col, btn_col = st.columns([3, 1])
        with header_col:
            st.subheader("QR Code Scanner")

        def start_scanning():
            st.session_state["scanning_started"] = True
            st.session_state["student_id"] = None
            st.session_state["face_verified"] = None
            st.session_state["show_result"] = False

        def stop_scanning():
            st.session_state["scanning_started"] = False
            st.session_state["student_id"] = None
            st.session_state["face_verified"] = None
            st.session_state["show_result"] = False

        # Create ctx once when scanning begins
        if st.session_state["scanning_started"] and "cam_ctx" not in st.session_state:
            st.session_state["cam_ctx"] = init_camera()

        with btn_col:
            if st.session_state["scanning_started"]:
                st.button("Stop Scanning", key="stop-btn",
                          on_click=stop_scanning)
            else:
                st.button("Start Scanning", key="start-btn",
                          on_click=start_scanning)

            if st.session_state["scanning_started"] and st.session_state.get("student_id") is None:
                student_id, name, course, image_bytes = scan_qr_and_get_student(st.session_state["cam_ctx"])
                if student_id:
                    st.session_state["student_id"] = student_id
                    st.session_state["name"] = name
                    st.session_state["course"] = course
                    st.session_state["image_path"] = image_bytes
                    st.success("QR Found!")
    with col2:
        st.subheader("Facial Recognition")

        with st.container() as face_scan_container:
            if st.session_state["student_id"]:
                if st.session_state["face_verified"] is None:
                    result = face_match_with_qr(
                        st.session_state["name"],
                        st.session_state["image_path"],
                        st.session_state["cam_ctx"],     # <--- pass camera ctx here`
                    )
        st.session_state["face_verified"] = result
        if st.session_state["face_verified"] is True:
            show_success_result(
                st.session_state["student_id"],
                st.session_state["name"],
                st.session_state["course"],
                st.session_state["image_path"]
            )
        elif st.session_state["face_verified"] is False:
            show_failed_result()
        else:
            st.session_state["face_verified"] = "no_face"
            show_noDetect_result()

# ========================
# QR Generator UI
# ========================

elif menu == "QR Generator":

    set_bg("bckground/qrcode_bg.jpg")

    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>QR Generator</h1>
            <p>Generate QR codes for students.</p>
            <hr style='margin:10px 0 10px 0;'>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        div.stButton > button { 
            display: block; 
            margin: 0 auto; 
            border: 1px solid #57ffe0 !important; 
            color: #57ffe0 !important;
        }
        div.stButton > button:hover {
            border: 1px solid #fae100 !important; 
            color: #fae100 !important;
        }

        div[data-testid="stSpinner"] > div { display: flex; justify-content: center; }
        div[data-testid="stSpinner"] > div > div { text-align: center; }
        </style>
        """,
        unsafe_allow_html=True
    )

    if st.button("Generate QR Codes"):
        with st.spinner("Generating QR codes and sending emails..."):
            try:
                generate_qr()
            except Exception as e:
                st.error(f"Error: {e}")

# ================================================
# Admin Site ((Will separate out in the future))
# ================================================

elif menu == "Admin View":

    set_bg("bckground/admin_bg.jpg")

    st.markdown(
        "<h1 style='text-align: center;'>Admin View</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center;'>View and monitor student database.</p>",
        unsafe_allow_html=True
    )

    # CUR VER modified from code set above, but image path combined with image shown
    if os.path.exists("studentdb.xlsx"):
        df = pd.read_excel("studentdb.xlsx")

        df.insert(0, "No",  range(1, len(df) + 1))

        df = df.rename(columns={
            'student_id': 'Student ID',
            'name': 'Name',
            'image_path': 'Image',
            'email': 'Email',
        })

        st.markdown("<div style='margin-top: 20px;'>", unsafe_allow_html=True)

        col1, col2, col3, col4, col5, col6 = st.columns([0.5, 1, 1.5, 2, 2, 1])

        header_style = "background-color: rgba(14, 17, 23, 0.8); color:white; text-align:center; font-weight:bold; padding:12px 0; border-radius:5px;"

        headers = ["No", "Student ID", "Name",
                   "Image", "Email", "QR Generator"]
        for col, header in zip([col1, col2, col3, col4, col5, col6], headers):
            col.markdown(
                f"<div style='{header_style}'>{header}</div>", unsafe_allow_html=True)

        st.divider()

        if "img_toggle" not in st.session_state:
            st.session_state["img_toggle"] = {}

        for i, (idx, row) in enumerate(df.iterrows(), 1):
            col1, col2, col3, col4, col5, col6 = st.columns(
                [0.5, 1, 1.5, 2, 2, 1])

            col1.markdown(
                f"<div style='text-align:center; font-weight:bold;'>{i}</div>", unsafe_allow_html=True)
            col2.markdown(
                f"<div style='text-align:center; font-weight:bold;'>{row['Student ID']}</div>", unsafe_allow_html=True)
            col3.markdown(
                f"<div style='text-align:center; font-weight:bold;'>{row['Name']}</div>", unsafe_allow_html=True)

            key = f"img_{i}"
            if key not in st.session_state["img_toggle"]:
                st.session_state["img_toggle"][key] = False

            with col4:
                key = f"img_toggle_{i}"
                if key not in st.session_state:
                    st.session_state[key] = False

                if st.button(row['Image'].split('/')[-1], key=f"btn_{i}"):
                    st.session_state[key] = not st.session_state[key]

                if st.session_state[key]:
                    try:
                        st.image(row['Image'], use_container_width=True)
                    except:
                        st.warning("Image not found")

            col5.markdown(
                f"<div style='text-align:center; font-weight:bold;'>{row['Email']}</div>", unsafe_allow_html=True)

            with col6:
                if st.button("Send QR", key=f"send_qr_{row['Student ID']}"):
                    with st.spinner(f"Generating QR for {row['Name']}..."):
                        generate_ind_qr(row['Student ID'],
                                        row['Name'], row['Email'])
                    st.markdown(
                        "<div style='text-align:center; color:#155724; background-color:#d4edda; "
                        "border:1px solid #c3e6cb; border-radius:5px; padding:8px;'>QR sent!</div>",
                        unsafe_allow_html=True
                    )

            if i < len(df):
                st.divider()

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <style>
                .stButton > button {
                    background-color: #57ffe0 !important;
                    color: black !important;
                    border: none;
                    border-radius: 5px;
                    padding: 5px 5px;
                    font-weight: bold;
                    width: 100%;
                }
                .stButton > button:hover {
                    background-color: #fae100 !important;
                    color: black !important;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("No database found.")
