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
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import threading
import av
from streamlit_autorefresh import st_autorefresh

RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    # If your campus network is strict, add a TURN server here.
}
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
qr_lock = threading.Lock()  # avoid race when updating session_state from processor

class QRScanner(VideoProcessorBase):
    def __init__(self):
        self.det = cv.QRCodeDetector()
        self.last_qr = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Try multi first
        ok, decoded_info, points, _ = self.det.detectAndDecodeMulti(img)
        if not ok:
            data, pts, _ = self.det.detectAndDecode(img)
            decoded_info = [data] if data else []
            points = [pts] if pts is not None and data else None

        # draw polygon(s)
        if points is not None and len(points) > 0:
            for pts in points:
                if pts is None or len(pts) == 0: 
                    continue
                cv.polylines(img, [pts.astype(int)], True, (0, 255, 255), 2)

        # handle found QR(s)
        for data in decoded_info:
            if not data:
                continue
            qr_data = data.strip()
            if self.last_qr == qr_data:
                break  # debounce same QR spam
            self.last_qr = qr_data

            parts = qr_data.split("|")
            if len(parts) >= 2:
                student_id, name = parts[0].strip(), parts[1].strip()

                try:
                    df = pd.read_excel("studentdb.xlsx")
                except Exception as e:
                    cv.putText(img, f"DB error: {e}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    break

                match = df[(df["student_id"] == student_id) & (df["name"] == name)]
                if not match.empty:
                    course = match.iloc[0]["course"]
                    img_path = str(match.iloc[0]["image_path"])
                    if os.path.exists(img_path):
                        with open(img_path, "rb") as f:
                            image_bytes = f.read()
                        # Update session state to advance the flow
                        with qr_lock:
                            st.session_state["student_id"] = student_id
                            st.session_state["name"] = name
                            st.session_state["course"] = course
                            st.session_state["image_path"] = image_bytes
                            st.session_state["qr_found"] = True
                    else:
                        with qr_lock:
                            st.session_state["qr_error"] = f"Image file not found: {img_path}"
                else:
                    with qr_lock:
                        st.session_state["qr_error"] = "No match found in Excel."

        # HUD
        msg = "Scanning for QR..." if not st.session_state.get("qr_found") else "QR found!"
        cv.putText(img, msg, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def start_qr_scanner_ui():
    st.subheader("QR Code Scanner")

    st.session_state.setdefault("qr_found", False)
    st.session_state.setdefault("qr_error", None)
    st.session_state.setdefault("qr_seen_once", False)  # debouncer

    ctx = webrtc_streamer(
        key="qr-stream",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=QRScanner,
    )

    # 🔁 Poll every 500ms so UI notices updates from the video thread
    st_autorefresh(interval=500, key="qr_watch")

    if st.session_state.get("qr_error"):
        st.error(st.session_state["qr_error"])
        st.session_state["qr_error"] = None

    # When QR is found the processor already filled: student_id, name, course, image_path
    if st.session_state.get("qr_found") and not st.session_state["qr_seen_once"]:
        st.session_state["qr_seen_once"] = True   # prevent repeated toasts
        st.success(f"QR Found: {st.session_state['student_id']} • {st.session_state['name']}")
        # Optional: stop the QR stream to free camera before face verify
        if ctx and ctx.state.playing:
            ctx.stop()
        # Force a one-time rerun to progress immediately
        st.rerun()

# def scan_qr_and_get_student():
#     df = pd.read_excel("studentdb.xlsx")

#     cap = cv.VideoCapture(0)
#     st.info("Scanning for QR code...")

#     st_frame = st.empty()
#     student_id, name, course, image_path = None, None, None, None

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         decoded_objs = decode(frame)
#         for obj in decoded_objs:
#             qr_data = obj.data.decode('utf-8').strip()
#             parts = qr_data.split('|')
#             if len(parts) < 2:
#                 st.warning(f"QR code format invalid: '{qr_data}'")
#                 continue

#             student_id, name = parts[0].strip(), parts[1].strip()
#             match = df[(df['student_id'] == student_id) & (df['name'] == name)]

#             if not match.empty:
#                 course = match.iloc[0]['course']
#                 image_path = match.iloc[0]['image_path']

#                 if not os.path.exists(image_path):
#                     st.error(f"Image file not found: {image_path}")
#                     return None, None, None, None

#                 with open(image_path, "rb") as f:
#                     image_path = f.read()

#                 st.success("Match found in Excel.")
#                 cap.release()
#                 return student_id, name, course, image_path
#             else:
#                 st.error("No match found in Excel.")

#         st_frame.image(frame, channels="BGR", caption="QR Scanner")

#     cap.release()
#     return None, None, None, None


# ========================
# Face Recognition
# ========================
from collections import deque

class FaceVerifier(VideoProcessorBase):
    def __init__(self, proper_name: str, ref_image_bytes: bytes, tolerance: float = 0.50):
        self.name = proper_name
        self.tolerance = tolerance
        self.status = "pending"  # "pending" | "match" | "nomatch" | "noface"
        self.message = ""
        self._dist_hist = deque(maxlen=5)
        self._have_ref = False

        # Precompute reference encoding
        try:
            nparr = np.frombuffer(ref_image_bytes, np.uint8)
            ref_bgr = cv.imdecode(nparr, cv.IMREAD_COLOR)
            ref_rgb = cv.cvtColor(ref_bgr, cv.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(ref_rgb)
            if encs:
                self.ref_encoding = encs[0]
                self._have_ref = True
            else:
                self.status = "noface"
                self.message = "No face found in reference image."
        except Exception as e:
            self.status = "noface"
            self.message = f"Reference image error: {e}"

        # (Optional) YOLO for bigger boxes; `face_recognition` already locates faces though.
        self.yolo = YOLO("yolov8n-face.pt")

    def _ui_conf(self, distance):
        # same as your UI mapping
        return 100.0 / (1.0 + math.exp(8 * (float(distance) - float(self.tolerance))))

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if not self._have_ref:
            cv.putText(img, "No reference face available", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # detect face ROI (YOLO → largest box)
        results = self.yolo(img, verbose=False)
        best = None
        best_area = 0
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].int().tolist()
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area, best = area, (x1, y1, x2, y2)

        if best is None:
            # fallback: let face_recognition find face if YOLO misses
            rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb, model="hog")
            if locs:
                t, r, b, l = max(locs, key=lambda t: (t[2]-t[0])*(t[1]-t[3]))
                best = (l, t, r, b)

        if best is None:
            cv.putText(img, "No face detected. Adjust pose/lighting...", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            # mark transient "noface" only if we never saw a face at all
            if not self._dist_hist:
                self.status = "noface"
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        x1, y1, x2, y2 = best
        face_roi = img[y1:y2, x1:x2]
        rgb_face = cv.cvtColor(face_roi, cv.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb_face)

        color = (0, 0, 255)
        label = "Unknown"

        if encs:
            dist = float(face_recognition.face_distance([self.ref_encoding], encs[0])[0])
            self._dist_hist.append(dist)
            conf = self._ui_conf(dist)
            is_match = dist <= self.tolerance
            label = f"{self.name if is_match else 'Unknown'} | dist={dist:.3f} | conf~{conf:.0f}%"
            color = (34, 139, 34) if is_match else (0, 0, 255)

            # accept only when several consecutive frames are below threshold
            if len(self._dist_hist) == self._dist_hist.maxlen and all(d <= self.tolerance for d in self._dist_hist):
                self.status = "match"
                self.message = f"MATCH ✅ {self.name} | median={np.median(self._dist_hist):.3f}"
            elif len(self._dist_hist) == self._dist_hist.maxlen:
                self.status = "nomatch"
                self.message = f"NOT MATCH ❌ median={np.median(self._dist_hist):.3f}"

        else:
            self.status = "noface"
            label = "No encodings (pose/lighting?)"

        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv.putText(img, label, (x1, max(20, y1 - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def start_face_verify_ui():
    st.subheader("Facial Recognition")

    student_id = st.session_state.get("student_id")
    name = st.session_state.get("name")
    ref_bytes = st.session_state.get("image_path")

    if not student_id or not ref_bytes:
        st.info("Scan a valid QR first.")
        return

    # Allow threshold tweak
    tol = st.slider("Face distance threshold (lower=stricter)", 0.30, 0.80, 0.50, 0.01)

    def factory():
        return FaceVerifier(name, ref_bytes, tolerance=tol)

    ctx = webrtc_streamer(
        key="face-stream",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=factory,
    )

    # Read back status from the processor and trigger your dialogs
    if ctx and ctx.video_processor:
        status = ctx.video_processor.status
        msg = ctx.video_processor.message

        if status == "match":
            st.success(msg or "Face matched.")
            st.session_state["face_verified"] = True
            # show your dialog & announce
            announce_name(name)
            show_success_result(
                st.session_state["student_id"],
                name,
                st.session_state["course"],
                ref_bytes
            )

        elif status == "nomatch":
            st.error(msg or "Face not matched.")
            st.session_state["face_verified"] = False
            show_failed_result()

        elif status == "noface":
            # Only show dialog if we never had encodings yet and user expects a result
            if st.session_state.get("face_verified") is None:
                # st.warning("No face detected yet…")
                # You also have a ‘no face’ dialog:
                show_noDetect_result()

# def face_match_with_qr(proper_name, image_path):
#     TOLERANCE = 0.50

#     def ui_conf(distance, thresh=TOLERANCE):
#         return 100.0 / (1.0 + math.exp(8 * (float(distance) - float(thresh))))

#     yolo_model = YOLO("yolov8n-face.pt")
#     # image_path is bytes from excel so need to convert back to np array
#     nparr = np.frombuffer(image_path, np.uint8)
#     # ref_image is now cv image
#     ref_image = cv.imdecode(nparr, cv.IMREAD_COLOR)
#     # Convert BGR to RGB
#     ref_rgb = cv.cvtColor(ref_image, cv.COLOR_BGR2RGB)
#     # Get face encodings
#     ref_encodings = face_recognition.face_encodings(ref_rgb)
#     if not ref_encodings:
#         st.error("No face found in reference image.")
#         return None

#     ref_encoding = ref_encodings[0]
#     matched = None
#     countdown = 3

#     st.info("Adjust your face... capturing in 3 seconds")
#     st_frame = st.empty()
#     cap = cv.VideoCapture(0)

#     while countdown > 0:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         cv.putText(frame, f"Capturing in {countdown}s", (50, 70),
#                    cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 127), 3)
#         st_frame.image(frame, channels="BGR", caption="Face Recognition")
#         time.sleep(1)
#         countdown -= 1

#     ret, frame = cap.read()
#     if not ret:
#         st.error("No camera detected or face not captured.")
#         cap.release()
#         st.stop()

#     raw_frame = frame.copy()
#     results = yolo_model(frame, verbose=False)
#     largest_area, best_box = 0, None

#     for r in results:
#         for box in r.boxes:
#             x1, y1, x2, y2 = box.xyxy[0].int().tolist()
#             area = (x2 - x1) * (y2 - y1)
#             if area > largest_area:
#                 largest_area, best_box = area, (x1, y1, x2, y2)

#     if best_box:
#         x1, y1, x2, y2 = best_box
#         face_roi = raw_frame[y1:y2, x1:x2]
#         rgb_face = cv.cvtColor(face_roi, cv.COLOR_BGR2RGB)
#         encodings = face_recognition.face_encodings(rgb_face)
#         display_name, label_extra, color = "Unknown", "", (0, 0, 255)

#         if encodings:
#             face_distances = float(face_recognition.face_distance(
#                 [ref_encoding], encodings[0])[0])
#             conf = ui_conf(face_distances, TOLERANCE)
#             is_match = face_distances <= TOLERANCE

#             if is_match:
#                 display_name, color, matched = proper_name, (34, 139, 34), True
#                 st.success(
#                     f"Face matched: {display_name} (dist = {face_distances:.3f} | conf ~ {conf:.0f}%)")
#                 announce_name(display_name)
#                 matched = True
#             else:
#                 st.error(
#                     f"Face does not match reference (dist = {face_distances:.3f} | conf ~ {conf:.0f}%)")
#                 matched = False

#             label_extra = f" | dist = {face_distances:.3f} | conf ~ {conf:.0f}%"
#         else:
#             st.warning(
#                 "No face encoding from camera frame. Try better lighting / frontal pose.")
#             matched = None

#         cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         cv.putText(frame, f"{display_name}{label_extra}", (x1, max(20, y1 - 10)),
#                    cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
#         st_frame.image(frame, channels="BGR", caption="Face Recognition")
#     else:
#         st.warning("No face detected. Please move closer and face the camera.")
#         st_frame.image(frame, channels="BGR", caption="Face Recognition")
#         matched = None

#     cap.release()
#     return matched

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

    st.markdown(
        """
        <style> div.stButton > button { display: block; margin: 0 auto; border: 1px solid #57ffe0 !important; color: #57ffe0 !important; } div.stButton > button:hover { border: 1px solid #fae100 !important; color: #fae100 !important; } </style>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # Session defaults
    st.session_state.setdefault("scanning_started", False)
    st.session_state.setdefault("student_id", None)
    st.session_state.setdefault("name", None)
    st.session_state.setdefault("course", None)
    st.session_state.setdefault("image_path", None)
    st.session_state.setdefault("face_verified", None)

    with col1:
        header_col, btn_col = st.columns([3, 1])
        with header_col:
            st.subheader("QR Code Scanner")

        def start_scanning():
            st.session_state["scanning_started"] = True
            st.session_state["qr_found"] = False
            st.session_state["qr_error"] = None
            st.session_state["student_id"] = None
            st.session_state["name"] = None
            st.session_state["course"] = None
            st.session_state["image_path"] = None
            st.session_state["face_verified"] = None

        def stop_scanning():
            st.session_state["scanning_started"] = False

        with btn_col:
            if st.session_state["scanning_started"]:
                st.button("Stop Recording", key="stop-btn", on_click=stop_scanning)
            else:
                st.button("Start Recording", key="start-btn", on_click=start_scanning)

        if st.session_state["scanning_started"] and not st.session_state.get("student_id"):
            start_qr_scanner_ui()
        elif st.session_state.get("student_id"):
            st.success(f"Ready for face verify: {st.session_state['student_id']} • {st.session_state['name']}")

    with col2:
        st.subheader("Facial Recognition")
        if st.session_state.get("student_id"):
            start_face_verify_ui()
        else:
            st.info("Please scan a QR first.")

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
        div.stButton {
            display: flex;
            justify-content: center;
        }
        div.stButton > button {
            border: 1px solid #57ffe0 !important; 
            color: #57ffe0 !important;
        }
        div.stButton > button:hover {
            border: 1px solid #fae100 !important; 
            color: #fae100 !important;
        }

        div[data-testid="stSpinner"] > div { 
            display: flex; 
            justify-content: center; 
        }
        div[data-testid="stSpinner"] > div > div { 
            text-align: center; 
        }
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
                div.stButton {
                    display: block;
                    width: 100% !important;
                }
                div.stButton > button {
                    background-color: #57ffe0 !important;
                    color: black !important;
                    border: none;
                    border-radius: 5px;
                    padding: 5px 5px;
                    font-weight: bold;
                    width: 100% !important;
                }
                div.stButton > button:hover {
                    background-color: #fae100 !important;
                    color: black !important;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("No database found.")
