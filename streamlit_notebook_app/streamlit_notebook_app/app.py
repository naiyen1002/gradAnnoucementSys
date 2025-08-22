import streamlit as st
import sys, io, contextlib

st.set_page_config(page_title="Notebook → Streamlit", layout="wide")
st.title("Notebook → Streamlit Runner")
st.caption("This Streamlit app was auto-generated from a Jupyter notebook. "
           "Use the sidebar to run all cells in order and view outputs.")

with st.sidebar:
    st.header("Controls")
    run_all = st.button("▶️ Run all cells")
    show_code = st.checkbox("Show code", value=True)
    reset_ns = st.checkbox("Reset state before running", value=True)
    st.markdown("---")
    st.markdown("**Tip:** If your code reads local files, upload them below and use their saved path.")

# File uploader area for data files the notebook may expect
st.subheader("Optional: Upload data files")
uploads = st.file_uploader("Upload one or more files used by your notebook code (CSV, images, etc.)", accept_multiple_files=True)
saved_paths = []
if uploads:
    import tempfile, os
    data_dir = tempfile.mkdtemp(prefix="uploaded_data_")
    for f in uploads:
        outp = os.path.join(data_dir, f.name)
        with open(outp, "wb") as w:
            w.write(f.getbuffer())
        saved_paths.append(outp)
    with st.expander("Saved file paths"):
        for p in saved_paths:
            st.write(p)

# Patch matplotlib to render figures in Streamlit
try:
    import matplotlib.pyplot as plt
    def _st_show(*args, **kwargs):
        st.pyplot(plt.gcf())
    plt.show = _st_show
except Exception:
    pass

# Execution namespace (persists unless reset)
if "nb_globals" not in st.session_state or reset_ns:
    st.session_state.nb_globals = {"__name__": "__main__"}
ns = st.session_state.nb_globals

st.markdown("### Cell 1")
code_1 = r"""
pip install tf-keras
"""
if show_code:
    st.code(code_1, language="python")
if run_all:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            exec(code_1, ns)
            st.success("Cell 1 executed.")
        except Exception as e:
            st.error("Error in Cell 1:")
            st.exception(e)
    if stdout.getvalue().strip():
        st.subheader("Stdout (Cell 1)")
        st.text(stdout.getvalue())
    if stderr.getvalue().strip():
        st.subheader("Stderr (Cell 1)")
        st.text(stderr.getvalue())


st.markdown("### Cell 2")
code_2 = r"""
pip install deepface
"""
if show_code:
    st.code(code_2, language="python")
if run_all:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            exec(code_2, ns)
            st.success("Cell 2 executed.")
        except Exception as e:
            st.error("Error in Cell 2:")
            st.exception(e)
    if stdout.getvalue().strip():
        st.subheader("Stdout (Cell 2)")
        st.text(stdout.getvalue())
    if stderr.getvalue().strip():
        st.subheader("Stderr (Cell 2)")
        st.text(stderr.getvalue())


st.markdown("### Cell 3")
code_3 = r"""
pip install pyzbar
pip install "qrcode[pil]"
"""
if show_code:
    st.code(code_3, language="python")
if run_all:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            exec(code_3, ns)
            st.success("Cell 3 executed.")
        except Exception as e:
            st.error("Error in Cell 3:")
            st.exception(e)
    if stdout.getvalue().strip():
        st.subheader("Stdout (Cell 3)")
        st.text(stdout.getvalue())
    if stderr.getvalue().strip():
        st.subheader("Stderr (Cell 3)")
        st.text(stderr.getvalue())


st.markdown("### Cell 4")
code_4 = r"""
# import qrcode
# import cv2 as cv
# from pyzbar.pyzbar import decode


# def generate_qr_code(data, filename="qr_code.png", box_size=10, border=4):

#     qr = qrcode.QRCode(
#         version=1, 
#         error_correction=qrcode.constants.ERROR_CORRECT_H, 
#         box_size=box_size,
#         border=border,
#     )
    
#     # Add data to QR code
#     qr.add_data(data)
#     qr.make(fit=True)
    
#     # Create image
#     img = qr.make_image(fill_color="black", back_color="white")
    
#     # Save image
#     img.save(filename)
#     print(f"QR Code saved as {filename}")

# # Example usage:
# name = input("Enter student name: ")
# student_id = input("Enter student ID: ")
# face_image_path = input("Enter path to face image (or URL): ")

# # Create data dictionary
# student_data = {
#     "name": name,
#     "id": student_id,
#     "face_image": face_image_path
# }

# # Generate QR
# filename = f"{student_id}_qr.png"
# generate_qr_code(student_data, filename)
"""
if show_code:
    st.code(code_4, language="python")
if run_all:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            exec(code_4, ns)
            st.success("Cell 4 executed.")
        except Exception as e:
            st.error("Error in Cell 4:")
            st.exception(e)
    if stdout.getvalue().strip():
        st.subheader("Stdout (Cell 4)")
        st.text(stdout.getvalue())
    if stderr.getvalue().strip():
        st.subheader("Stderr (Cell 4)")
        st.text(stderr.getvalue())


st.markdown("### Cell 5")
code_5 = r"""
#generate QR
import pandas as pd
import qrcode
import os
import smtplib
from email.message import EmailMessage

##SMTP Config
sender_email = "jhlee-wm22@student.tarc.edu.my"
sender_pass = "moseepjnzqxsdttn"
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
        print(f"Warning----------- \n Skipping {name} (ID: {student_id}) — invalid or missing email")
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

    #Email body
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
        msg.add_attachment(file_data, maintype="image", subtype="png", filename=file_name)

    # Send email
    server.send_message(msg)
    print(f"Sent QR to {name} ({email})")

server.quit()
print("QR codes generated for all students.")
"""
if show_code:
    st.code(code_5, language="python")
if run_all:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            exec(code_5, ns)
            st.success("Cell 5 executed.")
        except Exception as e:
            st.error("Error in Cell 5:")
            st.exception(e)
    if stdout.getvalue().strip():
        st.subheader("Stdout (Cell 5)")
        st.text(stdout.getvalue())
    if stderr.getvalue().strip():
        st.subheader("Stderr (Cell 5)")
        st.text(stderr.getvalue())


st.markdown("### Cell 6")
code_6 = r"""
##run thisssssssssss
import cv2 as cv
import pandas as pd
import face_recognition
from ultralytics import YOLO
from pyzbar.pyzbar import decode

# QR Code Scan Function

def scan_qr_and_get_student():
    df = pd.read_excel("studentdb.xlsx")

    cap = cv.VideoCapture(0)
    print(" Scanning for QR code... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        decoded_objs = decode(frame)
        for obj in decoded_objs:
            qr_data = obj.data.decode('utf-8')
            student_id, name = qr_data.split('|')

            print(f"QR Detected: ID={student_id}, Name={name}")

            match = df[(df['student_id'] == student_id) & (df['name'] == name)]
            if not match.empty:
                image_path = match.iloc[0]['face_image_path']
                print(f"Match found in Excel. Face image path: {image_path}")
                cap.release()
                cv.destroyAllWindows()
                return student_id, name, image_path
            else:
                print("No match found in Excel.")

        cv.imshow("QR Scanner", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv.destroyAllWindows()
            return None, None, None

    cap.release()
    cv.destroyAllWindows()
    return None, None, None

# Face Recognition Function

def face_match_with_qr(student_id, name, reference_image_path):
    model = YOLO("yolov8n.pt")

    # Load reference encoding
    ref_image = face_recognition.load_image_file(reference_image_path)
    ref_encoding = face_recognition.face_encodings(ref_image)[0]

    cap = cv.VideoCapture(0)
    print("🎥 Face recognition started... Press 'q' to quit.")

    matched = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                face_roi = frame[y1:y2, x1:x2]

                try:
                    rgb_face = cv.cvtColor(face_roi, cv.COLOR_BGR2RGB)
                    enc = face_recognition.face_encodings(rgb_face)

                    if len(enc) > 0:
                        face_distances = face_recognition.face_distance([ref_encoding], enc[0])
                        conf = (1 - face_distances[0]) * 100 
                    
                        if conf >= 70: 
                            text = f"{name} ({student_id})  {conf:.2f}%"
                            color = (0, 255, 0)
                            matched = True
                            print(f"✅ Face match! ID={student_id}, Name={name}, Confidence={conf:.2f}%")
                        else:
                            text = f"❌ Unmatched ({conf:.2f}%)"
                            color = (0, 0, 255)
                            print(f"❌ Face does not match QR code! Confidence={conf:.2f}%")
                    
                        # Draw bounding box + label
                        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv.putText(frame, text, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                except Exception as e:
                    print("Error:", e)

        cv.imshow("Face Recognition", frame)

        if matched:  
            # Stop face recognition and go back to QR scanner
            cap.release()
            cv.destroyAllWindows()
            return True  

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()
    return False

# MAIN 
while True:
    student_id, name, image_path = scan_qr_and_get_student()
    if student_id is None:  
        break  
    
    result = face_match_with_qr(student_id, name, image_path)
    if result:
        print("Returning to QR scan for next student...")
        continue
"""
if show_code:
    st.code(code_6, language="python")
if run_all:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            exec(code_6, ns)
            st.success("Cell 6 executed.")
        except Exception as e:
            st.error("Error in Cell 6:")
            st.exception(e)
    if stdout.getvalue().strip():
        st.subheader("Stdout (Cell 6)")
        st.text(stdout.getvalue())
    if stderr.getvalue().strip():
        st.subheader("Stderr (Cell 6)")
        st.text(stderr.getvalue())

st.divider()
st.caption("Auto-generated • Edit app.py to customize UI and interactions.")
