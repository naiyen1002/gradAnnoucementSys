import pyttsx3

def list_all_voices():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')

    for i, voice in enumerate(voices):
        print("="*40)
        print(f"Index: {i}")
        print(f"ID: {voice.id}")
        print(f"Name: {voice.name}")
        print(f"Languages: {voice.languages}")
        print(f"Gender: {voice.gender}")
        print(f"Age: {voice.age}")

list_all_voices()

# ----------------------------------------------------------------------

def face_match_with_qr(student_id, name, reference_image_path):
    model = YOLO("yolov8n.pt")

    # Load reference encoding
    ref_image = face_recognition.load_image_file(reference_image_path)
    ref_encoding = face_recognition.face_encodings(ref_image)[0]

    cap = cv.VideoCapture(0)
    st.info("Adjust your face... capturing in 3 seconds")
    st_frame = st.empty()

    countdown = 3
    start_time = time.time()
    while countdown > 0:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the 54321
        text = f"Capturing in {countdown}s"
        cv.putText(frame, text, (50, 70), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 2)

        st_frame.image(frame, channels="BGR", caption="Face Recognition")

        # Wait a bit
        time.sleep(1)
        countdown -= 1

    # Capture
    ret, frame = cap.read()
    if not ret:
        cap.release()
        text = f"No face detected"
        return False

    results = model(frame, conf=0.5)
    matched = False

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
                        text = f"{conf:.2f}%"
                        color = (34, 139, 34)
                        matched = True
                        st.success(f"Face matched! Confidence = {conf:.2f}")
                        # ID={student_id}, Name={name}, Confidence={conf:.2f}%

                        # Draw bounding box + label
                        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        # cv.putText(frame, text, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        # cv.putText(frame, f"Course: RSW", (x1, y1 - 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        # cv.putText(frame, f"Name: {name}", (x1, y1 - 70), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                        announce_name(name)
                        cap.release()
                        return True 
                    else:
                        text = f"Unmatched ({conf:.2f}%)"
                        color = (0, 0, 255)
                        st.warning(f"Face does not match QR code! Confidence = {conf:.2f}%")

                        # Draw only the red rectangle + unmatched label
                        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        # cv.putText(frame, text, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
 
            except Exception as e:
                st.error("Error:", e)

    st_frame.image(frame, channels="BGR", caption="Face Recognition")  

    cap.release()
    # cv.destroyAllWindows()
    return matched