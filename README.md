# ============================
# Streamlit Cloud Env 
# ============================

1. At first, ensure that the local devices has been install all the required engine/library
    # ========================
    # Importing Commands
    # ========================
    pip install -r requirements.txt
    pip install pyttsx3
    pip install qrcode
    pip install opencv-python
    pip install pandas
    pip install numpy
    pip install face-recognition
    pip install ultralytics
    pip install pyzbar
    pip install streamlit
    pip install streamlit-option-menu
    pip install streamlit-aggrid
    pip install gTTS

3. Before using the verification system, you need to key in required information into the Excel file
   * Remarks:
      - Image need to place in "student_image" folder
      - Naming put as id number, example like "24WMT01010.jpg"
    
4. Access to this URL --> XXX

* Remarks for System Usage: Since operations work under session_state, so there may be some minor issue when system proceeding
  - Example:
    - When the camera is open, navigate to Admin View
      - The output: When get back to the Dashboaord, the UI will be dimmed and not being clear yet,
      - In this case, you are required refresh the page
      # Click "Stop Scanning" button first to close the camera, then only proceed to the other section page  
  
# ==============================================================
# System Usage (Dashboard - Scan and Verification)
# ==============================================================

This section is allow the student to scan their QR Code and Facial Recognition to verify their identity.

- Click the "Start Scanning" button

- Show the QR Code to the camera

  After the QR Code is successfully matched, it will automatically proceed to the Facial Recognition section

  When face successfully scanned, the system will show the verification result and announce the student name

  Then, it will prompt back to QR Code Scan camera automatically.

* Remarks:
    - The process of scanning the face is about 3 seconds, be ready after QR successfully verified
    - The entire workflow of (QR Code Scan --> Match info with Excel --> Face Scan --> Show Result) will be continuosly proceed
    - If wish to stop the procedure, click the "Stop Scanning" button 

# ==============================================================
# System Usage (QR Generator)
# ==============================================================

This section is allow the administrators to generate the QR Codes and send to all students that exists in the Excel file.

- Click the "Generate QR" button (And the QR will be sent to all of the students)

# ==============================================================
# System Usage (Admin View)
# ==============================================================

This section is allow the administrators to check the student data, and handle urgent cases like sending QR code to the specific student.

- Click the "Send QR" button (And the QR will be sent to the specific student)


