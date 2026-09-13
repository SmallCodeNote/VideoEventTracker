# Video Event Tracker & OCR Tool

A desktop application that monitors video files, detects specific events based on brightness thresholds within specified Regions of Interest (ROI), and performs Optical Character Recognition (OCR).

This tool is designed for scenarios such as surveillance system monitoring, automated content analysis, or extracting text from specific areas in a video stream when certain visual conditions are met.

## ✨ Features

- **Multi-threaded Processing:** Seamlessly handles video decoding and OCR processing without freezing the GUI.
- **Dynamic ROI Management:** Define up to 5 independent regions, each with custom brightness thresholds and cooldown timers.
- **Interactive UI:** Drag and drop ROIs or OCR boxes directly on the video preview for precise placement.
- **Real-time Preview:** Displays active ROIs and current OCR detection areas using visual overlays.
- **Customizable OCR:** Fine-tune OCR accuracy using settings for padding, grayscale conversion, and thresholding.
- **Automatic Logging:** Automatically exports detected text to a customizable log file along with timestamps, frame indices, and video paths.
- **Configuration Persistence:** Save and load workspace settings (paths, ROI coordinates, etc.) via JSON files.

## 🛠 Prerequisites (Windows 11)

Ensure the following are installed before running this tool:

1.  **Python 3.12+**
    - Download from [python.org](https://www.python.org/downloads/windows/).

2.  **Tesseract OCR Engine**
    - Download the Windows installer from the [Tesseract page](https://github.com/tesseract-ocr/tesseract).
    - Install it (the default path for Windows is typically `C:\Program Files\Tesseract-OCR`).
    - You will need to specify the path to `tesseract.exe` within the application settings.

## 🚀 Installation

1.  **Clone the repository or download the files:**
    ```powershell
    git clone https://github.com/your_username/video-event-tracker.git
    cd video-event-tracker
    ```

2.  **Create a virtual environment (Recommended):**
    Open PowerShell in the project folder and run:
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```powershell
    pip install PySide6 opencv-python numpy pytesseract
    ```

## 📖 How to Use

1.  **Launch the Application:**
    Run the script using Python:
    ```powershell
    python VideoEventTracker.py
    ```

2.  **Configure Paths:**
    - Select the **Parent Folder** containing your `.mp4` files.
    - Select the **Log Output File Path** where results will be saved (e.g., `results.txt`).

3.  **Set Regions of Interest (ROI):**
    - Use the "Settings" tab to define ROIs.
    - Drag these boxes over the preview window to align them with specific objects or areas of interest.
    - Adjust **Thresholds** and **Cooldowns (ms)** to filter out noise or prevent duplicate log entries.

4.  **OCR Settings:**
    - Configure OCR boxes individually. Text processing will occur in these areas whenever a detection event is triggered in an active ROI.
    - If `tesseract.exe` is not in your system's default path, enter the correct path in the settings.

5.  **Start Processing:**
    Click **"Start Process."** The application will sequentially process all `.mp4` files in the folder, perform analysis, and update logs in real-time.

## ⚙️ Technical Details

- **GUI Framework:** PySide6 (Qt for Python)
- **Computer Vision:** OpenCV (`cv2`)
- **OCR Engine:** Tesseract OCR via `pytesseract`
- **Concurrency:** Uses `QThread` to ensure the video processing loop does not block the main UI thread.
- **Image Processing Pipeline:** 
    1. Frame Capture $\rightarrow$ Grayscale Conversion.
    2. ROI Brightness Analysis (Average Pixel Value).
    3. If Threshold is exceeded $\rightarrow$ Region Cropping $\rightarrow$ Apply Padding/Grayscale/Thresholding $\rightarrow$ Tesseract OCR.

## 📄 License
This project is licensed under [The Unlicense](https://unlicense.org/).
