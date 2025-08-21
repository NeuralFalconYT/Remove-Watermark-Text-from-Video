

# Remove-Watermark-Text-from-Video  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeuralFalconYT/video-watermark-remover/blob/main/easyocr_watermark_remove.ipynb) <br>
[![HuggingFace Space Demo](https://img.shields.io/badge/🤗-Space%20demo-yellow)](https://huggingface.co/spaces/NeuralFalcon/Remove-Watermark-Text-from-Video)

🎥 **Goal**: This project focuses on removing (blurring) text watermarks from videos.  
We use **EasyOCR** to detect watermark text regions (supports **English** and **Chinese**), then apply **OpenCV blurring techniques** to mask the detected text. Finally, we use **FFmpeg** to restore the original audio into the processed video.  

---

## ✨ Features  
- Detects watermark text in videos using **EasyOCR**  
- Supports **English** and **Chinese** text detection  
- Removes watermarks using different **blur techniques**  
- Restores the original **audio track** with FFmpeg  
- Provides a **web interface** using **Gradio**  

---


## 📊 Examples  

### Chinese Example  
| Original  | Watermark Removed |
|----------------|-------------------|
| <video src="https://github.com/user-attachments/assets/52f93f68-d56a-45d4-a113-eb75d3686ebd" width="360" controls></video> | <video src="https://github.com/user-attachments/assets/b8cb8075-90ee-41b3-9d64-db9b151555bc" width="360" controls></video> |

### English Example  
| Original  | Watermark Removed |
|----------------|-------------------|
| <video src="https://github.com/user-attachments/assets/95fadaf0-7ffc-43e7-b0a9-59844f54ae71" width="360" controls></video> | <video src="https://github.com/user-attachments/assets/39ce62b8-4c3c-4b66-8afd-cdfccfb9c079" width="360" controls></video> |




---

## 🖥️ App Interface  
![App Screenshot](https://github.com/NeuralFalconYT/Remove-Watermark-Text-from-Video/blob/main/examples/app.jpg)

---

## 🚀 Installation  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/NeuralFalconYT/Remove-Watermark-Text-from-Video.git
cd Remove-Watermark-Text-from-Video
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Install FFmpeg

* **Linux (Debian/Ubuntu)**

```bash
sudo apt update && sudo apt install -y ffmpeg
```

* **Windows / Mac**
  Download from [FFmpeg.org](https://ffmpeg.org/download.html) and add it to your system PATH.

---

## ▶️ Run the App

```bash
python app.py
```

---

## ⚙️ How It Works

1. **OCR Detection** → EasyOCR detects text regions (English & Chinese).
2. **Blurring** → OpenCV applies Gaussian/Median blur on detected text areas.
3. **Audio Recovery** → FFmpeg merges the processed video with original audio.
4. **Output** → Final watermark-free video is generated.




## 🔮 Notes on OCR Alternatives

While this project currently uses **EasyOCR**, there are many **newer OCR models** available today that may provide:

* **Better accuracy** in detecting text (especially complex fonts & styles)
* **Faster processing speeds** on both CPU and GPU


👉 If you want to experiment, you can **replace EasyOCR** with advanced OCR solutions (You can find latest ocr models on huggingface).

---

