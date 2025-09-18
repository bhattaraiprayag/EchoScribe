### **Configuration & Setup**
This guide provides instructions for setting up the project on different operating systems. Follow the steps that apply to your device.

### **Prerequisites**
You'll need a Python environment. It is recommended to use Anaconda to manage your dependencies. If you don't have it, you can download it from the official site.

1. **First create a new Python environment**:
```bash
conda create --name myenv python=3.10
conda activate myenv
```

## **Installation**
#### **For Devices with NVIDIA GPUs (non-Mac)**
1. **Install CUDA Toolkit**:
```bash
conda install nvidia/label/cuda-13.0.1::cuda-toolkit
```
2. **Install PyTorch with CUDA support**:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### **For Mac M-series Devices**
1. **Install PyTorch**:
```bash
pip3 install torch torchvision torchaudio
```

### **Common Dependencies**
After installing the required PyTorch version for your device, install the remaining dependencies:
```bash
pip install fastapi websockets uvicorn silero-vad numpy faster-whisper ffmpeg-python --no-cache-dir
```

## **Running the Application**
1. **Navigate to the backend directory**:
```bash
cd backend
```
2. **Start the server**:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
3. **Access the web application in your browser**:
```bash
http://localhost:8000/
```