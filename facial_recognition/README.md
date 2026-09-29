# Facial Recognition Hardware Test

> **Note:** These instructions assume you have a USB camera for your Raspberry Pi. If you don't, just drop these instructions and the scripts into ChatGPT or whatever AI you use and have it adapt the camera stuff for you. Or just don't use these instructions.

This is basically just a quick setup so we can see whether YuNet + SFace will actually run well on the Raspberry Pis we have before worrying about the actual facial recognition system.

The main things being tested are:

- camera compatibility
- live YuNet face detection
- SFace embeddings
- Pi stability
- temperature / CPU / RAM
- undervoltage or throttling
- random USB camera disconnects

## Setup

First install the basic stuff:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip wget
```

Go into the facial recognition folder:

```bash
cd facial_recognition
```

Make a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
pip install --upgrade pip
```

Install the Python packages:

```bash
pip install -r requirements.txt
```

## Download YuNet and SFace

Download YuNet:

```bash
wget -O yunet.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx
```

Download SFace:

```bash
wget -O sface.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

Make sure both are there:

```bash
ls -lh *.onnx
```

You should see:

```text
yunet.onnx
sface.onnx
```

## Test Your USB Camera

Plug your USB camera into the Pi.

Check if Linux sees it:

```bash
ls /dev/video*
```

It will probably show up as:

```text
/dev/video0
```

Then test whether OpenCV can actually read from it:

```bash
python - <<'PY'
import cv2

cap = cv2.VideoCapture(0)

print("Opened:", cap.isOpened())

ret, frame = cap.read()

print("Read frame:", ret)

if ret:
    print("Frame shape:", frame.shape)

cap.release()
PY
```

If it's working, you should get something like:

```text
Opened: True
Read frame: True
Frame shape: (480, 640, 3)
```

## Test Live Face Detection

Run:

```bash
python live_detect.py
```

You should get a live camera window with boxes around faces.

Press `q` to quit.

YuNet is the lightweight neural network doing the actual face detection.

## How the Recognition Test Works

YuNet finds the face and facial landmarks.

SFace then turns the aligned face into a 128-dimensional embedding vector.

That vector gets compared to a reference face using cosine similarity.

Higher similarity means the faces are more likely to be the same person.

## Add a Reference Face

The current script expects a reference image called:

```text
sammy1.jpeg
```

That filename is just what I used while testing.

You can either:

- rename your reference photo to `sammy1.jpeg`
- or open `live_recognize.py` and change:

```python
REFERENCE_IMAGE = "sammy1.jpeg"
```

to whatever your image is actually called.

Use a clear photo with:

- one visible face
- decent lighting
- not super blurry
- not heavily obstructed
- preferably mostly facing the camera

## Run Live Face Recognition

Run:

```bash
python live_recognize.py
```

The script will:

1. load the reference image
2. find the face
3. create the SFace embedding
4. open the camera
5. detect live faces
6. create embeddings for those faces
7. compare them using cosine similarity

If it thinks the face matches, you'll get something like:

```text
SAMMY 0.67
```

If it doesn't:

```text
UNKNOWN 0.14
```

The current threshold is around:

```text
0.36
```

Press `q` to quit.

The reference image only gets embedded once when the program starts. The embedding stays in RAM while the program is running. Right now it is not permanently saved anywhere.

## Basic Image Comparison Test

You can also run:

```bash
python compare_faces.py
```

That script is just for checking whether two photos of the same person get a noticeably higher cosine similarity than two different people.

My first test got roughly:

```text
Same person:       0.67
Different person:  0.14
```

Don't treat those exact numbers like gospel. Lighting, angles, camera quality, distance, and image quality will all affect it.

## Pi Health Logger

I also made a basic Pi health logger because my Pi was being weird as hell and I wanted to know whether it was actually the models or just the hardware.

Open another terminal, activate the venv:

```bash
source .venv/bin/activate
```

Then run:

```bash
python pi_health_logger.py
```

It logs:

- temperature
- CPU usage
- CPU frequency
- RAM usage
- undervoltage
- throttling
- whether `/dev/video0` is still there

It saves everything to:

```text
pi_health_log.csv
```

Stop it with:

```text
Ctrl+C
```

## Write Down Your Hardware

If you test this, it would help to write down what you're actually using so we can compare results later.

```text
Your name:
Raspberry Pi model:
RAM:
Operating system:
Camera:
Power supply:
Power supply wattage:
Cooling:
USB hub:
How long you tested:
Any camera disconnects:
Any undervoltage warnings:
Anything else weird:
```

The power supply part actually matters.

My Pi was randomly dropping the USB camera and acting like the code was broken, but the logs showed undervoltage events.

## If Your Pi Starts Acting Stupid

Run:

```bash
vcgencmd get_throttled
```

and:

```bash
dmesg | tail -n 50
```

If you see stuff like:

```text
Undervoltage detected!
```

or:

```text
USB disconnect
```

then your hardware/power setup might be the problem, not the models.

Jacob said he'd test this on his Pi, so that's probably the main machine we're comparing against. But if anyone else wants to try it and thinks their setup is better, go for it.

Mine definitely is not the machine we should judge this off of.

If anything breaks, crashes, disconnects, throttles, or just acts weird, lmk what happened and what hardware you were using.

## Files

```text
facial_recognition/
??? README.md
??? requirements.txt
??? live_detect.py
??? live_recognize.py
??? compare_faces.py
??? pi_health_logger.py
??? test_assets/
    ??? test_face.jpg
```

The `.onnx` model files are not stored in the repo. You download them with the commands above.

## Current Status

This is not the final facial recognition system.

Right now this is just to see whether the setup runs reliably on someone else's Pi.

If that works, then we can worry about the actual enrollment system, saving embeddings, handling multiple people, and integrating it with the drone.
