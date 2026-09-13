@echo off
cd /d c:\Users\anasw\Downloads\ai_vision_navigator
echo Training started: %DATE% %TIME% > training_log.txt
venv\Scripts\python.exe dataset\scripts\train_custom_model.py --epochs 15 --imgsz 416 --batch 8 --device cpu --auto-copy >> training_log.txt 2>&1
echo Training ended: %DATE% %TIME% >> training_log.txt
