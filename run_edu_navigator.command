#!/bin/bash
cd "$(dirname "$0")"
echo "Edu-Navigator 실행을 준비 중입니다..."
python3 -m pip install -r requirements.txt
echo "서버를 시작합니다. 잠시만 기다려 주세요..."
(sleep 2; open "http://localhost:5001") &
python3 app.py
