#!/bin/bash
# EdgeHand surgical video node on the Jetson (loops the clip; phase events -> laptop 192.168.55.100:5007)
pkill -f "[e]dgehand.surgical_feed" 2>/dev/null
cd ~/edgehand-src
exec ~/edgehand-venv/bin/python -u -m edgehand.surgical_feed ~/edgehand-demo/clippingdemo_540p.mp4 \
    ~/edgehand-demo/clippingdemo.csv --laptop 192.168.55.100 --loop "$@" > ~/feed.log 2>&1
