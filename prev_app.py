import os

import time
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
import time
from scipy.signal import sawtooth
from time import sleep

import matplotlib.pyplot as plt
from time import sleep

import cv2
from tensorflow import device
import torch
from numpy import random

from norfair.drawing.draw_boxes import draw_boxes
from retinaface import RetinaFace

from norfair.drawing import Paths
from norfair import Detection, Tracker, draw_tracked_boxes, Paths, draw_boxes

from utils import convert_bboxes_to_absolute, get_norfair_detections, centroid_distance, UI_box, draw_border

import tempfile
import streamlit as st


# Function to generate an ECG-like signal
def generate_ecg_waveform(duration, sampling_rate, heart_rate):
    t = np.linspace(0, duration, int(sampling_rate * duration))
    # Create a synthetic ECG signal: this is a simplified version
    # Real ECG signals are more complex and vary among individuals
    ecg = 0.5 * np.sin(2 * np.pi * heart_rate * t / 60)  # Simplified heart rhythm
    ecg += 0.5 * sawtooth(2 * np.pi * heart_rate * t / 60, 0.5)  # Add sawtooth for QRS complex
    return t, ecg

def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame


realWidth = 320
realHeight = 240
videoWidth = int(realWidth*.75)
videoHeight = int(realHeight/2)
w_pad = int((realWidth - videoWidth)/2)
h_pad = int((realHeight - videoHeight)/2)

videoChannels = 3
videoFrameRate = 10

# Color Magnification Parameters

levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0

#Output Display Parameters

font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (20, 30)
bpmTextLocation = (videoWidth//2 + 5, 30)
fontScale = 1
fontColor = (0,255,0)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 3

# Initialize Gaussian Pyramid
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels+1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))

# Bandpass Filter for Specified Frequencies

frequencies = (1.0*videoFrameRate) * np.arange(bufferSize) / (1.0*bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

# Heart Rate Calculation Variables

bpmCalculationFrequency = 15
bpmBufferIndex = 0
bpmBufferSize = 10
bpmBuffer = np.zeros((bpmBufferSize))

ind = 0
################################
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
wh_deque   = {}



def draw_boxes(img, bbox, object_id, identities=None, offset=(0, 0)):

    height, width, _ = img.shape
    bpm = 60
    # print("height, width", height, width)
    calilbration_constant = .01
    # remove tracked point from buffer if object is lost
    if identities is not None:
      for key in list(data_deque):
        if key not in identities:
          data_deque.pop(key)
      for key in list(wh_deque):
        if key not in identities:
          wh_deque.pop(key)

    for i, box in enumerate(bbox):
        # print(box)
        x1, y1, x2, y2 = [i for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        box_height = (y2-y1)

        # code to find center of bottom edge
        center = (int((x2+x1)/ 2), int((y2+y2)/2))
        # code to find wh
        wh = ((x2-x1), (y2-y1))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:
          data_deque[id] = deque(maxlen = 64)
        if id not in wh_deque:
          wh_deque[id] = deque(maxlen = 64)

      
        # add center to buffer
        data_deque[id].appendleft(center)
        # add wh to buffer
        wh_deque[id].appendleft(wh)

#Crop#Image####################################
        xc = int((x2+x1)/2)
        yc = int((y2+y1)/2)
        w_mean = 1
        h_mean = 1
        for i in range(len(wh_deque[id])):
            w_mean = w_mean + wh_deque[id][i][0]
            h_mean = h_mean + wh_deque[id][i][1]
        w_mean = int(w_mean/len(wh_deque[id]))
        h_mean = int(h_mean/len(wh_deque[id]))
        xmin = max(0, int(xc - w_mean/2))
        xmax = min(width, int(xc + w_mean/2))
        ymin = max(0, int(yc - h_mean/2))
        ymax = min(height, int(yc + h_mean/2))

        crop_img = img[ymin:ymax, xmin:xmax]

        dim = (realWidth, realHeight)

        crop_img = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)
#Heart#Rate#Detection################################
        global bufferIndex, ind, bpmBufferIndex

        frame = crop_img.copy()
        detectionFrame = frame[h_pad:realHeight-h_pad, w_pad:realWidth-w_pad, :]
        # print(detectionFrame.shape)

        # Construct Gaussian Pyramid
        videoGauss[bufferIndex] = buildGauss(detectionFrame, levels+1)[levels]
        fourierTransform = np.fft.fft(videoGauss, axis=0)

        # Bandpass Filter
        fourierTransform[mask == False] = 0

        # Grab a Pulse
        if bufferIndex % bpmCalculationFrequency == 0:
            ind = ind + 1
            for buf in range(bufferSize):
                fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            hz = frequencies[np.argmax(fourierTransformAvg)]
            bpm = 60.0 * hz
            bpmBuffer[bpmBufferIndex] = bpm
            bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

        # Amplify
        filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
        filtered = filtered * alpha

        # Reconstruct Resulting Frame
        filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
        outputFrame = detectionFrame + filteredFrame
        outputFrame = cv2.convertScaleAbs(outputFrame)

        bufferIndex = (bufferIndex + 1) % bufferSize
        heartrate = 0.0
        bpm_text = "Calculating.."
        
        frame[h_pad:realHeight-h_pad, w_pad:realWidth-w_pad, :] = outputFrame
        cv2.rectangle(frame, (w_pad , h_pad), (realWidth-w_pad, realHeight-h_pad), boxColor, 3)

        if ind > bpmBufferSize:
            heartrate = int(bpmBuffer.mean())/100
            bpm_text =  str(int(bpmBuffer.mean())) 
            # cv2.putText(frame, "BPM: %d" % bpmBuffer.mean(), bpmTextLocation, font, fontScale, fontColor, lineType)
        # else:
        #     cv2.putText(frame, "Calculating...", loadingTextLocation, font, fontScale, fontColor, lineType)

        crop_img = frame.copy()

#Show#Image#################################
        x_offset = width-320
        y_offset = 5 +230*list(wh_deque.keys()).index(id)
        x_end = x_offset + crop_img.shape[1]
        y_end = y_offset + crop_img.shape[0]
        # print(y_offset, y_end, x_offset, x_end)
        # print(crop_img.shape)
        img[y_offset:y_end, x_offset:x_end] = crop_img

######################################

        UI_box(box, img, label='man', color=(255,255,255), line_thickness=2)

        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue

            # generate dynamic thickness of trails
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)


        # box text and bar
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]

        ## bounding box number id
        cv2.rectangle(
             img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), (255, 0, 255), -1)
        #text number id
        cv2.putText(img, label, (x1, y1 +
                                  t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
    return img, heartrate, bpm_text, bpm



def process_uploaded_file (stream_path, c1_text, stframe, use_webcam):
  tracker = Tracker(distance_function="iou", distance_threshold=20)
  filename = 'output.mp4'
  output_dir = 'result'
  input_file = stream_path
  cap = cv2.VideoCapture(input_file)
  fps=30
  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
  fram= 1
  inc = 1
  save_result = True
  bpm_list = []
  prev_data = []
  prev_time = 0
  
    #-------------- pulse data------------
  pulse_list = []
  plot_placeholder = st.empty()
  # Initialize an empty DataFrame for storing ECG data
  duration = 5  # Duration of ECG data to display (in seconds)
  sampling_rate = 500  # Sampling rate in Hz
  initial_t = np.linspace(0, duration, int(sampling_rate * duration))
  ecg_data = pd.DataFrame({'Time': initial_t, 'ECG': np.zeros_like(initial_t)})


  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  if save_result:
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  video_writer = cv2.VideoWriter(save_path, fourcc, fps, (int(width), int(height)))

  while True:
      ret, frame = cap.read()
      if not ret:
        break
      img_h, img_w, _ = frame.shape
      sample_frame = frame.copy()
      print(f"running frame {fram}/{frame_count}")

      output = RetinaFace.detect_faces(frame)
      face_detections = [output[face]['facial_area'] for face in output.keys()]

       #------------------------TRACKER-----------------------

      norfair_detections = get_norfair_detections(face_detections)
      tracked_objects = tracker.update(detections=norfair_detections)

      for obj in tracked_objects:
        if obj.estimate is None:
          continue
        bbox = [convert_bboxes_to_absolute(obj)]
        object_id = 0

        frame, heartrate, bpm_text, bpm = draw_boxes(frame, bbox, object_id)
        bpm_list.append(heartrate)
        pulse = np.interp(bpm, (60, 80), (0, 1))
        pulse_list.append(pulse)

        curr_time = time.time()
        fps = int(1/(curr_time - prev_time))
        prev_time = curr_time
    
        cv2.putText(frame,'FPS:' + str(fps),(50,70),cv2.FONT_HERSHEY_SIMPLEX,1,(149,255,149),2)

        stframe.image(frame, channels = 'BGR',use_column_width=True)
        c1_text.write(f"<h1 style='text-align: centre; color: red; font-size: 50px;'>{bpm_text}</h1>", unsafe_allow_html=True)
        
        pulse_frequency = bpm / 60.0

        # Generate the ECG-like waveform for a short period (e.g., 1 second)
        new_t, new_ecg_waveform = generate_ecg_waveform(1, sampling_rate, bpm)

        # Update the DataFrame with new ECG data
        new_time_values = new_t + ecg_data['Time'].iloc[-1]  # Append new time values
        new_data = pd.DataFrame({'Time': new_time_values, 'ECG': new_ecg_waveform})
        ecg_data = pd.concat([ecg_data, new_data], ignore_index=True)

        # Keep only the last 'duration' seconds of data
        ecg_data = ecg_data[ecg_data['Time'] >= (ecg_data['Time'].iloc[-1] - duration)]

        # Plot the ECG-like waveform using st.line_chart
        plot_placeholder.line_chart(ecg_data.set_index('Time'))

        # Sleep for a short duration before updating the plot
        # sleep(0.1)  # Update every 1 second (adjust as needed)

        #-----------------pulse chart----------------------------
        if save_result:
          video_writer.write(frame)

      fram+=1

  cap.release()
  video_writer.release()



def main():
    st.image('white logo new.webp')
    st.title("Breathing-Rate Monitoring App")
    st.sidebar.title("Settings")
    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    use_webcam = False
    st.sidebar.markdown('---')

    enable_gpu = st.sidebar.checkbox("Enable GPU")
    
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov",'avi','asf', 'm4v'])

    Demo_video  = "heartrate.mp4"
    tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

    if not video_file_buffer:
        if use_webcam:
            video = cv2.VideoCapture(0, cv2.CAP_ARAVIS)
            tffile.name = 0
        else:
            video = cv2.VideoCapture(Demo_video)
            tffile.name = Demo_video
            demo_video = open(tffile.name, 'rb')
            demo_bytes = demo_video.read()

            st.sidebar.text("Input video")
            st.sidebar.video(demo_bytes)

    else:
        tffile.write(video_file_buffer.read())
        demo_video = open(tffile.name, 'rb')
        demo_bytes = demo_video.read()

        st.sidebar.text("Input Video")
        st.sidebar.video(demo_bytes)

    print(tffile.name)

    stframe = st.empty()
    # col1, col2, col3 = st.columns(spec=3, gap="large")

    # with col1:
    #     st.markdown("** **")
    #     column1_text = st.markdown(" ")

    # with col2:
    #     st.markdown("**BPM**")
    #     column2_text = st.markdown("0")

    # with col3:
    #     st.markdown("** **")
    #     column3_text = st.markdown(" ")

    # st.markdown("<hr/>", unsafe_allow_html=True)
    st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
    st.sidebar.markdown("**Beathing Per Minute (BPM)**")
    bpm_readings = st.sidebar.markdown("Calculating...")

    #load model and process frame
    process_uploaded_file(tffile.name, bpm_readings, stframe, use_webcam)

    st.text('Video Processed')
    
if __name__== '__main__':
    try: 
        main()
    except SystemExit:
        pass



