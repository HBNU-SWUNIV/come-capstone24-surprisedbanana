
import numpy as np 
import pandas as pd

from scipy import ndimage

from joblib import load
from collections import Counter
import socket
import serial
import time
import numpy as np 
import pandas as pd

def maximum(raw, box_size, mode='nearest'): # Maximum Filter
    raw_maximum=ndimage.maximum_filter(raw,box_size,mode=mode)
    return raw_maximum.real

def minimum(raw, box_size, mode='nearest'): # Maximum Filter
    raw_minimum=ndimage.minimum_filter(raw,box_size,mode=mode)
    return raw_minimum.real

def denoise_fft(data, ifftn): # Fast Fourier Transformation
    fft_signal = np.fft.fft(data)
    
    # Reconstruct the original signal
    fft_signal[ifftn:len(fft_signal)//2]=0
    fft_signal[len(fft_signal)//2:-ifftn]=0
    reconstructed_signal = np.fft.ifft(fft_signal)
    
    return reconstructed_signal.real

def smooth(x,beta): # Kaiser Window Smoothing
    window_len=11  # extending the data at beginning and at the end to apply the window at the borders

    s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    w = np.kaiser(window_len,beta)
    y = np.convolve(w/w.sum(),s,mode='valid')
    return y[5:len(y)-5]


model = load('model777.joblib')


def connect_to_unity():
    sock = None
    while sock is None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('127.0.0.1', 5005))  # 유니티 서버 IP와 포트
            print("Connected to Unity")
        except socket.error:
            print("Connection failed, retrying in 5 seconds...")
            time.sleep(5)
            sock = None
    return sock

# 자동 재연결을 시도
sock = connect_to_unity()

ser = serial.Serial('COM3', 115200, timeout=1)
skip = 1
data_batch = []
rock_count = 0
count = 0
prev = 0
prev_list = []
# 데이터 읽기를 시도하는 루프
try:
    while True:
        if ser.in_waiting > 0:
            data = ser.readline()
            # ISO-8859-1
            decoded_data = data.decode('ISO-8859-1').rstrip()

            if skip == 0:
                try:
                    int_data = [int(x) for x in decoded_data.split(',')]
                    data_batch.append(int_data)
                except ValueError:
                    print("데이터 변환 중 오류 발생:", decoded_data)

                # 190행이 모였는지 확인
                if len(data_batch) == 190:
                    # 리스트를 DataFrame으로 변환
                    df = pd.DataFrame(data_batch, columns=['a0', 'a1', 'a2', 'a3', 'a4', 'a5'])
                    depths_list = []
                    for i in range(6):
                        column = f'a{i}'  # 필터링된 데이터의 각 채널 컬럼명
                        depths = smooth(denoise_fft(maximum(df[column], 5)-minimum(df[column], 5), 5), 5)
                        depths_list.append(depths)

                    df_2d = pd.DataFrame(depths_list)

                    df_2d_cleaned = df_2d.dropna(axis=1, how='any')

                    result = df_2d_cleaned.to_numpy().flatten()

                    probabilities = model.predict_proba(result)

                    # 임계값 설정 (예: 70% 이상일 때만 예측 결과를 사용)
                    threshold = 0.8
                    # 각 샘플의 확률을 순회
                    
                    # 각 클래스에 대한 예측 확률 중 가장 높은 값을 가져옵니다.
                    max_prob = np.max(probabilities)  # 가장 높은 확률 값을 구합니다.
                    max_arg = np.argmax(probabilities)

                    # 가장 높은 확률 값이 threshold 이상인지 확인
                    if max_prob >= threshold:
                        count = max_arg
                            
                    # 0: rest, 1:rock, 2: scissors, 3: right, 4: left, 5: fire, 6: paper, 7: paper_left, 8: paper_right
                    # +100 : start
                    
                    if count == 0:
                        rock_count=0
                        print("rest")
                    elif count == 1:
                        rock_count+=1
                        print("rock")
                    elif count == 2:
                        rock_count=0
                        print("scissors")
                    elif count == 3:
                        rock_count=0
                        print("right")
                    elif count == 4:
                        rock_count=0
                        print("left")
                    elif count == 5:
                        rock_count=0
                        print("fire")
                    elif count == 6:
                        rock_count=0
                        print("paper")
                    elif count == 7:
                        rock_count=0
                        print("paper_left")
                    elif count == 8:
                        rock_count=0
                        print("paper_right")
                        
                    if rock_count >= 30:
                        sock.sendall(str(count+200).encode())
                    elif prev == count:
                        sock.sendall(str(count).encode())
                    else:
                        sock.sendall(str(count+100).encode())
                    prev = count

                    data_batch = data_batch[20:]

            skip = 0

except KeyboardInterrupt:
    print("Connection lost, reconnecting...")
    sock.close()
    sock = connect_to_unity()
    print('프로그램을 종료합니다.')
    ser.close()  # 시리얼 포트 닫기