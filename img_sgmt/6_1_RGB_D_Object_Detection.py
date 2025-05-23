'''
전방 300mm 위치에 있는 턱이 50mm 이상일 때 레그 모드로 전환하라는 문구가 뜨는 코드이다.
이때 카메라의 지상으로부터의 높이는 130mm이다.
ROI(Region Of Interest)에 대해서만 deprojection을 진행해 물체의 형상을 파악한다.
이때 deprojection은 무거운 연산이기 때문에 SKIP_FRAMES를 이용해 화면 렉을 줄인다.
RGB-D 카메라에서 카메라로부터 물체까지의 거리가 Z축 정보이다. X축은 좌우, Y축은 상하.
여기에 YOLOv8 nano를 이용한 object detection까지 진행한 코드이다.
'''

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

def main():
    # 1) 상수 설정
    CAMERA_HEIGHT = 0.13 # 카메라 렌즈가 지상으로부터 130mm 위치에 있다고 설정
    PIXEL_OFFSET = 60 # 화면 하단에서 위로 60px (≈300 mm 전방) 이는 경험에 의한 값이므로 적절하게 수정 필요요
    STEP_THRESH = 0.05 # 턱 감지 임계: 50 mm
    ROI_HALF_W = 10 # ROI 너비: 중앙 ±10px
    SKIP_FRAMES = 5 # 5프레임에 1번만 무거운 연산

    # 2) YOLOv8 nano 모델 로드
    model = YOLO('yolov8n.pt')

    # 3) RealSense 초기화 (15fps로 낮춤)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth,  640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color,  640, 480, rs.format.bgr8, 15)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    frame_idx = 0 # deproject 계산을 위해 프레임 계산
    try:
        while True:
            # 1) 매 프레임 수신 및 정렬렬
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            
            # 2) ROI 정의
            frame_idx += 1
            h, w = depth_frame.get_height(), depth_frame.get_width()
            cx = w // 2
            y_ground = h -10
            y_front = max(0, y_ground - PIXEL_OFFSET)
            
            # 3) 화면 데이터 Numpy로 변환
            depth_img = np.asanyarray(depth_frame.get_data())
            color_img = np.asanyarray(color_frame.get_data())

            # 4) 무거운 ROI -> 3D 연산은 SKIP_FRAMES 주기로만 수행
            if frame_idx % SKIP_FRAMES == 0:
                # depth 정보를 얻을 수 있는 instance
                intrin = depth_frame.get_profile() \
                                    .as_video_stream_profile() \
                                    .get_intrinsics()
                
                # ROI 스캔해서 높이 중간값 구하기
                heights = []
                for dx in range(-ROI_HALF_W, ROI_HALF_W+1):
                    x = cx + dx
                    if x < 0 or x >= w:
                        continue
                    z = depth_frame.get_distance(x, y_front)
                    if z <= 0:
                        continue
                    p = rs.rs2_deproject_pixel_to_point(
                            intrin, [x, y_front], z)
                    heights.append(CAMERA_HEIGHT - p[1])
                
                if heights:
                    object_height = float(np.median(heights))
                else:
                    object_height = 0.0

                # 턱 감지
                if object_height > STEP_THRESH:
                    cv2.putText(color_img, "Switch to Leg Mode", 
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                1.2, (0, 0, 255), 2)
                    
                # 디버깅을 위한 ROI 표시
                cv2.rectangle(color_img, 
                              (cx - ROI_HALF_W, y_front - 2), 
                              (cx + ROI_HALF_W, y_front + 2), 
                              (255, 0, 0), 1)
                cv2.circle(color_img, (cx, y_front), 3, (0, 255, 0), -1)

            # 5) YOLOv8 nano Object Detection
            results = model.predict(source = color_img, 
                                    conf = 0.25, # confidence threshold
                                    iou = 0.45, # NMS IoU threshold
                                    imgsz = 640, # inference size
                                    device = 'cpu')
            
            # results 는 list of Results, 보통 [0]만 사용
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"
                # Bounding box
                cv2.rectangle(color_img, (x1,y1), (x2,y2), (0,255,0), 2)
                # Label
                cv2.putText(color_img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                
            # 매 프레임: 컬러맵 적용 & 화면 출력
            depth_cm = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=0.03),
                cv2.COLORMAP_JET
            )
            vis = np.hstack((color_img, depth_cm))
            cv2.imshow("RealSense", vis)

            # 종료
            if cv2.waitKey(1) == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()