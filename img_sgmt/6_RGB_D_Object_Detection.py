'''
레그 모드 전환 문구가 뜨면서 동시에 YOLOv8 nano 모델을 이용해 object detection을 하는 코드드
'''

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

def main():
    # ─── 1) 설정 ─────────────────────────────────────────────────────────
    CAMERA_HEIGHT = 0.13    # 렌즈 높이: 130 mm → 0.13 m
    STEP_THRESH   = 0.05    # 50 mm
    PIXEL_OFFSET  = 60      # 하단에서 위로 60px: 테스트 환경 기준

    # YOLOv8 nano 모델 로드
    model = YOLO('yolov8n.pt')  
    # ─────────────────────────────────────────────────────────────────────

    # ─── 2) RealSense 초기화 ────────────────────────────────────────────
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.depth,  640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color,  640, 480, rs.format.bgr8, 30)
    profile  = pipeline.start(config)
    align    = rs.align(rs.stream.color)
    # ─────────────────────────────────────────────────────────────────────

    try:
        while True:
            # 1) 프레임 수신 및 정렬
            frames      = pipeline.wait_for_frames()
            aligned     = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 2) NumPy 변환
            depth_img = np.asanyarray(depth_frame.get_data())
            color_img = np.asanyarray(color_frame.get_data())
            h, w      = depth_img.shape
            cx        = w // 2
            y_ground  = h - 10
            y_front   = max(0, y_ground - PIXEL_OFFSET)

            # 3) 턱 높이 계산 (지상 기준)
            intrin   = depth_frame.get_profile() \
                                 .as_video_stream_profile() \
                                 .get_intrinsics()
            z_front  = depth_frame.get_distance(cx, y_front)
            p_front  = rs.rs2_deproject_pixel_to_point(intrin, [cx, y_front], z_front)
            object_h = CAMERA_HEIGHT - p_front[1]
            
            # 4) 턱 감지 문구
            if object_h > STEP_THRESH:
                cv2.putText(color_img, "Switch to Leg Mode",
                            (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0,0,255), 2)
            # 디버깅용 포인트 표시
            cv2.circle(color_img, (cx, y_front), 5, (255,0,0), -1)

            # ─── 5) YOLOv8-nano 오브젝트 디텍션 ─────────────────────────────
            # predict() 에서 size, conf 항목을 조정할 수 있습니다.
            results = model.predict(source=color_img,
                                    conf=0.25,    # confidence threshold
                                    iou=0.45,     # NMS IoU threshold
                                    imgsz=640,    # inference size
                                    device='cpu') # CPU 혹은 '0' for GPU

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
            # ─────────────────────────────────────────────────────────────────

            # 6) 컬러맵 적용된 깊이 영상
            depth_cm = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # 7) 화면 병합 & 출력
            vis = np.hstack((color_img, depth_cm))
            cv2.imshow("RealSense + YOLOv8-nano", vis)

            # 8) 종료키(ESC)
            if cv2.waitKey(1) == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
