'''
전방 300mm 위치에 있는 턱이 50mm 이상일 때 레그 모드로 전환하라는 문구가 뜨는 코드
이때 카메라의 지상으로부터의 높이는 130mm
'''

import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # 1) 설정
    CAMERA_HEIGHT = 0.13    # 렌즈 높이: 130 mm → 0.13 m
    PIXEL_OFFSET = 60      # 화면 하단에서 위로 60px (≈300 mm 전방)
    # PIXEL_OFFSET은 경험에 의한 값. 적절하게 바꿔주어야 함.
    STEP_THRESH = 0.05    # 50 mm

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth,  640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color,  640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            h, w = depth_frame.get_height(), depth_frame.get_width()
            cx = w // 2
            y_ground = h - 10
            y_front = max(0, y_ground - PIXEL_OFFSET)

            # 2) intrinsics 가져오기
            intrin = depth_frame.get_profile() \
                               .as_video_stream_profile() \
                               .get_intrinsics()

            # 3) 전방 픽셀 깊이 얻고 3D 포인트로 변환
            z_front = depth_frame.get_distance(cx, y_front)
            p_front = rs.rs2_deproject_pixel_to_point(intrin,
                                                     [cx, y_front],
                                                     z_front)
            # p_front[1] 은 카메라 축 기준 아래 방향의 거리(m)

            # 4) 지상 기준 물체 높이 계산
            #    : 카메라 높이 – (카메라에서 물체까지의 수직 거리)
            object_height = CAMERA_HEIGHT - p_front[1]

            # 5) 문구 표시
            color_img = np.asanyarray(color_frame.get_data())
            if object_height > STEP_THRESH:
                cv2.putText(
                    color_img,
                    "Switch to Leg Mode",
                    (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0,0,255),
                    2
                )

            # 6) 디버깅용 포인트 표시
            cv2.circle(color_img, (cx, y_front), 4, (255,0,0), -1)

            # 7) 시각화
            depth_img = np.asanyarray(depth_frame.get_data())
            depth_cm = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=0.03),
                cv2.COLORMAP_JET
            )
            vis = np.hstack((color_img, depth_cm))
            cv2.imshow("RealSense", vis)

            # 8) 종료 키
            if cv2.waitKey(1) == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()