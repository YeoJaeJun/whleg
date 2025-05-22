import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # 1) 파이프라인 설정
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 2) 스트림(enable_stream) 설정: 깊이(depth)와 컬러(color)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # 3) 스트리밍 시작
    pipeline.start(config)
    
    try:
        while True:
            # 4) 프레임 대기 및 가져오기
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # 5) NumPy 배열로 변환
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # 6) 깊이 영상에 컬러맵 적용 (시각화 용이) alpha = 255/(원하는 최대 거리(mm))
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.3),
                cv2.COLORMAP_JET
            )
            
            # 7) 두 영상을 가로로 붙여서 한 윈도우에 표시
            images = np.hstack((color_image, depth_colormap))
            
            cv2.imshow('RealSense Color + Depth', images)
            
            # 8) ESC 키(키코드 27) 누르면 루프 종료
            if cv2.waitKey(1) == 27:
                break

    finally:
        # 9) 스트리밍 정리
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
