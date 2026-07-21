import cv2
import numpy as np
import socket
from datetime import datetime
from pythonosc import udp_client

# =========================================================
# カメラ設定 (高解像度に戻します)
# =========================================================
CAMERA_ID = 1
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# =========================================================
# OSC (UDP) 通信設定
# =========================================================
OSC_IP = "192.168.1.100"  # ★Arduino(ESP32)のIPアドレスに変更してください
OSC_PORT = 8888
OSC_ADDRESS = "/contour"

# =========================================================
# 白色判定のしきい値
# =========================================================
BRIGHTNESS_TH = 140
CHROMA_TH = 45
B_MINUS_R_TH = -25

def create_white_mask(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.int16)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    brightness = (r + g + b) / 3.0
    chroma = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])

    white_condition = (brightness > BRIGHTNESS_TH) & (chroma < CHROMA_TH) & ((b - r) > B_MINUS_R_TH)
    mask = white_condition.astype(np.uint8) * 255

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask

def extract_high_change_contour(mask):
    """
    「変化の多いところ」だけを抽出した輪郭データを返す
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    max_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(max_contour) < 1000:
        return None

    # --- 変化の多い部分（角）だけを残す処理 ---
    # 0.005 という数値が「どれくらい細かい変化まで拾うか」の感度です。
    # 小さくするほど細かい変化を拾い（頂点が増える）、大きくするほど大雑把になります。
    epsilon = 0.005 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    
    return approx.reshape(-1, 2).tolist()

def main():
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Webカメラを開けませんでした。")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

    print("========== 操作方法 ==========")
    print("q : プログラムを終了")
    print("==============================")

    while True:
        # 意図的なフレームレート低下 (time.sleep) を削除し、最速で回します
        ret, frame = cap.read()
        if not ret:
            print("フレームを取得できませんでした")
            break

        mask = create_white_mask(frame)
        
        # 変化の大きい座標のみを取得
        contour_points = extract_high_change_contour(mask)

        result = frame.copy()
        
        green_overlay = np.zeros_like(frame)
        green_overlay[mask == 255] = (0, 255, 0)
        result = cv2.addWeighted(result, 1.0, green_overlay, 0.3, 0)

        if contour_points is not None:
            # OSC送信用に一次元配列に変換: [x1, y1, x2, y2, ...]
            flat_points = []
            for pt in contour_points:
                flat_points.extend([int(pt[0]), int(pt[1])])
            
            try:
                client.send_message(OSC_ADDRESS, flat_points)
                # 送信できている座標数を確認
                print(f"[OSC送信] 抽出された頂点数: {len(contour_points)} (パケット内要素数: {len(flat_points)})")
            except Exception as e:
                print(f"OSC送信エラー: {e}")

            # 描画処理: 残った「変化の大きい頂点」を黄色い点で可視化
            pts = np.array(contour_points, dtype=np.int32)
            cv2.polylines(result, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            for pt in pts:
                cv2.circle(result, tuple(pt), 4, (0, 255, 255), -1)
        else:
            cv2.putText(result, "Not Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("Camera", result)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    