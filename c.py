import cv2
import numpy as np
import socket
import json
from datetime import datetime

# =========================================================
# カメラ設定 (高解像度)
# =========================================================
CAMERA_ID = 1
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# =========================================================
# UDP通信設定
# =========================================================
UDP_IP = "192.168.1.100"  # ★Arduino(ESP32)のIPアドレスに変更してください
UDP_PORT = 8888           # Arduino側で待ち受けるポート番号

# =========================================================
# 白色判定のしきい値
# =========================================================
BRIGHTNESS_TH = 140
CHROMA_TH = 45
B_MINUS_R_TH = -25

def create_white_mask(frame):
    """
    白領域を抽出し、ノイズを除去した二値マスクを作成する
    """
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

    # 変化の多い部分（角）だけを残す処理
    epsilon = 0.005 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    
    # JSONで送りやすいように二次元配列のリストに変換
    return approx.reshape(-1, 2).tolist()

def main():
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Webカメラを開けませんでした。")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # UDP通信用ソケットの準備
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print("========== 操作方法 ==========")
    print("q : プログラムを終了")
    print(f"※ 抽出した座標を {UDP_IP}:{UDP_PORT} へ UDP(JSON)送信しています")
    print("==============================")

    while True:
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
            # --- JSON形式にしてUDP送信 ---
            payload = {"contour": contour_points}
            json_str = json.dumps(payload)
            payload_size = len(json_str.encode('utf-8'))
            
            try:
                sock.sendto(json_str.encode('utf-8'), (UDP_IP, UDP_PORT))
                # サイズと頂点数を出力してパケット溢れがないか確認
                print(f"[JSON送信] 頂点数: {len(contour_points)}, サイズ: {payload_size} bytes")
            except OSError as e:
                # 複雑すぎる図形が映って1400バイトを超えた場合の警告
                print(f"UDP送信エラー (サイズ超過の可能性): {e} / {payload_size} bytes")
            # -----------------------------

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
    sock.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    