import cv2
import numpy as np
import socket
import json
from datetime import datetime

# 事前に pip install pyk4a を実行してください
from pyk4a import PyK4A, Config, ColorResolution, DepthMode

# =========================================================
# UDP通信設定
# =========================================================
UDP_IP = "192.168.1.100"  # ★Arduino(ESP32)のIPアドレスに変更
UDP_PORT = 8888

# =========================================================
# 深度（距離）のしきい値設定 (単位: ミリメートル)
# =========================================================
# テトラポットが存在する距離の範囲を指定します。
# 例: カメラから50cm(500mm) 〜 80cm(800mm) の間にある物体だけを抽出
DEPTH_MIN = 500
DEPTH_MAX = 800

def create_depth_mask(depth_image):
    """
    深度画像から、指定した距離範囲の物体だけを抽出した二値マスクを作成する
    """
    # 指定した距離の範囲内なら255（白）、それ以外なら0（黒）にする
    mask = cv2.inRange(depth_image, DEPTH_MIN, DEPTH_MAX)

    # ノイズ除去（赤外線の反射ノイズなどを消す）
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask

def extract_high_change_contour(mask):
    """
    「変化の多いところ」だけを抽出した輪郭データを返す
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # ノイズを拾わないよう、最大の輪郭を取得
    max_contour = max(contours, key=cv2.contourArea)
    
    # ピクセル面積が小さすぎる場合は無視
    if cv2.contourArea(max_contour) < 1000:
        return None

    # 変化の多い部分（角）だけを残す処理
    epsilon = 0.005 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    
    return approx.reshape(-1, 2).tolist()

def main():
    # =========================================================
    # Azure Kinectの初期設定
    # =========================================================
    k4a = PyK4A(Config(
        color_resolution=ColorResolution.RES_720P,
        depth_mode=DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    ))
    k4a.start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print("========== 操作方法 ==========")
    print("q : プログラムを終了")
    print(f"※ 抽出した座標を {UDP_IP}:{UDP_PORT} へ UDP(JSON)送信しています")
    print("==============================")

    while True:
        # Kinectからフレーム（カラー画像と深度画像）を取得
        capture = k4a.get_capture()
        if capture.depth is None or capture.color is None:
            continue

        # カラー画像 (表示用)
        color_image = capture.color[:, :, :3]  # BGRAからBGRへ変換
        
        # 深度画像 (距離データ mm)
        depth_image = capture.depth

        # 深度情報を使ってテトラポットの領域だけを切り出す
        mask = create_depth_mask(depth_image)
        
        # 変化の大きい座標のみを取得
        contour_points = extract_high_change_contour(mask)

        # 表示用の合成
        result = color_image.copy()
        green_overlay = np.zeros_like(color_image)
        green_overlay[mask == 255] = (0, 255, 0)
        result = cv2.addWeighted(result, 1.0, green_overlay, 0.3, 0)

        if contour_points is not None:
            # UDP(JSON)送信
            payload = {"contour": contour_points}
            json_str = json.dumps(payload)
            payload_size = len(json_str.encode('utf-8'))
            
            try:
                sock.sendto(json_str.encode('utf-8'), (UDP_IP, UDP_PORT))
            except OSError as e:
                print(f"UDP送信エラー: {e}")

            # 描画処理: 近似された頂点を黄色で描画
            pts = np.array(contour_points, dtype=np.int32)
            cv2.polylines(result, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            for pt in pts:
                cv2.circle(result, tuple(pt), 4, (0, 255, 255), -1)
        else:
            cv2.putText(result, "Not Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 深度画像を人間が見やすいようにグレースケール化して表示
        depth_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        cv2.imshow("Color & Contour", result)
        cv2.imshow("Depth Mask", mask)
        cv2.imshow("Depth Raw", depth_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    k4a.stop()
    sock.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    