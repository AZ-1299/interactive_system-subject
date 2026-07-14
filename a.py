import cv2
import numpy as np
import csv
from datetime import datetime

# =========================================================
# カメラ設定
# =========================================================
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# =========================================================
# 白色判定のしきい値
# =========================================================
BRIGHTNESS_TH = 140
CHROMA_TH = 45
B_MINUS_R_TH = -25

# 抽出するキーポイントの数
KEYPOINT_COUNT = 4

def create_white_mask(frame):
    """
    白領域を抽出し、ノイズを除去した二値マスクを作成する
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.int16)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # 明るさと色味の計算
    brightness = (r + g + b) / 3.0
    chroma = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])

    # 白っぽい領域を抽出
    white_condition = (brightness > BRIGHTNESS_TH) & (chroma < CHROMA_TH) & ((b - r) > B_MINUS_R_TH)
    mask = white_condition.astype(np.uint8) * 255

    # モルフォロジー処理によるノイズ除去
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask

def extract_4_keypoints(mask):
    """
    マスクから最大の白領域を見つけ、その上部境界線から4つのキーポイントを抽出する
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    max_contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(max_contour) < 1000:
        return None

    points = max_contour.reshape(-1, 2)

    top_edge = {}
    for x, y in points:
        if x not in top_edge or y < top_edge[x]:
            top_edge[x] = y

    if not top_edge:
        return None

    sorted_x = sorted(top_edge.keys())
    
    if len(sorted_x) < KEYPOINT_COUNT:
        return None

    indices = np.linspace(0, len(sorted_x) - 1, KEYPOINT_COUNT, dtype=int)
    
    keypoints = []
    for idx in indices:
        target_x = sorted_x[idx]
        keypoints.append([target_x, top_edge[target_x]])

    return np.array(keypoints)

# =========================================================
# メイン処理
# =========================================================
def main():
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Webカメラを開けませんでした。CAMERA_IDを確認してください。")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("========== 操作方法 ==========")
    print("Enter : 現在の画面で4つのキーポイントを取得・確定")
    print("c     : 取得したキーポイントをクリア（リセット）")
    print("p     : 4点のピクセル座標をコンソールへ表示")
    print("s     : 画像と座標CSVを保存")
    print("q     : プログラムを終了")
    print("==============================")

    # 取得したキーポイントを保持する変数
    locked_keypoints = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレームを取得できませんでした")
            break

        # 白領域のマスク取得
        mask = create_white_mask(frame)
        result = frame.copy()
        
        # 位置合わせ支援のため、白領域は常に半透明の緑でハイライト
        green_overlay = np.zeros_like(frame)
        green_overlay[mask == 255] = (0, 255, 0)
        result = cv2.addWeighted(result, 1.0, green_overlay, 0.3, 0)

        # Enterキーで確定されたキーポイントがあれば描画
        if locked_keypoints is not None:
            pts = locked_keypoints.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(result, [pts], False, (0, 0, 255), 3, cv2.LINE_AA)

            for i, pt in enumerate(locked_keypoints):
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(result, (x, y), 8, (255, 255, 0), -1)
                cv2.circle(result, (x, y), 10, (0, 0, 0), 2)
                cv2.putText(result, f"P{i+1}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.putText(result, "Keypoints Locked", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(result, "Press ENTER to detect keypoints", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Camera", result)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF

        # 終了
        if key == ord('q'):
            break
            
        # Enterキー (OpenCVではASCIIコードの13)
        elif key == 13:
            locked_keypoints = extract_4_keypoints(mask)
            if locked_keypoints is not None:
                print("キーポイントを取得しました。")
            else:
                print("白領域が検出できなかったため、取得に失敗しました。")

        # cキーでクリア
        elif key == ord('c'):
            locked_keypoints = None
            print("キーポイントをリセットしました。")

        # 座標を出力
        elif key == ord('p'):
            if locked_keypoints is not None:
                print("\n========== 4つのキーポイント座標 ==========")
                for i, pt in enumerate(locked_keypoints):
                    print(f"P{i+1}: (x: {pt[0]}, y: {pt[1]})")
                print("==========================================\n")
            else:
                print("キーポイントが取得されていません。Enterキーを押してください。")

        # 保存
        elif key == ord('s'):
            if locked_keypoints is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"capture_{timestamp}.png", result)
                
                with open(f"keypoints_{timestamp}.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Point", "X", "Y"])
                    for i, pt in enumerate(locked_keypoints):
                        writer.writerow([f"P{i+1}", pt[0], pt[1]])
                print(f"画像とCSV(keypoints_{timestamp}.csv)を保存しました。")
            else:
                print("保存する座標がありません。Enterキーを押してください。")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    