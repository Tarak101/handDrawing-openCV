import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
drawing_image = np.zeros((480, 640, 3), dtype=np.uint8)
prev_x, prev_y = None, None

colors = {
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0)
}
color_names = list(colors.keys())
current_color = colors['red']

def draw_color_palette(image):
    """Draw color palette on the top right horizontally with spacing."""
    height, width, _ = image.shape
    palette_x = width - 540  
    spacing = 20  
    box_size = 100  
    for i, color_name in enumerate(color_names):
        color = colors[color_name]
        cv2.rectangle(image, (palette_x + i * (box_size + spacing), 10), (palette_x + box_size + i * (box_size + spacing), 10 + box_size), color, -1)

def check_if_hand_open(hand_landmarks, image_shape):
    """Check if all five fingers are open."""
    image_height, image_width = image_shape[:2]
    fingers_tips_ids = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                        mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                        mp_hands.HandLandmark.PINKY_TIP]
    fingers_tips = [hand_landmarks.landmark[tip].y * image_height for tip in fingers_tips_ids]
    fingers_mcp = [hand_landmarks.landmark[mcp].y * image_height for mcp in
                   [mp_hands.HandLandmark.THUMB_CMC, mp_hands.HandLandmark.INDEX_FINGER_MCP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_MCP,
                    mp_hands.HandLandmark.PINKY_MCP]]
    return all(tip < mcp for tip, mcp in zip(fingers_tips, fingers_mcp))

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        drawing_image = cv2.resize(drawing_image, (image.shape[1], image.shape[0]))

        # Convert the BGR image to RGB.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Draw hand landmarks on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=2)
                )

                # If all fingers are open --> clear the drawing image.
                if check_if_hand_open(hand_landmarks, image.shape):
                    drawing_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
                    continue
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                x1 = int(index_finger_tip.x * image.shape[1])
                y1 = int(index_finger_tip.y * image.shape[0])
                x2 = int(thumb_tip.x * image.shape[1])
                y2 = int(thumb_tip.y * image.shape[0])

                # Checking if the index finger tip is over the color palette.
                palette_x = image.shape[1] - 540  
                spacing = 20  
                box_size = 100  
                for i, color_name in enumerate(color_names):
                    if palette_x + i * (box_size + spacing) <= x1 <= palette_x + box_size + i * (box_size + spacing) and 10 <= y1 <= 10 + box_size:
                        current_color = colors[color_name]

                # Draw only if the index finger and thumb form an "L" shape.
                if abs(x1 - x2) > 50 and abs(y1 - y2) > 50 and y2 > y1 and x2 > x1:
                    if prev_x is not None and prev_y is not None:
                        cv2.line(drawing_image, (prev_x, prev_y), (x1, y1), current_color, 5)
                    prev_x, prev_y = x1, y1
                else:
                    # If the fingers do not form an "L" shape, stop drawing.
                    prev_x, prev_y = None, None
        else:
            prev_x, prev_y = None, None
        draw_color_palette(image)

        # Combine the webcam image with the drawing image.
        combined_image = cv2.addWeighted(image, 0.5, drawing_image, 0.5, 0)

        cv2.imshow('MediaPipe Hands Drawing', cv2.flip(combined_image, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()