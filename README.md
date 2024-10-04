# Hand Gesture Drawing with OpenCV & MediaPipe

This project is an interactive hand gesture-based drawing tool that allows you to draw and erase using your webcam. It uses **MediaPipe** for hand gesture recognition and **OpenCV** for live video processing. Different gestures are used to select colors, draw, and erase on the canvas.

## Features

- **Hand Gesture Recognition**: Detects hand landmarks using MediaPipe for real-time gesture recognition.
- **Drawing Mode**: Draw on the canvas when only the index finger is open.
- **Erasing Mode**: Erase the drawing when all five fingers are open.
- **Color Palette Selection**: Choose between predefined colors (red, yellow, blue, green) by hovering your index finger over the palette.
- **Real-Time Webcam Feed**: Draw directly on the live webcam feed using hand movements.
- **Selfie View**: The camera feed is flipped horizontally for intuitive interaction.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Tarak101/handDrawing-openCV.git
   ```

2. **Install the required dependencies**:
   ```bash
   pip install opencv-python mediapipe numpy
   ```

3. **Run the application**:
   ```bash
   python hand_gesture_drawing.py
   ```

## How to Use

1. **Launch the program**: Open the application and position yourself in front of the webcam.
2. **Select Colors**: Hover your index finger over the color palette to change colors.
3. **Draw**: Extend only your index finger to draw on the canvas.
4. **Erase**: Open all five fingers to clear the drawing.

## Gestures

- **Draw**: Index finger open, others closed.
- **Erase**: All fingers open.
- **Color Selection**: Hover your index finger over the desired color in the top-right corner.

## Tech Stack

- **Python**
- **OpenCV**
- **MediaPipe**
- **NumPy**

## Contribution

Feel free to fork the project and submit pull requests. Contributions are welcome!
