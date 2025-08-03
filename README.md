# Nose Draw Game 🎨👃

A fun and interactive drawing game where you use your nose to draw shapes! Built with Python, OpenCV, and MediaPipe.

## 🎮 Features

- **Nose-controlled drawing**: Use your nose movements to draw on the canvas
- **Multiple shapes**: Draw window, star, triangle, cloud, tree, house, and flower
- **Real-time scoring**: Get instant feedback on your drawing accuracy
- **High score system**: Track and display top 5 scores with player names
- **Full-screen gameplay**: Immersive side-by-side camera and canvas display
- **Reference images**: See what to draw with clear reference shapes
- **Pause/Resume**: Take breaks during gameplay
- **Snapshot system**: Save your drawing progress
- **Tough scoring**: Challenging scoring system that rewards accuracy

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- Webcam
- Good lighting for face detection

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/nose-draw-game.git
cd nose-draw-game
```

2. Install required packages:
```bash
pip install opencv-python mediapipe numpy scikit-image
```

3. Run the game:
```bash
python nosedraw.py
```

## 🎯 How to Play

1. **Start the game**: Press ENTER from the main menu
2. **Enter your name**: Type your name in the dialog (or press ESC for Anonymous)
3. **Position yourself**: Sit in front of your webcam with good lighting
4. **Draw with your nose**: Move your nose to draw on the canvas
5. **Match the shape**: Try to draw the reference shape as accurately as possible
6. **Submit**: Press ENTER when you're done drawing
7. **Check your score**: See how well you did and if you made the high score list!

## 🎮 Controls

| Key | Action |
|-----|--------|
| **ENTER** | Start game / Submit drawing |
| **P** | Pause/Resume game |
| **C** | Clear canvas |
| **S** | Take snapshot |
| **V** | View snapshots |
| **Q** | Return to menu |
| **R** | Restart game |
| **ESC** | Quit game |

## 🏆 High Score System

- Top 5 scores are displayed on the main menu
- Scores include player name, percentage, and shape drawn
- Automatic high score detection and storage
- Persistent score tracking across game sessions

## 🎨 Available Shapes

- **Window**: Complex window with frames and panes
- **Star**: 5-pointed star pattern
- **Triangle**: Simple triangle shape
- **Cloud**: Multiple overlapping circles
- **Tree**: Trunk with foliage circles
- **House**: Complete house with roof, door, and windows
- **Flower**: Petals, stem, and leaves

## 🛠️ Technical Details

### Technologies Used
- **OpenCV**: Computer vision and image processing
- **MediaPipe**: Face mesh detection and nose tracking
- **NumPy**: Numerical computing and array operations
- **Scikit-image**: Structural similarity index for scoring

### Game Features
- **Real-time nose tracking**: Smooth drawing with nose movements
- **Advanced scoring**: SSIM-based scoring with edge detection
- **Full-screen display**: Immersive gaming experience
- **Responsive UI**: Interactive menus and dialogs

## 📁 Project Structure

```
nose-draw-game/
├── nosedraw.py          # Main game file
├── test_camera.py       # Camera test utility
├── clear_scores.py      # Clear high scores utility
├── test_high_scores.py  # Add sample high scores
├── high_scores.json     # High scores data (auto-generated)
└── README.md           # This file
```

## 🎯 Scoring System

The game uses a sophisticated scoring algorithm that considers:
- **Structural similarity** (SSIM) between drawn and reference shapes
- **Pixel density** comparison
- **Edge detection** accuracy
- **Shape-specific penalties** for different complexity levels
- **Drawing amount** penalties (too much or too little)

## 🐛 Troubleshooting

### Common Issues

1. **Camera not detected**:
   - Ensure your webcam is connected and working
   - Check if other applications are using the camera
   - Try running `test_camera.py` to verify camera access

2. **Face not detected**:
   - Improve lighting conditions
   - Position yourself closer to the camera
   - Ensure your face is clearly visible

3. **Game not starting**:
   - Check that all dependencies are installed
   - Ensure Python 3.7+ is being used
   - Verify OpenCV and MediaPipe installation

### Performance Tips

- **Good lighting**: Ensure your face is well-lit for better detection
- **Stable position**: Sit at a comfortable distance from the camera
- **Smooth movements**: Move your nose slowly for better drawing accuracy
- **Clear background**: Avoid cluttered backgrounds for better face detection

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 🙏 Acknowledgments

- **OpenCV** for computer vision capabilities
- **MediaPipe** for face mesh detection
- **NumPy** for numerical operations
- **Scikit-image** for image similarity metrics

## 🎉 Have Fun!

Enjoy drawing with your nose! Challenge your friends to beat your high scores and see who can draw the most accurate shapes.

