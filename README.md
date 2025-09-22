
# Driver Monitoring System (DMS)

An intelligent **Driver Monitoring System** that detects driver drowsiness and distraction in real-time, providing alerts and simulating a virtual car interior.


## Features

- **Drowsiness Detection**: Monitors eye closure and head nods to detect fatigue.  
- **Alerts**: Triggers horn and warning sounds based on driver status.  
- **Driver Status**: Displays real-time status: GREEN, YELLOW, RED.  
- **Virtual Car Interior**: Uses a car interior background for realistic simulation.  
- **Smooth Speed Animation**: Simulates vehicle speed changes when the driver is drowsy.



## Libraries Used

- **#OpenCV**: Real-time video processing & display  
- **#MediaPipe**: Face mesh, hands, and selfie segmentation  
- **#Pygame**: Sound alerts & horn  
- **#Python**: Core logic & integration  

## Installation

Install my-project with npm

```bash
git clone https://github.com/varadrz/driver-monitoring-system.git
cd driver-monitoring-system
```
    
    pip install opencv-python mediapipe pygame numpy


## Working
How It Works

- Uses MediaPipe Face Mesh to detect eyes and head pose.

- Calculates Eye Aspect Ratio (EAR) to detect eye closure.

- Estimates head pitch for forward nod detection.

- Uses MediaPipe Selfie Segmentation to overlay the virtual car interior.

- Integrates sound alerts and speed animation via Pygame.
## Contributing

Contributions are always welcome!

Feel free to fork the repo and submit improvements. Suggestions and feedback are welcome!

Please adhere to this project's `code of conduct`.

