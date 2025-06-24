# Turret Control and Calibration System

## Turret Control

### TCP Protocol

The turret uses a simple TCP protocol on port 8888. Commands are ASCII text ending with `\n`:

```
LASER:ON               # Turn laser on
LASER:OFF              # Turn laser off
ROTATE:yaw,pitch       # Move to position
PING                   # Test connection
```

Coordinate system:
- Yaw: -1.0 (left) to 0.0 (right) 
- Pitch: 0.0 (down) to 0.5 (up)

Example: `ROTATE:-0.25,0.3\n` moves the turret to aim left and slightly up.

## Calibration Process

### 1. Checkerboard Detection

The system detects a checkerboard pattern to establish reference points. Checkerboards work well because:
- Corner intersections provide precise coordinates
- Computer vision can detect them automatically  
- One pattern gives 20-50+ reference points
- Each corner gets numbered (0, 1, 2...)

### 2. Manual Initial Setup

Before auto-calibration starts, you manually position the laser on corner 0 using WASD keys. This gives the system a starting reference point and confirms everything is working.

### 3. Automatic Calibration

The system then automatically moves the laser to each corner using PID control:

**Movement Control:**
- Uses camera feedback to track the laser dot in real-time
- PID controller adjusts movement based on distance to target
- Close to target = small precise movements
- Far from target = larger fast movements
- Special handling for tricky row-to-row transitions

**Computer Vision Tracking:**
- Detects red laser dot using color filtering
- Finds circular shapes of appropriate size
- Calculates precise center position
- Provides feedback for movement control

### 4. Data Collection

Each corner visit records:
- Image coordinates (x, y pixels)
- Turret angles (yaw, pitch) 

This data solves the coordinate transformation problem. We need multiple points because:
- Camera lenses distort images
- Perspective changes across the image
- Turret response isn't perfectly linear
- More points = better accuracy

## Calibration Models

The system builds three different models and uses the best one:

### Polynomial Regression
Fits curved relationships using polynomial features like x², xy, y².

**Pros:** Fast, smooth predictions, works well between calibration points
**Cons:** Bad data points mess up the whole model

### RANSAC Regression  
Same as polynomial but automatically ignores outlier data points.

**Pros:** Handles bad calibration data, robust results
**Cons:** Slower, may reject good but unusual points

### Homography Transformation
Uses camera geometry principles to model the transformation.

**Pros:** Geometrically correct, works great for flat targets
**Cons:** Assumes everything is on one plane, limited flexibility