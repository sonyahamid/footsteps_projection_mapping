# Footsteps Projection Mapping Demo

This project is a small **projection mapping prototype** that simulates footsteps appearing on a floor. It demonstrates how coordinates from a **camera view** can be mapped onto a **flat floor surface**, animated there, and then warped into a **projector output** so the animation appears correctly on the physical floor.

The demo shows how projection mapping works by visualizing three stages of the pipeline:

1. **Floor space (top-down view)** – where the footsteps are generated and animated.
2. **Projector output** – the warped image that would be sent to a projector.
3. **Camera space** – what the camera would see from an angled position.

The footsteps follow a generated walking trail and rotate to match the direction of movement, creating a simple walking animation.



# What You Can Tweak

A few parts of the demo are easy to modify:

### Floor Size

You can change the size of the simulated floor in `test_demo.py`:

```python
floor_w = 1000
floor_h = 1000
```

This controls the coordinate space where footsteps are drawn.

---

### Camera and Projector Corners

These define the **calibration** for the system.

```python
cam_pts = [...]
proj_pts = [...]
```

Adjusting these points changes how the floor is warped between camera view and projector output.

---

### Footstep Appearance

The footsteps come from the animated GIF:

```
white_foot.gif
```

You can replace this file with any other footstep animation.

You can also tweak things like:

* step spacing
* fade duration
* animation speed
* step offset

inside `projection_mapping.py`.

---

### Walking Path

The demo generates a random trail:

```python
trail = make_random_trail()
```

You can replace this with your own list of coordinates if you want footsteps to follow a specific path.



# How to Run It

### 1. Install dependencies

```bash
pip install opencv-python numpy pillow
```


### 2. Run the demo

```bash
python test_demo.py
```


### Controls

```
Q  → quit
R  → generate a new random walking trail
```

Three windows will appear showing the different stages of the projection mapping process.


# Files

```
projection_mapping.py   # main projection mapping pipeline
test_demo.py            # visualization demo
white_foot.gif          # animated footstep graphic
```


This project is meant as a **simple sandbox for experimenting with projection mapping concepts**, including perspective warping, coordinate transforms, and animated overlays.
