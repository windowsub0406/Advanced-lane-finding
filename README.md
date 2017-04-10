# Advanced Lane Finding Project  
  
  
<p align="center">
    <img src="images/main.jpg" width="480" alt="main_image" /><br>
    <b>result image(watch the full video below)</b><br>
</p>

## Introduction  
  
>This is **Advanced lane finding project** of Udacity's Self-Driving Car Engineering Nanodegree.
We already completed [lane finding project](https://github.com/windowsub0406/finding-lane-lines) in the first project. In that project, we could find lane lines and made robust algorighm for shadow and some of occlusion. It might be enough in the straight highway. But there are many curve lines in the road and that's why we need to detect curve lanes. In this project we'll find lane lines more specifically with computer vision.  
  
## Environment  
  
#### software  
  
>Windows 10(x64), Python 3.5, OpenCV 3.1.0

## Files
  
[`main.py`](main.py) : main code  
[`calibration.py`](calibration.py) : get calibration matrix  
[`threshold.py`](threshold.py) : sobel edge & hls color  
[`finding_lines.py`](finding_lines.py) : find & draw lane lines with sliding widow search  
[`finding_lines_w.py`](finding_lines_w.py) : find & draw lane lines with sliding window search using weighted average method (for the [`challenge_video`](challenge_video.mp4))


## The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---

## Camera Calibration

#### When a camera looks at 3D objects in the real world and transforms them into a 2D image, it's not perfect because of a distortion. And the distortion brings an erroneous information.(e.g. changed object shape, bent lane lines) So, we have to undo the distortion for getting useful data. 

The code for camera calibration step is contained in the [`calibration.py`](calibration.py).  

I compute the camera matrix(intrinsic parameters) and distortion coefficients using the `cv2.calibrateCamera()` function with 20 9*6 sized [chessboard images](camera_cal). And applied this distortion correction to the test image using the `cv2.undistort()` function.  
  
<p align="center">
    <img src="images/calibration.jpg" width="640" alt="calib_image" /><br>    
</p>
  
## Pipeline  
  
<p align="center">
    <img src="images/process.jpg" alt="process_image" /><br>
    <b>General Process</b><br>
</p>

If an image loaded, we immediately undo distortion of the image using calculated calibration information.
  
### 1. Crop Image  
  

<p align="center">
    <img src="images/crop_img.jpg" width="640" alt="crop" /><br>
</p>  

In image, a bonnet and background are not necessary to find lane lines. Therefore, I cropped the inconsequential parts.  
  
### 2. Lane Finding  
  
I used two approaches to find lane lines.  
a **Gradient** approach and a **Color** approach.
The code for lane finding step is contained in the [`threshold.py`](threshold.py).  

In gradient approach, I applied Sobel operator in the x, y directions. And calculated magnitude of the gradient in both the x and y directions and direction of the gradient. I used red channel of RGB instead of grayscaled image.  
And I combined them based on this code :  
  
`gradient_comb[((sobelx>1) & (mag_img>1) & (dir_img>1)) | ((sobelx>1) & (sobely>1))] = 255`  
    
<p align="center">
    <img src="images/gradient.jpg" width="640" alt="gradient" /><br>
</p>  
  

In Color approach, I used red channel of RGB Color space and H,L,S channel of HSV Color space. Red color(255,0,0) is included in white(255,255,255) and yellow(255,255,0) color. That's way I used it. Also I used HLS Color space because we could be robust in brightness.  
I combined them based on this code :  

`hls_comb[((s_img>1) & (l_img == 0)) | ((s_img==0) & (h_img>1) & (l_img>1)) | (R>1)] = 255`  
  
With this method, I could eliminate unnecessary shadow information.  

<p align="center">
    <img src="images/color.jpg" width="640" alt="color" /><br>
</p>  
  
  
This is combination of color and gradient thresholds.  

<p align="center">
    <img src="images/combination.jpg" width="640" alt="combination" /><br>
</p>  
  
  
### 3. Perspective Transform  
  
 We can assume the road is a flat plane. Pick 4 points of straight lane lines and apply perspective transform to the lines look straight. It is also called `Bird's eye view`.  
  
<p align="center">
    <img src="images/warp.jpg" width="640" alt="warp" /><br>
</p>  


### 4. Sliding Window Search  
  
The code for Sliding window search is contained in the [`finding_lines.py`](finding_lines.py) or [`finding_lines_w.py`](finding_lines.py).  

In the video, we could predict the position of lane lines by checking previous frame's information. But we need an other method for a first frame.  

In my code, if the frame is first frame or lost lane position, found first window position using histogram. Just accumulated non-zero pixels along the columns in the lower 2/3 of the image.  

<p align="center">
    <img src="images/histogram.jpg" width="320" alt="hist" /><br>
</p>  
  
  
In the course, we estimated curve line by using all non-zero pixels of windows. Non-zero piexels include **color information** and **gradient information** in bird's eyes view binary image. It works well in [`project_video`](project_video.mp4).  

**But it has a problem.**  
  
<p align="center">
    <img src="images/nonzero.jpg" width="480" alt="nonzero" /><br>
</p>  
  
This is one frame of [`challenge_video`](challenge_video.mp4).  
In this image, there are some cracks and dark trails near the lane lines. Let's check the result.  

If we fit curve lines with non-zero pixels, the result is here.
<p align="center">
    <img src="images/nonzero2.jpg" width="640" alt="nonzero2" /><br>
</p> 

As you can see, we couldn't detect exact lane positions. Because our gradient information have cracks information and it occurs error of position.  

So, I used **`weighted average`** method.
I put **0.8** weight value to color information and **0.2** to gradient information. And calculated x-average by using weighted average in the window. 
This is the result.  
<p align="center">
    <img src="images/weight.jpg" width="640" alt="weight" /><br>
</p>  
  

### 5. Road information  
  
<p align="center">
    <img src="images/road_info.jpg" width="480" alt="road_info" /><br>
</p>  
  
In my output video, I included some road informations.

#### Lane Info
* estimate lane status that is a straight line, or left/right curve. To decide this, I considered a radius of curvature and a curve direction.  
  
#### Curvature
* for calculating a radius of curvature in real world, I used U.S. regulations that require a minimum lane width of 3.7 meters. And assumed the lane's length is about 30m.  
  
#### Deviation
* Estimated current vehicle position by comparing image center with center of lane line.  

#### Mini road map
* The small mini map visualizes above information.

---

## Result  

>Project Video (Click for full HD video)
  
[![Video White](images/project_gif.gif?raw=true)](https://youtu.be/z0U_zPPL9cY)  
  
  
  
>Challenge Video (Click for full HD video)
  
[![Video White](images/challenge_result.gif?raw=true)](https://youtu.be/xri9c7xW1S4)

  
---
  
## Reflection  
  
I gave my best effort to succeed in challenge video. It wasn't easy. I have to change most of the parameters of project video. It means that the parameters strongly influenced by road status(bright or dark) or weather.  
To keep the deadline, I didn't try harder challenge video yet. It looks really hard but It could be a great challenge to me.  
