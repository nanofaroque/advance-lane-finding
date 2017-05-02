##Advanced Lane Finding Project 
###(Udacity Nanodegree Project 4)

---

<!-- **Advanced Lane Finding Project** -->

The following are the goals and steps of this project:

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("birds-eye view").
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Warp the detected lane boundaries back onto the original image.
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_result1.jpg "Result of Distortion Correction on Chessboard Image"
[image2]: ./output_images/undistort_result2.jpg "Result of Distortion Correction on Road Image"
[image3]: ./output_images/rgb_colourspace.png "Test Image Shown in RGB Colour Space"
[image4]: ./output_images/hls_colourspace.png "Test Image Shown in HLS Colour Space"

[image5]: ./output_images/rgb_white_threshold.png " "
[image6]: ./output_images/rgb_yellow_threshold.png " "
[image7]: ./output_images/hls_yellow_threshold.png " "
[image8]: ./output_images/warp_verify.png "Perspective Transform Output"
[image9]: ./output_images/lower_half_n_histogram.png "Lower Half Image and Histogram"
[image10]: ./output_images/plotlines_on_bin_img.jpg "Lines fitted on lane pixels"
[image11]: ./output_images/curvature_formula.png "Formula for Radius of Curvature"
[image12]: ./output_images/final_result.png "Lanes Projected on Original Image"

<!-- [video1]: ./output_images/rgb_yellow_threshold.png " " -->

#### This writeup explains the steps I followed in my implementation by following the [Rubric Points](https://review.udacity.com/#!/rubrics/571/view)

*The codes for steps on single image pipeline are is IPython notebook, [pipeline_session.ipynb](https://github.com/toluwajosh/CarND-Advanced-Lane-Lines/blob/master/pipeline_session.ipynb), while code for the implementation on project video is in [lane_line.py](https://github.com/toluwajosh/CarND-Advanced-Lane-Lines/blob/master/lane_line.py)*

---
<!-- ###Writeup / README
####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point. -->  
<!-- You're reading it! -->

###1.0 Camera Calibration

####Computation of camera calibration matrix and distortion coefficients given a set of chessboard images

Starting with camera calibration, see cell 2 for codes in this step. Here, I prepared 'object points' which are points in real world space (x, y, z) coordinates of the chessboard corners. Since we assume the chessboard is on a flat plane, then z=0 so we only consider (x, y). To do this, I went through each chessboard image in the image folder and use the opencv function cv2.findChessboardCorners() to find the corners of the chessboard. I then append these corners to the image points (imgpoints) array to keep the chessboard corners. I also replicated the object points and append to 'objpoints' array to correspond to the found image points.
After this, I used the outputs 'objpoints' and 'imgpoints' to compute the camera calibration matrix and distortion coefficients needed for correcting distortion on images by using the opencv cv2.calibrateCamera() function.
To be sure about the success of the calibration, I tested it on one of the chessboard images to correct its distortion by using the cv2.undistort() function. The figure below shows the image before distortion correction and after correction.x

![alt text][image1]

####Apply a distortion correction to raw images

I demonstrated the distortion correction on one of the test images. The result is shown below.

![alt text][image2]

---

###2.0 Image Thresholding

####Use of color transforms, and gradients to create a thresholded binary image.
<!-- ####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images) -->

Before applying image thresholding, I investigated how the image appears in various colour spaces to decide on which channels will be useful for this purpose. The figures below a test image shown in different colour spaces.

![alt text][image3]

![alt text][image4]

From observing and earlier trials, I realized the best way to detect lane lines is to use the colour information. Since lane lines are usually yellow and white, it will be effective to detect only white and yellow patches of road. I therefore decided to use colour thresholding. I also realized the RGB channel is more effective in detecting the white patches, since the luminosity is not a separate channel, and the HLS channel is effect for detecting the yellow colours, since only one channel represents the Hue values. The following table shows the values I used to threshold white and yellow in the respective colour channels.

| Colour 		| Threshold Values   										| 
|:-------------:|:--------------------------------------------------------:	| 
| RGB White     | Lower = {100, 100, 200}, Upper = {255, 255, 255}       	| 
| RGB Yellow 	| Lower = {225, 180, 0}, Upper = {255, 255, 170} 			|
| HLS Yellow 	| Lower = {20, 120, 80}, Upper = {45, 200, 255}     		|

The code for thresholding is contained in the threshold_colours() function of the 6th cell in the pipeline_sessions.ipnyb file.

See Images below for output of thresholding from each colour space

![alt text][image5]

![alt text][image6]

![alt text][image7]

---

###3.0 Perspective Transform

####Apply a perspective transform to rectify binary image ("birds-eye view")
<!-- Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image. -->

I loaded a test image with straight lines, to obtain points on the road plane. Points were manually picked from image with straight road lines. I also chose destination points to transform to a birds eye view. This operation is in the Perspective Transform section (3.0)　of the pipeline_sessions.ipynb file.

See below, the points I chose for source and destination:

<!-- The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

``` -->
<!-- This resulted in the following source and destination points: -->

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 270, 674      | 270, 674      | 
| 587, 455      | 270, 0      	|
| 694, 455     	| 1035, 0      	|
| 1035, 674     | 1035, 674     |

To verify the perspective transform, I drew line on points and transformed it to bird's eye view. Result is shown in figure below:

![alt text][image8]

---

###4.0 Fit Lane Lines

####Detect lane pixels and fit to find the lane boundary
<!-- Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial? -->

The following are steps I took in detecting lane line pixels and fitting lines to find lane boundary. Codes can be found in the same section in pipeline_sessions.ipynb file

I took a histogram along all columns in the lower half of the image to find portions of high contrast, which signifies lane lines.

![alt text][image9]

Next I implement a window search by finding the peak of the left and right halves of the histogram. These will be the starting point for the left and right lines. Then I create a window that searches from the bottom of the image to the top to find all non zero pixels within the window. This was done using the code snippet below



	# Step through the windows one by one
	for window in range(nwindows):
	    # Identify window boundaries in x and y (and right and left)
	    win_y_low = warped.shape[0] - (window+1)*window_height
	    win_y_high = warped.shape[0] - window*window_height
	    win_xleft_low = leftx_current - margin
	    win_xleft_high = leftx_current + margin
	    win_xright_low = rightx_current - margin
	    win_xright_high = rightx_current + margin

	    # Draw the windows on the visualization image
	    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 5) 
	    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 5) 

	    # Identify the nonzero pixels in x and y within the window
	    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
	    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
	    # Append these indices to the lists
	    left_lane_inds.append(good_left_inds)
	    right_lane_inds.append(good_right_inds)
	    # If you found > minpix pixels, recenter next window on their mean position
	    if len(good_left_inds) > minpix:
	        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
	    if len(good_right_inds) > minpix:        
	        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))



I then use these points to fit a 2nd order polynomial for both left and right lanes like shown below.
<!-- Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this: -->

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

![alt text][image10]

In subsequent frames, we only update the lane line pixels and do not need to do sliding window search again.

---

###5.0 Curvature and Vehicle Position

####Determine the curvature of the lane and vehicle position with respect to center
<!-- Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center. -->

I used the formula below to find the radius of curvature.

![alt text][image11]

A and B are the coefficients of the derivative of the second order polynomials used to find the fitted lines earlier. I calculated the radius of curvature at the point where y is maximum, which corresponds to the base of the image like thus;



	y_eval = np.max(ploty)
	left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
	right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])



I converted pixel values to meters, as in the measurement on the road using the following:



	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720.0 # meters per pixel in y dimension
	xm_per_pix = 3.7/700.0 # meters per pixel in x dimension

---

<!-- I did this in lines # through # in my code in `my_other_file.py` -->
###6.0 Lane Result

####Warp the detected lane boundaries back onto the original image
<!-- Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly. -->
<!-- I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image: -->
I warped the result back unto the original image using the inverse transformation matrix calculated earlier. The result is shown below:

![alt text][image12]

Next is to display radius of curvature and car offset from the center on the image. I did this in the final video

---

###Pipeline Result (video)

<!-- Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!). -->

Find in the link below, the final output of my pipeline for the project video

[Project Video Result](https://youtu.be/B16Fb0fPzi8)

---

###Discussion
#### Insights and observations from the project
<!-- 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust? -->

The most challenging part of this model is coming up with a lane line detection or thresholding algorithm that is robust for the project video and also good for the challenge video. I finally settled with using colour thresholding and none of line or brightness gradient since they are more affected by frame brightness and light conditions. The later can still be added to make pipeline more robust.

The final pipeline works well for the project video but fails in the challenge videos for places where there are very high brightness and where the road is very curved. The window search could be replaced with a better algorithm.

There are many places to improve, but I think a real system would rely on more than the monocular camera to analyse the lane curves.
