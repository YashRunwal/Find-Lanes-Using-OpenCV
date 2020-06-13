import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import PIL
import os


def get_undistorted_data():
    
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = glob.glob('camera_cal/*.jpg')
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
         
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (7,6), corners2, ret)
            cv2.waitKey(500)
            
    cv2.destroyAllWindows()
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    dist_pickle = {}
    dist_pickle['mtx']  = mtx
    dist_pickle['dist'] = dist
    
    pickle.dump(dist_pickle, open('camera_cal/cal_pickle.p', 'wb'))
    
    return dist_pickle
    

    
# Undistort Images
def undistort(img, cal_dir='camera_cal/cal_pickle.p'):
    
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
        mtx = file['mtx']
    dist = file['dist']
    
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    return undist


def apply_sobel(img, s_thresh=(100, 255), sx_thresh=(15, 255)):
    
    img = undistort(img)
    
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    return combined_binary



def perspective_warp(img,
                     dst_size=(1280,720),
                     src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
                     dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    # Arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    
    return warped



def inverse_perspective_warp(img,
                     dst_size=(1280,720),
                     dst=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
                     src=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    # Arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    
    return warped



def get_hist(img):
    
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
        
    return hist
    


left_a, left_b, left_c = [],[],[]
right_a, right_b, right_c = [],[],[]

def sliding_window(img, nwindows=9, margin=150, minpix = 1, draw_windows=True):
    global left_a, left_b, left_c,right_a, right_b, right_c 
    left_fit_= np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img))*255

    hist = get_hist(img)

    # find peaks of left and right halves
    midpoint = int(hist.shape[0]/2)
    leftx_base = np.argmax(hist[:midpoint])
    rightx_base = np.argmax(hist[midpoint:]) + midpoint
    
    
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
       
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        if draw_windows == True:
        
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (100,255,255), 3) 
        
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (100,255,255), 3) 
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])
    
    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])
    
    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])
    
    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
    right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]
    
    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty



def get_curve(img, leftx, rightx):
    
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    y_eval = np.max(ploty)
    ym_per_pix = 30.5/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/720 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    car_pos = img.shape[1]/2
    l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) /2
    center = (car_pos - lane_center_position) * xm_per_pix / 10
    
    # Now our radius of curvature is in meters
    
    return (left_curverad, right_curverad, center)


def draw_lanes(img, left_fit, right_fit):

    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_img = np.zeros_like(img)
    
    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))
    
    cv2.fillPoly(color_img, np.int_(points), (255,255, 0))
    inv_perspective = inverse_perspective_warp(color_img)
    inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)

    return inv_perspective



def image_pipeline():
    
    # Read Image
    img = cv2.imread('test_images/curved_lane.jpg')
    
    # Change the color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply Sobel Filtering
    dst = apply_sobel(img)
    
    # Applying Perspective Warp
    dst = perspective_warp(dst, dst_size=(1280,720))
    
    out_img, curves, lanes, ploty = sliding_window(dst)
#    print(np.asarray(curves).shape)
    curverad=get_curve(img, curves[0],curves[1])
#    print(curverad)
    img_ = draw_lanes(img, curves[0], curves[1])
    plt.imshow(img_, cmap='hsv')
    
    return img, dst, curves, img_, ploty, out_img


    

def show_plots(frame, dst, curves, img_, ploty, out_img):
    
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(100, 20))
    #f.tight_layout()
   
    ax1.imshow(frame)
    ax1.set_title('Original', fontsize=10)
    
    ax2.imshow(dst)
    ax2.set_title('Filter+Perspective Tform', fontsize=10)
    
    ax3.imshow(out_img)
    ax3.plot(curves[0], ploty, color='yellow', linewidth=30)
    ax3.plot(curves[1], ploty, color='yellow', linewidth=30)
    ax3.set_title('Sliding window+Curve Fit', fontsize=10)
    
    ax4.imshow(img_)
    ax4.set_title('Overlay Lanes', fontsize=10)
    
    
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    
    
def videos_pipeline():
    
    cap = cv2.VideoCapture("Videos/Video1.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('final_lane_finding_output.mp4', fourcc, 20.0, size)
    
    while(cap.isOpened()):
        
        _, frame = cap.read()
        
        # Change the color space
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply Sobel Filtering
        dst = apply_sobel(frame)
        
        # Applying Perspective Warp
        dst = perspective_warp(dst, dst_size=(1280,720))
        
        # Sliding Window Search
        out_frame, curves, lanes, ploty = sliding_window(dst)

        curverad=get_curve(frame, curves[0],curves[1])

        # Find Curve and draw lane
        img_ = draw_lanes(frame, curves[0], curves[1])

        # Start the recording
        cv2.imshow('Recording...', frame)
        
        # Combine the image to show the frame
        combined_image = cv2.addWeighted(frame, 0.8, img_, 1, 1)
        cv2.imshow("Resulting Image", combined_image)
        
        # Save the outpput
        out.write(combined_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    
        

if __name__ == '__main__':

    videos_pipeline()
    