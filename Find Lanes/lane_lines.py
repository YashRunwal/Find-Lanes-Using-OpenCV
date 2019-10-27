''' Finding Lanes '''
import cv2
import numpy as np


# Create Constants
THRESHOLD = 100
GAMMA = 1



def region_of_interest(image):
    height = image.shape[0] # along the vertical
    ''' the numbers in polygons are obtained by first using matplotlib and noting down the required coordinates'''
    polygons = np.array([
            [(200, height), (1100, height), (550, 250)]
            ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image



def optimize_lines_by_average_slope_intercept(image, detect_lines):
    left_fit  = []
    right_fit = []
    for line in detect_lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
        
        if left_fit:
            left_fit_average = np.average(left_fit, axis=0)
            print(left_fit_average, 'left')
            left_line = make_coordinates(image, left_fit_average)
            
        if right_fit:
            right_fit_average = np.average(right_fit, axis=0)
            print(right_fit_average, 'right')
            right_line = make_coordinates(image, right_fit_average)
        
    return np.array([left_line, right_line])
        
        
        
def make_coordinates(image, line_parameters):
    global slope, intercept
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    
    return np.array([x1, y1, x2, y2])
    
    

def display_lines(image, detect_lines):
    global line_image
    line_image = np.zeros_like(image)
    if detect_lines is not None:
        for line in detect_lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)  
    return line_image


'''
def display_image():
    global lane_image, final_image
    
    final_image = cv2.imread(str('test_image.jpg'))
    lane_image = np.copy(final_image)
    canny_image = gray_scale_canny(lane_image)
    required_image = region_of_interest(canny_image)

    detect_lines = cv2.HoughLinesP(required_image, 2, np.pi/180, 
                                  THRESHOLD, np.array([]), 
                                  minLineLength=40, maxLineGap=5)
    
    averaged_lines = optimize_lines_by_average_slope_intercept(lane_image, detect_lines)
    line_image = display_lines(lane_image, averaged_lines)
    combined_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, GAMMA)
    cv2.imshow("Resulting Image", combined_image)
    cv2.waitKey(0)
''' 

  
def gray_scale_canny(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5,5), 0)
    canny_image = cv2.Canny(blurred_image, 50, 150)
    return canny_image



def capture_video():
    cap = cv2.VideoCapture("test2.mp4")
    while(cap.isOpened()):
        _, frame = cap.read()
        canny_image = gray_scale_canny(frame)
        required_image = region_of_interest(canny_image)

        detect_lines = cv2.HoughLinesP(required_image, 2, np.pi/180, 
                                  THRESHOLD, np.array([]), 
                                  minLineLength=40, maxLineGap=5)
    
        averaged_lines = optimize_lines_by_average_slope_intercept(frame, detect_lines)
        line_image = display_lines(frame, averaged_lines)
        combined_image = cv2.addWeighted(frame, 0.8, line_image, 1, GAMMA)
        cv2.imshow("Resulting Image", combined_image)
        if cv2.waitKey(1) == ord('q'):
            break# 1 millisecond
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    capture_video()

