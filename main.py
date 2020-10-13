import numpy as np
import cv2
import copy


cap = cv2.VideoCapture("rtsp://admin:admin@192.168.88.97/media/video1") #"rtsp://admin:admin@192.168.88.97/media/video1"

if not cap.isOpened():
	print("Error opening video stream or file")

#done below in a cooler way
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_out = cv2.VideoWriter('heatmap_video.avi', cv2.VideoWriter_fourcc(*'mjpg'), 10.0, (width, height))
            

background_subtractor = cv2.createBackgroundSubtractorMOG2()

first_iteration_indicator = 1

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #save frame as .jpg file to directory ../frames at rate of 1 frame per second
    #delete oldest frame if number of files exceed 2000

    if first_iteration_indicator == 1:

        first_frame = copy.deepcopy(frame)
        #height, width = frame.shape[:2]
        accum_image = np.zeros((height, width), np.uint8)
        first_iteration_indicator = 0
    else:

        filter = background_subtractor.apply(frame)  # remove the background
        threshold = 2
        maxValue = 2
        ret, th1 = cv2.threshold(filter, threshold, maxValue, cv2.THRESH_BINARY)

        accum_image = cv2.add(accum_image, th1)

        color_image_video = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)

        video_frame = cv2.addWeighted(frame, 0.5, color_image_video, 0.5, 0) #0.7

        #name = "./frames/frame%d.jpg" % i
        #cv2.imwrite(name, video_frame)
        
        video_out.write(video_frame)
        
        # Display the resulting frame
        cv2.imshow('frame',video_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Function that generates a heatmap of most recent 5 minutes of frames

# Function that generates a heatmap of most recent 10 minutes of frames

# Function that generates a heatmap of most recent 30 minutes of frames


# make_video('./frames/', './output.avi')

color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
result_overlay = cv2.addWeighted(first_frame, 0.5, color_image, 0.5, 0)

# save the final heatmap
cv2.imwrite('diff-overlay.jpg', result_overlay)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()