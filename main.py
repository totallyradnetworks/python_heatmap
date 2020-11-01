#parameters
#thDivider :the longer the video, the higher this value should be. to make each "movement" less important

import numpy as np
import cv2
import copy
import os
import time
#from matplotlib import pyplot as plt


def main():
    cap = cv2.VideoCapture(0) #"rtsp://admin:admin@192.168.88.97/media/video1") #"rtsp://admin:admin@192.168.88.97/media/video1"

    if not cap.isOpened():
        print("Error opening video stream or file")

    #done below in a cooler way
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = int(cap.get(cv2.CAP_PROP_FPS))


    video_out = cv2.VideoWriter('heatmap_video.avi', cv2.VideoWriter_fourcc(*'mjpg'), 10.0, (width, height))
                

    background_subtractor = cv2.createBackgroundSubtractorMOG2()

    first_iteration_indicator = 1

    i=0

    oldtime = time.time()
    

    startOf5min=time.time()
    startOf10min=time.time()
    startOf30min=time.time()

    fileCounter5min=0
    fileCounter10min=0
    fileCounter30min=0

    # voltes=0

    thDivider=0
    while(cap.isOpened()):
    
        # Capture frame-by-frame
        ret, img = cap.read()

        #save frame as .jpg file to directory ../frames at rate of 1 frame per second
        #delete oldest frame if number of files exceed 2000

        # voltes+=1
        # print(voltes)
        if first_iteration_indicator == 1:

            first_frame = copy.deepcopy(img)
            #height, width = frame.shape[:2]
            accum_image = np.zeros((height, width), np.uint8)
            first_iteration_indicator = 0

            video_frame = copy.deepcopy(img)
        else:

            filter = background_subtractor.apply(img)  # remove the background
            threshold = 2
            maxValue = 2

            blur = cv2.GaussianBlur(filter,(5,5),0)

            ret, th1 = cv2.threshold(blur, threshold, maxValue, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            if not os.path.exists('frames'):
                os.makedirs('frames')

            if not os.path.exists('heatmap5min'):
                os.makedirs('heatmap5min')
            if not os.path.exists('heatmap10min'):
                os.makedirs('heatmap10min')
            if not os.path.exists('heatmap30min'):
                os.makedirs('heatmap30min')


            if time.time() - oldtime > 1:
                #print("it's been a second")
                name = "./frames/frame%d.jpg" % i
                cv2.imwrite(name, video_frame)

                oldtime = time.time()

                i+=1
                if (i>200): #circular, up to 200 images
                    i=0


            
            thDivider+=1
            if (thDivider==FPS): #thus, we will get one frame each second. make 2*FPS if we want each two seconds for instance
                name = "./heatmap5min/thframe%d.jpg" % fileCounter5min
                cv2.imwrite(name, th1)
                accum_image = cv2.add(accum_image, th1)
                thDivider = 0
                fileCounter5min += 1
                if (time.time() - startOf5min> 60 * 5) :  #circular, up to 5 mins of images
                    fileCounter5min=0
                
            
            color_image_video = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
            video_frame = cv2.addWeighted(img, 0.5, color_image_video, 0.5, 0) #0.7


            video_out.write(video_frame)
            
            # Display the resulting frame
            cv2.imshow('frame',video_frame)

    #plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    #  plt.title(titles[i])


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        



    generateHeatmap(height, width, 'heatmap5min')


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



  # Function that generates a heatmap by accumulating ALL of the heatimages of a given folder
def generateHeatmap(height, width, foldername):
    accum_image_xmin = np.zeros((height, width,3), np.uint8)
    for filename in os.listdir(foldername):
        heat_frame= cv2.imread('./' + foldername + '/'+ filename)
        accum_image_xmin = cv2.add(accum_image_xmin, heat_frame)

    accum_image_xmin_color = cv2.applyColorMap(accum_image_xmin, cv2.COLORMAP_HOT)
    cv2.imwrite('./' + foldername + '.jpg', accum_image_xmin_color)



        
if __name__ == "__main__":
    main()

