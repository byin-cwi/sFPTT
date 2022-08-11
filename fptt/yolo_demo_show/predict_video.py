#-----------------------------------------------------------------------#
#   predict.py combines single image prediction, camera detection, 
#   FPS testing and directory traversal detection into a single py file,
#   which can be modified by specifying "mode".
#-----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

# from yolo_video1 import YOLO
from yolo_video2 import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    #----------------------------------------------------------------------------------------------------------#
    #   "mode" is used to specify the mode of the test.：
    #   'predict' means a single image prediction. If you want to modify the prediction process, such as saving images, intercepting objects, etc., you can first read the detailed comments below.
    #   'video' means video detection, you can call the camera or video for detection, check the comments below for details.
    #   'fps' means test fps, the image used is street.jpg inside img, check the comments below for details.
    #   'dir_predict' means traverse the folder to detect and save. Default traverses img folder and saves img_out folder, see comments below for details.
    #----------------------------------------------------------------------------------------------------------#
    # mode = "predict"
    mode = "video"
    #----------------------------------------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = "./img/testv4.1.mp4"
    video_fps       = 60.0
    #-------------------------------------------------------------------------#
    #   test_interval is used to specify the number of times the image will be detected when measuring fps.
    #   Theoretically, the larger the test_interval, the more accurate the fps.
    #-------------------------------------------------------------------------#
    test_interval   = 100
    #-------------------------------------------------------------------------#
    #   dir_origin_path specifies the folder path of the image to be detected
    #   dir_save_path specifies the path where the detected images are saved
    #   dir_origin_path and dir_save_path are only valid when mode='dir_predict'
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"

    if mode == "predict":
        '''
        1、If you want to save the detected image, you can use r_image.save("img.jpg") to save it and modify it directly in predict.py. 
        2、If you want to get the coordinates of the prediction box, you can enter the yolo.detect_image function and read the four values of top, left, bottom and right in the drawing section.
        3、If you want to use the prediction box to intercept the next target, you can enter the yolo.detect_image function, and use the obtained top, left, bottom, right values in the drawing section.
        on the original map using the matrix to intercept the way.
        4. If you want to write additional words on the predicted image, such as the number of specific targets detected, you can enter the yolo.detect_image function and make a judgment on the predicted_class in the drawing section.
        For example, judge if predicted_class == 'car': that is, you can determine whether the current target is a car or not, and then just record the number. Use draw.text to write.
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            # out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
            # size    = (int(480), int(640))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        print('capture video')
        if not ref:
            raise ValueError("Failure to read the camera (video) correctly, please note whether the camera is installed correctly (whether the video path is filled in correctly).")

        fps = 0.0
        while(True):
            t1 = time.time()
            # Read a frame
            ref, frame = capture.read()
            if not ref:
                break
            # BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # to Image
            frame = Image.fromarray(np.uint8(frame))
            # run detection
            frame = np.array(yolo.detect_image(frame))
            # RGBtoBGR fit opencv format
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame,size)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
            # if cv2.waitKey(1) & 0xFF == ord('q'):
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
                
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
