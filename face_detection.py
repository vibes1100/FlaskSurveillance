def face_detection():
    # -- coding: utf-8 --
    """
    Created on Sat Jun 20 15:43:46 2020

    @author: ANISH
    """

    
    import cv2
#from keras_vggface.vggface import VGGFace


    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    #eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # Read the input image
    #img = cv2.imread('test.png')
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        _, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        print(len(faces))
        
        i = 0 
        for (x, y , w ,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 0), 4)
            r = max(w, h) 
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg = img[ny:ny+nr, nx:nx+nr]
            lastimg = cv2.resize(faceimg, (36, 36))
            i += 1
            
            cv2.imshow('image', lastimg)
            
        
        #fgmask = fgbg.apply(img)
        # Display the output
            
        cv2.imshow('img', img)
        ##########################################################
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
        #######################################################
        #cv2.imshow('mask',fgmask)
        
        k = cv2.waitKey(30) & 0xFF
        
        if k == 27 : 
            break

    cap.release()
    cap.destroyAllWindows()
    # import cv2 
    # import requests
    # import argparse 
    # import os
    
    # # parse arguments
    # parser = argparse.ArgumentParser(description='YOLO Face Detection')
    # parser.add_argument('--src', action='store', default=0, nargs='?', help='Set video source; default is usb webcam')
    # parser.add_argument('--w', action='store', default=320, nargs='?', help='Set video width')
    # parser.add_argument('--h', action='store', default=240, nargs='?', help='Set video height')
    # args = parser.parse_args()
    
    # # face detection endpoint (deepsight sdk runs as http service on port 5000)
    # face_api = "http://127.0.0.1:5000/inferImage?detector=yolo";
    
    # # capture frames from a camera
    # cap = cv2.VideoCapture(args.src)
    
    # # loop runs if capturing has been initialized.
    # while 1:    
            
    #     # reads frames from a camera
    #     ret, img = cap.read() 
    #     img = cv2.resize(img, (int(args.w),int(args.h)))    
    #     r, imgbuf = cv2.imencode(".bmp", img)    
    #     image = {'pic':bytearray(imgbuf)}
        
    #     r = requests.post(face_api, files=image)
    #     result = r.json()   
        
    #     if len(result) > 1:
    #         faces = result[:-1]
    #         for face in faces:
    #             rect = [face[i] for i in ['faceRectangle']][0]
    #             x,y,w,h, confidence = [rect[i] for i in ['left', 'top', 'width', 'height', 'confidence']]
    #             # discard if confidence is too low
    #             if confidence < 0.6:
    #                 continue
                    
    #             cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255),4,8) 
    
        
    #     cv2.imshow('YOLO Face detection',img)
    #     return img
    
    #     # Wait for Esc key to stop
    #     if cv2.waitKey(1) == 27:
    #         break
    
    # # Close the window
    # cap.release()
    
    # # De-allocate any associated memory usage
    # cv2.destroyAllWindows()
