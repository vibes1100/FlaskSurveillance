# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import time
import cv2
import os
# construct the argument parse and parse the arguments
cv2.ocl.setUseOpenCL(False)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", 
    help="path to input image")
ap.add_argument("-y", "--yolo", default="C:\\Users\\Vaibs\\Downloads\\VITSem6\\IoT\\Project\\yolo-coco",
    help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
# print(labelsPath)
LABELS = open(labelsPath).read().strip().split("\n")
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")
    # derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# cap = cv2.VideoCapture(0)
class VideoCamera(object):
    
    def __init__(self, camsrc='cam1'):
        cv2.ocl.setUseOpenCL(False)
        #capturing video
        if camsrc=='cam1':
            self.video = cv2.VideoCapture(0)
        elif camsrc=='cam2':
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            # self.video =cv2.VideoCapture("rtsp://192.168.0.177:8080/h264_ulaw.sdp")
            # self.video =cv2.VideoCapture("rtsp://192.168.0.177:5554/camera")
            self.video =cv2.VideoCapture("http://192.168.0.177:8080/video")
            for i in range(100):
                self.video.grab()
    
    def __del__(self):
        #releasing camera
        self.video.release()
    def read_frame(self):
        # load our input image and grab its spatial dimensions
        # while cap.isOpened():
        _, image = self.video.read()
        (H, W) = image.shape[:2]
        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()


        # show timing information on YOLO
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))
        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
            args["threshold"])
        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
        # show the output image
        # for i in range(4):
        #     self.video.grab()
        # cv2.imshow("Image", image)
        key = cv2.waitKey(1)
        # if key == ord('q'):
        #     break
        # elif key == ord('w'):
        #     cap.open("https://192.168.0.177:8080/video")
        #     frame=cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
        # cap.release()
        # cv2.destroyAllWindows()



    # from imutils.video import VideoStream
    # from imutils.video import FPS
    # import numpy as np
    # import argparse
    # import imutils
    # import time
    # import cv2

    # ap = argparse.ArgumentParser()
    # ap.add_argument("-p", "--prototxt", required=True,
    # 	help="path to Caffe 'deploy' prototxt file")
    # ap.add_argument("-m", "--model", required=True,
    # 	help="path to Caffe pre-trained model")
    # ap.add_argument("-c", "--confidence", type=float, default=0.2,
    # 	help="minimum probability to filter weak detections")
    # args = vars(ap.parse_args())

    # CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    # 	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    # 	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    # 	"sofa", "train", "tvmonitor"]
    # COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # print("[INFO] loading model...")
    # net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    # # initialize the video stream, allow the cammera sensor to warmup,
    # # and initialize the FPS counter
    # print("[INFO] starting video stream...")
    # vs = VideoStream(src=0).start()
    # time.sleep(2.0)
    # fps = FPS().start()

    # while True:
    # 	# grab the frame from the threaded video stream and resize it
    # 	# to have a maximum width of 400 pixels
    # 	frame = vs.read()
    # 	frame = imutils.resize(frame, width=400)
    # 	# grab the frame dimensions and convert it to a blob
    # 	(h, w) = frame.shape[:2]
    # 	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
    # 		0.007843, (300, 300), 127.5)
    # 	# pass the blob through the network and obtain the detections and
    # 	# predictions
    # 	net.setInput(blob)
    # 	detections = net.forward()
    #     # loop over the detections
    # 	for i in np.arange(0, detections.shape[2]):
    # 		# extract the confidence (i.e., probability) associated with
    # 		# the prediction
    # 		confidence = detections[0, 0, i, 2]
    # 		# filter out weak detections by ensuring the `confidence` is
    # 		# greater than the minimum confidence
    # 		if confidence > args["confidence"]:
    # 			# extract the index of the class label from the
    # 			# `detections`, then compute the (x, y)-coordinates of
    # 			# the bounding box for the object
    # 			idx = int(detections[0, 0, i, 1])
    # 			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    # 			(startX, startY, endX, endY) = box.astype("int")
    # 			# draw the prediction on the frame
    # 			label = "{}: {:.2f}%".format(CLASSES[idx],
    # 				confidence * 100)
    # 			cv2.rectangle(frame, (startX, startY), (endX, endY),
    # 				COLORS[idx], 2)
    # 			y = startY - 15 if startY - 15 > 15 else startY + 15
    # 			cv2.putText(frame, label, (startX, y),
    # 				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    #             # show the output frame
    # 	cv2.imshow("Frame", frame)
    # 	key = cv2.waitKey(1) & 0xFF
    # 	# if the `q` key was pressed, break from the loop
    # 	if key == ord("q"):
    # 		break
    # 	# update the FPS counter
    # 	fps.update()
    # fps.stop()
    # print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # # do a bit of cleanup
    # cv2.destroyAllWindows()
    # vs.stop()