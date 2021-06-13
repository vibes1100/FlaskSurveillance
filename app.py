from flask import Flask,request,jsonify, send_file, Response, render_template, redirect
import face_detection3_copy
from face_detection3_copy import VideoCamera
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

count = 0

@app.route('/',methods=['GET','POST'])
def index():
    # d={}
    # d['searched']=face_detection.face_detection()
    # print (d)
    # data= face_detection.face_detection()
    # img = Image.fromarray(data.astype('uint8'))
    # file_object = io.BytesIO()
    # img.save(file_object, 'PNG')
    # file_object.seek(0)

    # return (send_file(file_object, mimetype='image/PNG'), text)
    vidsrcs = [
        {'id': 'cam1', 'disp': 'Laptop Webcam'},
        {'id': 'cam2', 'disp': 'Phone Camera'}
    ]
    if request.method == 'POST':
        camsrc = request.form['camsrc']
        return render_template('index.html',
                            vidsrcs=vidsrcs,
                            camsrc=camsrc)
    else:
        return render_template('index.html',
                            vidsrcs=vidsrcs,
                            camsrc="cam1")

def gen(camera):
    global count
    while True:
        #get camera frame
        frame, counter= camera.read_frame()
        count = counter
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed/<camsrc>')
def video_feed(camsrc):
    return Response(gen(VideoCamera(camsrc)), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/countppl')
def countppl():
    global count
    return str(count)

if __name__ == '__main__':
    app.run(debug=True)