import tempfile
import streamlit as st
import mediapipe as mp
import cv2
from align_custom import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
from tf_graph import FaceRecGraph
import sys
import json
import numpy as np
import sqlite3
import time
from cvzone.PoseModule import PoseDetector
import urllib.request
import simplejpeg
import cvzone

TIMEOUT = 10  # 10 seconds

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
global FRGraph, MTCNNGraph, aligner, extract_feature, face_detect
FRGraph = FaceRecGraph()
MTCNNGraph = FaceRecGraph()
aligner = AlignCustom()
extract_feature = FaceFeature(FRGraph)
face_detect = MTCNNDetect(MTCNNGraph, scale_factor=2)  # scale_factor, rescales image for faster detection
DEMO_VIDEO = 'demo.mp4'
DEMO_IMAGE = 'demo.jpg'

icon = "Logo.ico"
# st.title('Secure.AI Corporate Dashboard')
st.set_page_config("Secure.AI", page_icon=None, layout="centered", initial_sidebar_state="auto")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 200px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 200px;
        margin-left: -200px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Secure.AI')
st.sidebar.subheader('Parameters')


def findPeople(features_arr, positions, thres=0.4, percent_thres=50):
    '''
    :param features_arr: a list of 128d Features of all faces on screen
    :param positions: a list of face position types of all faces on screen
    :param thres: distance threshold
    :return: person name and percentage
    '''
    connection = sqlite3.connect("SD", timeout=10)
    crsr = connection.cursor()
    f = open('./facerec_128D.txt', 'r')
    data_set = json.loads(f.read())
    returnRes = []
    for (i, features_128D) in enumerate(features_arr):
        result = "NA"
        smallest = sys.maxsize
        for person in data_set.keys():
            person_data = data_set[person][positions[i]]
            for data in person_data:
                distance = np.sqrt(np.sum(np.square(data - features_128D)))
                if distance < smallest:
                    smallest = distance;
                    result = person;

        percentage = min(100, 100 * thres / smallest)
        if percentage <= percent_thres:
            result = "NA"
        returnRes.append(result)
        now = ''
     #   crsr.execute("INSERT INTO SD VALUES (?,?);", (result))
      #  connection.commit()
    return returnRes


@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


app_mode = st.sidebar.selectbox('Processing Mode',
                                ['MATRIX']
                                )

if app_mode == 'MATRIX':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Webcam')
    rtsp_stream = st.sidebar.text_input("Input Camera Stream")
    #rtsp = st.sidebar.button('Use RTSP')
        
    

    global AFRS, GBAS
    AFRS = st.sidebar.checkbox("Facial Recognition", True)
    GBAS = st.sidebar.checkbox("Gesture based Alert")
    #OD = st.sidebar.checkbox("Object Detection")
    #VD = st.sidebar.checkbox("Vehicle Detection")
    #if record:
        #st.sidebar.checkbox("Record Stream in Server", value=True)
        #rtsp = ""
        #rtsp_add = st.sidebar.text_input("Add Camera Stream", "rtsp://")
        #add_Stream = st.sidebar.button('RTSP Stream +1')

    st.sidebar.markdown('---')
    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    # max faces

    st.sidebar.markdown('---')

    st.markdown(' ## Output')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)

        elif rtsp_stream:
            vid = cv2.VideoCapture(rtsp_stream)

        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO

    
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    # codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    #out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)
    fps = fps_input
    i = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    kpi1, kpi2, kpi3 = st.beta_columns(3)

    with kpi1:
        st.markdown("**FrameRate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Detected Faces**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**People in Frame**")
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)
    detector = PoseDetector()
    count = 0
    dir = 0
    pTime = 0

    while True:
        #        img_resp = urllib.request.urlopen(url)
        #       imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        #      frame =  simplejpeg.encode_jpeg(imgnp, -1)
        #     frame = simplejpeg.decode_jpeg(imgnp, colorspace='bgr', fastdct=False)

        _, frame = vid.read()
        hf, wf, cf = frame.shape
        hb, wb, cb = frame.shape
        img = detector.findPose(frame, False)
        lmList = detector.findPosition(frame, False)
        # u can certainly add a roi here but for the sake of a demo i'll just leave it as simple as this
        rects, landmarks = face_detect.detect_face(frame, 30);  # min face size is set to 80x80
        aligns = []
        positions = []

        for (i, rect) in enumerate(rects):
            aligned_face, face_pos = aligner.align(160, frame, landmarks[:, i])
            if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
                aligns.append(aligned_face)
                positions.append(face_pos)
            else:
                print("Align face failed")  # log

        if AFRS:
            if len(aligns) > 0:
                features_arr = extract_feature.get_features(aligns)
                recog_data = findPeople(features_arr, positions)
                for (i, rect) in enumerate(rects):
                    cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]),
                                  (255, 0, 0))  # draw bounding box for the face
                    # Emergency Detection Here
                    if GBAS:
                        if len(recog_data) != 0:
                            # Right Arm
                            angle = detector.findAngle(img, 12, 14, 16)
                            # # Left Arm
                            # angle = detector.findAngle(img, 11, 13, 15,False)
                            per = np.interp(angle, (210, 310), (0, 100))

                            if per == 100:

                                if dir == 0:
                                    count += 0.5
                                    dir = 1
                            if per == 0:

                                if dir == 1:
                                    count += 0.5
                                    dir = 0
                            print(count)

                        #imgResult = cvzone.overlayPNG(frame, frame, [0, hb - hf])
                        # _, imgResult = fpsReader.update(imgResult)

                        if count >= 5:
                            cv2.putText(frame, " Help!" + '\n' + recog_data[i][0] + " - " + str(recog_data[i][1]) + "",
                                        (rect[0], rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)



                cv2.putText(frame, recog_data[i][0] + " - " + str(recog_data[i][1]) + "",(rect[0], rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
               # currTime = time.time()
                fps = fps
                #prevTime = currTime
               # if record:
                    # st.checkbox("Recording", value=True)
                #    out.write(frame)
                # Dashboard
                kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
                kpi2_text.write(f"<h1 style='text-align: center; color: red;'></h1>", unsafe_allow_html=True)
                kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{recog_data[i][0]}</h1>",
                                unsafe_allow_html=True)


                frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
                
                #frame = image_resize(image=frame, width=160, height= 120)
                stframe.image(frame, channels='BGR', use_column_width=True)

    st.text('Video Processed')

    #output_video = open('output1.mp4', 'rb')
    #out_bytes = output_video.read()

    st.video(out_bytes)

    vid.release()
    out.release()
