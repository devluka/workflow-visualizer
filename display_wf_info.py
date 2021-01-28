##############################################################################################
# Usage: 
#     > python display_wf_info.py [video] [workflow_log_file] [projection_matrix] [map_tif_file]
#################################################################################################

import sys
import numpy as np
import cv2
import pandas as pd
from osgeo import gdal


def GPS2Pixel(lat, lon, InvProjMat, adfGeoTransform):
    x = (lon - adfGeoTransform[0]) / adfGeoTransform[1]
    y = (lat - adfGeoTransform[3]) / adfGeoTransform[5]

    mapPixels = np.array([[[x, y]]], dtype='float32')
    imgPixels = cv2.perspectiveTransform(mapPixels, InvProjMat)

    #print("At GPS2Pixel: ("+str(lat)+","+str(lon)+")->("+str(imgPixels[0][0][0])+","+str(imgPixels[0][0][1])+")")
    return int(imgPixels[0][0][0]), int(imgPixels[0][0][1])

def Pixel2GPS(img_x, img_y, ProjMat, adfGeoTransform):
    imgPixel = np.array([[[img_x,img_y]]], dtype='float32')

    mapPixel = cv2.perspectiveTransform(imgPixel, ProjMat);
    x = mapPixel[0][0][0]
    y = mapPixel[0][0][1]

    lat = adfGeoTransform[4] * x + adfGeoTransform[5] * y + adfGeoTransform[3];
    lon = adfGeoTransform[1] * x + adfGeoTransform[2] * y + adfGeoTransform[0];

    #print("At Pixel2GPS: ("+str(img_x)+","+str(img_y)+")->("+str(lat)+","+str(lon)+")")
    return lat,lon

def generate_transformation_da(ProjMat_file,tif_file):

    with open(ProjMat_file) as textFile:
        lines = [line.split() for line in textFile]

    ProjMat = np.array(lines, dtype = 'float32')
    InvProjMat = np.linalg.inv(ProjMat)

    try:
        ds=gdal.Open(tif_file)
    except:
        print("Print error")

    adfGeoTransform=ds.GetGeoTransform()

    return ProjMat, InvProjMat, adfGeoTransform

def display_boxes(cap, out, df, TPdf, ProjMat, InvProjMat, adfGeoTransform):

    # Colour codes of identified objects
    # 0: (0,127,255)   Person (Orange)
    # 1: (255,0,0)     Car (Blue)
    # 2: (51,255,51)   Truck (light Green)
    # 3: (0,153,0)     Bus (Dark Green)
    # 4: (255,229,204) Motor (Light Blue)
    # 5: (204,0,102)   Bike (Dark Purple)
    # 6: (255,153,204) Rider (Light Purple)
    # 7: ()            Traffic light (colour not defined)
    # 8: ()            Traffic sign (colour not defined)
    # 9: (160,160,160) Train (Grey)
    color_boxes = { '0': (0,127,255), '1': (255,0,0), '2': (51,255,51), '3': (0,153,0), '4': (255,229,204), '5': (204,0,102), '6': (255,153,204), '9': (160,160,160) }

    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:

            # extract rows according to the current video frame number
            df_current = df.loc[df['frame'] == current_frame]
            TPdf_current = TPdf.loc[TPdf['frame'] == current_frame]

            # Process the workflow log file 
            for _, row in df_current.iterrows():

                # extract x, y, w, h from dataframe
                x, y, w, h, lon, lat = int(row['x']), int(row['y']), int(row['w']), int(row['h']), float(row['lon']), float(row['lat'])
                label, cat = row['obj_id'], row['category']
                color = color_boxes.get(str(cat), (0,0,0))

                print(" ")
                print("Processing frame:" + str(current_frame) + " obj_id:" + str(label) + " cat:" + str(cat) + " x:" + str(x) + " y:" + str(y) + " w:" + str(w) + " h:" + str(h) + " lat:" + str(lat) + " lon:" + str(lon))

                # GPS position is considered to be at the center of the bounding box
                tmp_lat,tmp_lon = Pixel2GPS( x + (w/2), y + (h/2), ProjMat, adfGeoTransform)
                frame_pixels = GPS2Pixel(lat,lon, InvProjMat, adfGeoTransform)

                cv2.rectangle(frame, (x, y), (w + x, y + h), color, 3)
                cv2.rectangle(frame, (x, y - 25), (x + 12 * len(label), y ), color, -1)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, .6, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.circle(frame,frame_pixels,10, color, 10)
                cv2.putText(frame, label, frame_pixels, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, .6, (255, 255, 255), 2, cv2.LINE_AA)


            # Process the TP log file 
            for _, row in TPdf_current.iterrows():

                label = row['obj_id']
                TPlat, TPlon, TPts = row['TPlat'].split(','), row['TPlon'].split(','), row['TPts'].split(',')
                
                print("TP_info: Obj_id:" + str(label) + " lat:" + str(TPlat) + " lon:" + str(TPlon))

                for i in range(len(TPlat)-1):

                    TPframe_pixels1 = GPS2Pixel(float(TPlat[i]),float(TPlon[i]), InvProjMat, adfGeoTransform) 
                    TPframe_pixels2 = GPS2Pixel(float(TPlat[i+1]),float(TPlon[i+1]), InvProjMat, adfGeoTransform) 
                    cv2.line(frame, (TPframe_pixels1[0], TPframe_pixels1[1]), (TPframe_pixels2[0]+10,TPframe_pixels2[1]+10), (0,0,255), 2)
                    cv2.circle(frame, TPframe_pixels1, 10, (0,0,255), 10)

                    if (i == len(TPlat)-2):
                        cv2.circle(frame, TPframe_pixels2, 10, (0,0,255), 10)

            out.write(frame)
            cv2.imshow('Frame', frame)
            cv2.waitKey(1)
            input()

        else:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # increment frame counter
        current_frame += 1

def main():
    if(len(sys.argv) != 6):
        print("Usage: python display_wf_info.py <video_input> <Projection_Matrix> <geolocation_map> <log_file_at_edge> <log_file_at_TP>")
        exit()

    video_input_file = sys.argv[1]
    ProjMat_file = sys.argv[2]
    geolocMap_file = sys.argv[3]
    EdgeWFlog_file =sys.argv[4]
    TPlog_file =sys.argv[5]
    
    print("Displaying log information from " + EdgeWFlog_file + " and " + TPlog_file + " on video " + video_input_file)

    # Video upon which the information will be printed 
    cap = cv2.VideoCapture(video_input_file)

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", (1920, 1080))
    if not cap.isOpened():
        print("An error occured while trying to open a video or file")

    # Video output
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    width, height  = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter('TP.mp4', fourcc, 12, (width,height))

    # Log file include workflow execution information as defined in "headers" varibale 
    headers = 'cam_id frame timestamp category lat lon geohash speed yaw obj_id x y w h'.split()
    df = pd.read_csv(EdgeWFlog_file, delimiter=' ', names=headers)
    df = df[(df['obj_id'] == "20939_1") | (df['obj_id'] == "20939_27") | (df['obj_id'] == "20939_84")]

    # Log file include in the TP log file as defined in "TPheaders" varibale 
    TPheaders = 'frame ts obj_id TPlat TPlon TPts'.split()
    TPdf = pd.read_csv(TPlog_file, delimiter=' ', names=TPheaders)
    TPdf = TPdf[(TPdf['obj_id'] == "20939_1") | (TPdf['obj_id'] == "20939_27") | (TPdf['obj_id'] == "20939_84")]

    # Generate matrices
    ProjMat, InvProjMat, adfGeoTransform = generate_transformation_da(ProjMat_file, geolocMap_file)

    # Display workflow information into the video
    display_boxes(cap, out, df, TPdf, ProjMat, InvProjMat, adfGeoTransform)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
