import os
import sys
import numpy as np
import cv2
import pandas as pd
from osgeo import gdal

class DisplayInfo:
    previous_positions = 100

    def __init__(self, video_input, df, ProjMat, InvProjMat, adfGeoTransform, CDdf, cd_info_avail):
        self.video_input = video_input
        self.df = df
        self.ProjMat = ProjMat
        self.InvProjMat = InvProjMat
        self.adfGeoTransform = adfGeoTransform
        self.CDdf = CDdf
        self.cd_info_avail = cd_info_avail

    def display_video(self):
        # Video upon which the information will be printed 
        cap = cv2.VideoCapture(self.video_input)

        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Frame", (1920, 1080))
        if not cap.isOpened():
            print("An error occured while trying to open a video or file")

        # Video output
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        width, height  = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter('TP.mp4', fourcc, 12, (width,height))

        self.display_boxes(cap, out)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

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
    def color_boxes(obj_id)
        color_id1 = { 'person': (0,127,255), 'car': (255,0,0), 'truck': (51,255,51), 'bus': (0,153,0), 'motor': (255,229,204), 'bike': (204,0,102), 'rider': (255,153,204), 'train': (160,160,160) }
        color_id2 = { '0': (0,127,255), '1': (255,0,0), '2': (51,255,51), '3': (0,153,0), '4': (255,229,204), '5': (204,0,102), '6': (255,153,204), '7': (160,160,160) }

        return color_id1.get(str(cat), (0,0,0))

    def display_boxes(self, cap, out):

        current_frame = 0

        positions_dict = {}   

        while cap.isOpened():

            ret, frame = cap.read()
            if ret == True:
                # Apply a Blur filter for anonymisation purposes
                frame = cv2.GaussianBlur(frame, (11, 11), 0)

                # extract rows according to the current video frame number
                df_current = self.df.loc[self.df['frame'] == current_frame]

                if self.cd_info_avail:
                    CDdf_current = self.CDdf.loc[self.CDdf['frame'] == current_frame]
                
                # Process the workflow log file 
                for _, row in df_current.iterrows():
                    # extract x, y, w, h from dataframe
                    x, y, w, h, lon, lat = int(row['x']), int(row['y']), int(row['w']), int(row['h']), float(row['lon']), float(row['lat'])
                    TPlat, TPlon, TPts = row['TPlat'].split(','), row['TPlon'].split(','), row['TPts'].split(',')

                    label, cat = row['obj_id'], row['category']
                    color = color_boxes(str(cat))
                    
                    print(" ")
                    print("Processing frame:" + str(current_frame) + " obj_id:" + str(label) + " cat:" + str(cat) + " x:" + str(x) + " y:" + str(y) + " w:" + str(w) + " h:" + str(h) + " lat:" + str(lat) + " lon:" + str(lon) + " TPlat:" + str(TPlat) + " TPlon:" + str(TPlon))

                    # GPS position is considered to be at the center of the bounding box
                    tmp_lat,tmp_lon = self.Pixel2GPS( x + (w/2), y + (h/2))
                    frame_pixels = self.GPS2Pixel(lat,lon)
    
                    self.add_frame_pixels(positions_dict, label, frame_pixels)
                    self.print_frame_positions(positions_dict, frame)

                    cv2.rectangle(frame, (x, y), (w + x, y + h), color, 3)
                    cv2.rectangle(frame, (x, y - 25), (x + 12 * len(label), y ), color, -1)
                    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, .6, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.circle(frame,frame_pixels, 5, color, 5)

                    for i in range(len(TPlat)-1):
                        if (float(TPlat[i]) != 0.0):
                            TPframe_pixels1 = self.GPS2Pixel(float(TPlat[i]),float(TPlon[i])) 
                            TPframe_pixels2 = self.GPS2Pixel(float(TPlat[i+1]),float(TPlon[i+1])) 
                            cv2.line(frame, (TPframe_pixels1[0], TPframe_pixels1[1]), (TPframe_pixels2[0],TPframe_pixels2[1]), (0,0,255), 2)
                            cv2.circle(frame, TPframe_pixels1, 4, (0,0,255), 4)

                            if (i == 0):
                                cv2.line(frame, (frame_pixels[0], frame_pixels[1]), (TPframe_pixels1[0],TPframe_pixels1[1]), (0,0,255), 2)
                            if (i == len(TPlat)-2):
                                cv2.circle(frame, TPframe_pixels2, 4, (0,0,255), 4)
               
                if self.cd_info_avail:
                    #Process the CD log file
                    for _, row in CDdf_current.iterrows():
                        obj_ids_labes = row['obj_ids_collision']
                        GPS_label = row['GPS_info_about_collision']

                        GPS_coordinates = GPS_label.split(',')
                        CDlat, CDlon = GPS_coordinates[0], GPS_coordinates[1]
                        CDframe_pixels = self.GPS2Pixel(float(CDlat), float(CDlon))
                        
                        cv2.circle(frame, CDframe_pixels, 10, (0,0,0), 10)
                        cv2.putText(frame, obj_ids_labes, CDframe_pixels, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, .6, (255, 255, 255), 2, cv2.LINE_AA)
                        print("GPS_info: Obj_ids "  + str(obj_ids_labes) + " gps_info_collision " + str(GPS_label))

                out.write(frame)
                cv2.imshow('Frame', frame)
                cv2.waitKey(1)
                input()

            else:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # increment frame counter
            current_frame += 1


    def print_frame_positions(self, positions_dict, image, spacing=20):
        for i, (key, val) in enumerate(positions_dict.items()):
            if len(val) > 1:
                for i in range(len(val) - 1):
                    pixel1 = val[i]
                    pixel2 = val[i+1]

                    cv2.line(image, (pixel1[0], pixel1[1]), (pixel2[0], pixel2[1]), (0,255,0), 2)
                    cv2.circle(image, pixel1, 4, (0,255,0), 3)

                    if (i == len(val)-2):
                        cv2.circle(image, pixel2, 4, (0,255,0), 3)


    def add_frame_pixels(self, positions_dict, object_id, pixels):
        if object_id not in positions_dict.keys():
            positions_dict[object_id] = [pixels]
        else:
            positions_dict[object_id].append(pixels)

        if len(positions_dict[object_id]) > self.previous_positions:
            positions_dict[object_id].pop(0)


    def Pixel2GPS(self, img_x, img_y):
        imgPixel = np.array([[[img_x,img_y]]], dtype='float32')

        mapPixel = cv2.perspectiveTransform(imgPixel, self.ProjMat)
        x = mapPixel[0][0][0]
        y = mapPixel[0][0][1]

        lat = self.adfGeoTransform[4] * x + self.adfGeoTransform[5] * y + self.adfGeoTransform[3]
        lon = self.adfGeoTransform[1] * x + self.adfGeoTransform[2] * y + self.adfGeoTransform[0]

        # print(f"agfGeo[0]: {adfGeoTransform[0]} [1] {adfGeoTransform[1]} [2] {adfGeoTransform[2]} [3] {adfGeoTransform[3]} [4] {adfGeoTransform[4]}")
        # print("At Pixel2GPS: ("+str(img_x)+","+str(img_y)+")->("+str(lat)+","+str(lon)+")")
        return lat,lon


    def GPS2Pixel(self, lat, lon):
        x = (lon - self.adfGeoTransform[0]) / self.adfGeoTransform[1]
        y = (lat - self.adfGeoTransform[3]) / self.adfGeoTransform[5]

        mapPixels = np.array([[[x, y]]], dtype='float32')
        imgPixels = cv2.perspectiveTransform(mapPixels, self.InvProjMat)

        # print("At GPS2Pixel: ("+str(lat)+","+str(lon)+")->("+str(imgPixels[0][0][0])+","+str(imgPixels[0][0][1])+")")
        return int(imgPixels[0][0][0]), int(imgPixels[0][0][1])
