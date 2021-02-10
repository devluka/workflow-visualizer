import os
import sys
import numpy as np
import cv2
import pandas as pd
from osgeo import gdal
from display_info_map import MapVis
from display_wf_info import DisplayInfo

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


def create_CD_dataframe(CDlog_file):
    #Log file include the CD log file as defined in "CDheaders" variable
    CDheaders = 'frame obj_ids_collision GPS_info_about_collision'.split()
    CDdf = pd.read_csv(CDlog_file, delimiter=' ', names=CDheaders)
    CDdf = CDdf[(CDdf['obj_ids_collision'] == "20939_1,20939_27") | (CDdf['obj_ids_collision'] == "20939_84,20939_27")]

    return CDdf

def create_WF_dataframe(EdgeWFlog_file):
    # Log file include workflow execution information as defined in "headers" varibale 
    headers = 'cam_id frame timestamp category lat lon geohash speed yaw obj_id x y w h frame_tp timestamp_last_tp TPlat TPlon TPts'.split()
    df = pd.read_csv(EdgeWFlog_file, delimiter=' ', names=headers)
    #df = df[(df['obj_id'] == "20939_1") | (df['obj_id'] == "20939_27") | (df['obj_id'] == "20939_84")]

    return df


def main():
    print(len(sys.argv))
    if(len(sys.argv) != 6 and len(sys.argv) != 5):
        print("Usage: python main.py <video_input> <Projection_Matrix> <geolocation_map> <log_file_at_edge> [<log_file_at_CD>] (optional)")
        exit()
    
    video_input_file = sys.argv[1]
    ProjMat_file = sys.argv[2]
    geolocMap_file = sys.argv[3]
    EdgeWFlog_file =sys.argv[4]

    cd_info_avail = False

    if len(sys.argv) == 6:
        cd_info_avail = True
        CDlog_file = sys.argv[5]
        print("Displaying log information from " + EdgeWFlog_file + " and " + CDlog_file + " on video " + video_input_file)
    else:
        print("Displaying log information from " + EdgeWFlog_file + " on video " + video_input_file)


    df = create_WF_dataframe(EdgeWFlog_file)

    CDdf = None
    if cd_info_avail:
        CDdf = create_CD_dataframe(CDlog_file)

    map_vis = MapVis(data=df,
                map_path='map.png',  
                osm_points=(44.6630, 10.9281, 44.6533, 10.9450)) # Upper left and lower right coordinates of the map
    map_vis.wf_tp_osm_position()

    ProjMat, InvProjMat, adfGeoTransform = generate_transformation_da(ProjMat_file, geolocMap_file)

    display_info = DisplayInfo(video_input=video_input_file, df=df, ProjMat=ProjMat, InvProjMat=InvProjMat, adfGeoTransform=adfGeoTransform, CDdf=CDdf, cd_info_avail=cd_info_avail)
    display_info.display_video() 


if __name__ == "__main__":
    main()
