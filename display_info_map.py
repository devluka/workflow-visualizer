import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from osgeo import gdal

class MapVis:
    def __init__(self, data, map_path, osm_points):
        self.data = data
        self.map_path = map_path
        self.osm_points = osm_points
        self.image = Image

    def wf_tp_osm_position(self):
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
        color_boxes = { 'person': (204,102,0), 'car': (0,0,153), 'truck': (51,255,51), 'bus': (0,153,0), 'motor': (204,255,229), 'bike': (204,0,102), 'rider': (255,153,204), 'train': (160,160,160) }

        self.image = Image.open(self.map_path, 'r')  
        img_points_tp = []
        
        draw = ImageDraw.Draw(self.image)

        for _, row in self.data.iterrows():
            cat = row['category']
            lat, lon, TPlat, TPlon = row['lat'], row['lon'], row['TPlat'].split(','), row['TPlon'].split(',')

            color = color_boxes.get(str(cat), (0,0,0))
            
            lat_lon = (float(lat), float(lon))
            x1, y1 = self.scale_to_img(lat_lon, (self.image.size[0], self.image.size[1]))
            
            #drawing points for objects
            draw.point((x1, y1), color)
            
            for TPlat,TPlon in zip(TPlat, TPlon):
                TPlat_TPlon = (float(TPlat),float(TPlon))
                x1, y1 = self.scale_to_img(TPlat_TPlon, (self.image.size[0], self.image.size[1]))
                img_points_tp.append((x1, y1))
            #drawing points for TP
            draw.point(img_points_tp, 'black')

        self.image.save('results.png')


    def scale_to_img(self, lat_lon, h_w):
        """
        Conversion from latitude and longitude to the image pixels.
        It is used for drawing the GPS records on the map image.
        """
        old = (self.osm_points[2], self.osm_points[0])
        new = (0, h_w[1])
        y = ((lat_lon[0] - old[0]) * (new[1] - new[0]) / (old[1] - old[0])) + new[0]
        old = (self.osm_points[1], self.osm_points[3])
        new = (0, h_w[0])
        x = ((lat_lon[1] - old[0]) * (new[1] - new[0]) / (old[1] - old[0])) + new[0]
      
        return int(x), h_w[1] - int(y)
