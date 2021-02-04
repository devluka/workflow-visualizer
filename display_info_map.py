import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from osgeo import gdal

class MapVis:
    def __init__(self, data, data_tp, tp_avail, map_path, osm_points):
        self.data = data
        self.data_tp = data_tp
        self.tp_avail = tp_avail
        self.map_path = map_path
        self.osm_points = osm_points
        self.image = Image

    def wf_tp_osm_position(self):
        self.image = Image.open(self.map_path, 'r')  
        img_points = []
        
        draw = ImageDraw.Draw(self.image)

        if self.tp_avail:
            img_points_tp = []
            for _, row in self.data_tp.iterrows():
                TPlat, TPlon = row['TPlat'].split(','), row['TPlon'].split(',')

                for lat,lon in zip(TPlat, TPlon):
                    lat_lon = (float(lat),float(lon))
                    x1, y1 = self.scale_to_img(lat_lon, (self.image.size[0], self.image.size[1]))
                    img_points_tp.append((x1, y1))
                draw.point(img_points_tp, 'black')

        wf_data = tuple(zip(self.data['lat'].values, self.data['lon'].values))
        for data in wf_data:
            x1, y1 = self.scale_to_img(data, (self.image.size[0], self.image.size[1]))
            img_points.append((x1, y1))  
        draw.point(img_points, 'red')

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
