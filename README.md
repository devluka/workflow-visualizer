# Process a video based on an offline log file
  Render the Offline logs of the CLASS workflows into the processed video


### pip3 dependencies
```bash
pip3 install numpy
pip3 install opencv-python
pip3 install wheel
pip3 install pandas
pip3 install -U matplotlib
pip3 install --upgrade Pillow
```
### geolocation_map file path
```bash
/home/nvidia/masa_map.tif
```
### How to run it
```python
python main.py <video_input> <Projection_Matrix> <geolocation_map> <log_file_at_edge> [<log_file_at_CD>] (optional)
```
Agruments:

* video_input : path to the video upon which the information will be printed 
* projection_matrix : path to the projection matrix file
* geolocation_map: path to the .tif file
* log_file_at_edge: path to log file which includes workflow execution information
* log_file_at_CD: path to the CD log file
