#!/bin/bash
set -e
echo "Installing OpenCV 4.6.0 on your Raspberry Pi 64-bit OS"
echo "It will take atleast 2 hours!"
cd ~

# Installing dependencies
sudo apt install -y build-essential cmake git unzip pkg-config
sudo apt install -y libjpeg-dev libtiff-dev libpng-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y libgtk2.0-dev libcanberra-gtk* libgtk-3-dev
sudo apt install -y libgstreamer1.0-dev gstreamer1.0-gtk3
sudo apt install -y libgstreamer-plugins-base1.0-dev gstreamer1.0-gl
sudo apt install -y libxvidcore-dev libx264-dev
sudo apt install -y python3-dev python3-numpy python3-pip
sudo apt install -y libtbb2 libtbb-dev libdc1394-22-dev
sudo apt install -y libv4l-dev v4l-utils
sudo apt install -y libopenblas-dev libatlas-base-dev libblas-dev
sudo apt install -y liblapack-dev gfortran libhdf5-dev
sudo apt install -y libprotobuf-dev libgoogle-glog-dev libgflags-dev
sudo apt install -y protobuf-compiler
sudo sed -ie 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=3000/' /etc/dphys-swapfile
sudo /etc/init.d/dphys-swapfile stop
sudo /etc/init.d/dphys-swapfile start

# Downloading latest version
cd ~ 
sudo rm -rf opencv*
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.6.0.zip 
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.6.0.zip
unzip opencv.zip 
unzip opencv_contrib.zip 

# Administrator privilages to make life easier
mv opencv-4.6.0 opencv
mv opencv_contrib-4.6.0 opencv_contrib

# Cleaning up .zip files
rm opencv.zip
rm opencv_contrib.zip

# Set install dir
cd ~/opencv
mkdir build
cd build

# Run cmake
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
-D ENABLE_NEON=ON \
-D WITH_OPENMP=ON \
-D WITH_OPENCL=OFF \
-D BUILD_TIFF=ON \
-D WITH_FFMPEG=ON \
-D WITH_TBB=ON \
-D BUILD_TBB=ON \
-D WITH_GSTREAMER=ON \
-D BUILD_TESTS=OFF \
-D WITH_EIGEN=OFF \
-D WITH_V4L=ON \
-D WITH_LIBV4L=ON \
-D WITH_VTK=OFF \
-D WITH_QT=OFF \
-D OPENCV_ENABLE_NONFREE=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D BUILD_EXAMPLES=OFF ..

# Run make
make -j4
sudo make install
sudo ldconfig

# Clean up (frees 300 MB)
make clean
sudo apt update
sudo sed -ie 's/CONF_SWAPSIZE=3000/CONF_SWAPSIZE=100/' /etc/dphys-swapfile
sudo /etc/init.d/dphys-swapfile stop
sudo /etc/init.d/dphys-swapfile start

echo "Congratulations!"
echo "You've successfully installed OpenCV 4.6.0 on your Raspberry Pi 64-bit OS"