FROM nvidia/cuda:7.5-cudnn5-devel
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION 2.7
ENV OPENCV_VERSION 3.1.0
RUN NUM_CORES=$(nproc)

RUN apt-get update && \
	apt-get install -y --no-install-recommends \
		ca-certificates \
		pkg-config \
		build-essential \
		libfreetype6-dev \
		libpng12-dev \
		wget \
		python$PYTHON_VERSION-dev \
		unzip \
		cmake \
		git \
		ssh \
		libatlas-base-dev \
		libboost-all-dev \
		gfortran \
		libtbb2 \
		libtbb-dev \
		libjasper-dev \
		libgtk2.0-dev \
		libavcodec-dev \
		libavformat-dev \
		libswscale-dev \
		libjpeg-dev \
		libtiff-dev \
		libhdf5-dev \
		nano \
		screen \
	&& rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
RUN pip install numpy matplotlib


#dlib
WORKDIR /
RUN git clone https://github.com/davisking/dlib.git
WORKDIR /dlib
RUN python setup.py install --yes USE_AVX_INSTRUCTIONS

#OpenCV
WORKDIR /
RUN wget https://github.com/Itseez/opencv/archive/${OPENCV_VERSION}.zip -O opencv3.zip
RUN unzip -q opencv3.zip
RUN mv /opencv-${OPENCV_VERSION} /opencv
RUN	rm opencv3.zip

RUN mkdir /opencv/build
WORKDIR /opencv/build


RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
#	-D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
	-D PYTHON_EXECUTABLE=$(which python) \
	-D BUILD_EXAMPLES=OFF \
	-D INSTALL_C_EXAMPLES=OFF \
	-D INSTALL_PYTHON_EXAMPLES=OFF \
	-D INSTALL_TESTS=OFF \
	-D BUILD_opencv_java=OFF \
	-D WITH_IPP=OFF  \
	-D WITH_TBB=ON \
	-D BUILD_NEW_PYTHON_SUPPORT=ON \
	-D WITH_QT=OFF ..

#cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") -D PYTHON_EXECUTABLE=$(which python) -D BUILD_EXAMPLES=OFF -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D INSTALL_TESTS=OFF -D BUILD_opencv_java=OFF -D WITH_IPP=OFF -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_QT=OFF ..

RUN make -j$NUM_CORES
RUN make install && make clean
RUN ldconfig
#for some reason the cv2.so isnt put anywhere useful.
RUN ln -s /opencv/build/lib/cv2.so /usr/lib/python2.7/dist-packages/


WORKDIR /

COPY requirements.txt .
RUN pip install -r requirements.txt

# Get the maxmind geoip db. TODO: have this auto-update
WORKDIR /usr/local/lib/python2.7/dist-packages/maxminddb
RUN wget http://geolite.maxmind.com/download/geoip/database/GeoLite2-Country.mmdb.gz
RUN gzip -d GeoLite2-Country.mmdb.gz

WORKDIR /

# Make ssh dir
RUN mkdir /root/.ssh/

RUN apt-get update

###############
# Caffe
###############

#some duplicates from above removed
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgflags-dev \
        libgoogle-glog-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-numpy \
        python-pip \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

#        python-dev \

ENV CLONE_TAG=master

WORKDIR /opt
RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git
#RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git .
ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT
#WORKDIR caffe
RUN for req in $(cat python/requirements.txt) pydot; do pip install $req; done

RUN mkdir build
#RUN cd build
WORKDIR build
RUN ls

RUN ldconfig

RUN cmake -DUSE_CUDNN=ON -DBUILD_python=ON -DBUILD_python_layer=ON ..
RUN make all -j"$(nproc)"
RUN make install
#RUN make runtest  #you can't run gpu stuff from a dockerfile....jesus ...https://github.com/NVIDIA/nvidia-docker/issues/153

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

COPY score.py /opt/caffe/python
COPY jrlayers.py /opt/caffe/python
COPY surgery.py /opt/caffe/python

COPY images_and_labels.txt /root
COPY image.jpg /root
COPY deploy.prototxt /root
COPY solver.prototxt /root
COPY train.prototxt /root
COPY val.prototxt /root

CMD ["bash"]