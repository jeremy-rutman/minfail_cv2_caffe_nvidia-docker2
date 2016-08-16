import copy
import os
import caffe
import logging
import sys

logging.basicConfig(level=logging.DEBUG)
import numpy as np
from PIL import Image
import cv2
import random


######################################################################################3
# MULTILABEL
#######################################################################################

class JrMultilabel(caffe.Layer):
    """
    Load (input image, label vector) pairs where label vector is like [0 1 0 0 0 1 ... ]

    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        example
        layer {
            name: "data"
            type: "Python"
            top: "data"
            top: "label"
            python_param {
            module: "jrlayers"
            layer: "JrMultilabel"
            param_str: "{\'images_and_labels_file\': \'/home/jeremy/image_dbs/tamara_berg/web1\', \'mean\': (104.00699, 116.66877, 122.67892)}"
            }
        """
        # config
        params = eval(self.param_str)

        self.images_and_labels_file = params['images_and_labels_file']
        self.mean = np.array(params['mean'])
        self.random_init = params.get('random_initialization', True) #start from random point in image list
        self.random_pick = params.get('random_pick', True) #pick random image from list every time
        self.seed = params.get('seed', 1337)
        self.new_size = params.get('new_size',None)
        self.batch_size = params.get('batch_size',1)  #######Not implemented, batchsize = 1
        self.augment_images = params.get('augment',False)
        self.augment_max_angle = params.get('augment_max_angle',5)
        self.augment_max_offset_x = params.get('augment_max_offset_x',10)
        self.augment_max_offset_y = params.get('augment_max_offset_y',10)
        self.augment_max_scale = params.get('augment_max_scale',1.2)
        self.augment_max_noise_level = params.get('augment_max_noise_level',0)
        self.augment_max_blur = params.get('augment_max_blur',0)
        self.augment_do_mirror_lr = params.get('augment_do_mirror_lr',True)
        self.augment_do_mirror_ud = params.get('augment_do_mirror_ud',False)
        self.augment_crop_size = params.get('augment_crop_size',(227,227)) #
        self.augment_show_visual_output = params.get('augment_show_visual_output',False)
        self.augment_distribution = params.get('augment_distribution','uniform')
        self.n_labels = params.get('n_labels',21)

        #on the way out
        self.images_dir = params.get('images_dir',None)

        print('imfile {} mean {} imagesdir {} randinit {} randpick {} '.format(self.images_and_labels_file, self.mean,self.images_dir,self.random_init, self.random_pick))
        print('see {} newsize {} batchsize {} augment {} augmaxangle {} '.format(self.seed,self.new_size,self.batch_size,self.augment_images,self.augment_max_angle))
        print('augmaxdx {} augmaxdy {} augmaxscale {} augmaxnoise {} augmaxblur {} '.format(self.augment_max_offset_x,self.augment_max_offset_y,self.augment_max_scale,self.augment_max_noise_level,self.augment_max_blur))
        print('augmirrorlr {} augmirrorud {} augcrop {} augvis {}'.format(self.augment_do_mirror_lr,self.augment_do_mirror_ud,self.augment_crop_size,self.augment_show_visual_output))

        self.idx = 0
        print('images+labelsfile {} mean {}'.format(self.images_and_labels_file,self.mean))
        # two tops: data and label
        if len(top) != 2:
            print('len of top is '+str(len(top)))
#            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")
#
        # load indices for images and labels
        #if file not found and its not a path then tack on the training dir as a default locaiton for the trainingimages file
        if self.images_and_labels_file is not None:
            if not os.path.isfile(self.images_and_labels_file) and not '/' in self.images_and_labels_file:
                if self.images_dir is not None:
                    self.images_and_labels_file = os.path.join(self.images_dir,self.images_and_labels_file)
            if not os.path.isfile(self.images_and_labels_file):
                print('COULD NOT OPEN IMAGES/LABELS FILE '+str(self.images_and_labels_file))
                return
            self.images_and_labels_list = open(self.images_and_labels_file, 'r').read().splitlines()
            self.n_files = len(self.images_and_labels_list)
            logging.debug('images and labels file: {} n: {}'.format(self.images_and_labels_file,self.n_files))
    #        self.indices = open(split_f, 'r').read().splitlines()
        else:
            print('option not supported')
#            return
#            self.imagefiles = [f for f in os.listdir(self.images_dir) if self.imagefile_suffix in f]

        self.idx = 0
        # randomization: seed and pick
#        print('imgslbls [0] {} [1] {}'.format(self.images_and_labels_list[0],self.images_and_labels_list[1]))
        if self.random_init:
            random.seed(self.seed)
            self.idx = random.randint(0, self.n_files-1)
        if self.random_pick:
            random.shuffle(self.images_and_labels_list)
#        print('imgslbls [0] {} [1] {}'.format(self.images_and_labels_list[0],self.images_and_labels_list[1]))
        logging.debug('initial self.idx is :'+str(self.idx)+' type:'+str(type(self.idx)))

        spinner = spinning_cursor()
        ##check that all images are openable and have labels
        ## and ge t
        good_img_files = []
        good_label_vecs = []
        check_files = False
        if check_files:
            print('checking image files')
            for line in self.images_and_labels_list:
                imgfilename = line.split()[0]
                img_arr = Image.open(imgfilename)
                in_ = np.array(img_arr, dtype=np.float32)

                if img_arr is not None:
                    vals = line.split()[1:]
                    label_vec = [int(i) for i in vals]
                    self.n_labels = len(vals)
                    label_vec = np.array(label_vec)
                    self.n_labels = len(label_vec)
                    if self.n_labels == 1:
                        label_vec = label_vec[0]    #                label_vec = label_vec[np.newaxis,...]  #this is required by loss whihc otherwise throws:
    #                label_vec = label_vec[...,np.newaxis]  #this is required by loss whihc otherwise throws:
    #                label_vec = label_vec[...,np.newaxis,np.newaxis]  #this is required by loss whihc otherwise throws:
    #                F0616 10:54:30.921106 43184 accuracy_layer.cpp:31] Check failed: outer_num_ * inner_num_ == bottom[1]->count() (1 vs. 21) Number of labels must match number of predictions; e.g., if label axis == 1 and prediction shape is (N, C, H, W), label count (number of labels) must be N*H*W, with integer values in {0, 1, ..., C-1}.

                    if label_vec is not None:
                        if len(label_vec) > 0:  #got a vec
                            good_img_files.append(imgfilename)
                            good_label_vecs.append(label_vec)
                            sys.stdout.write(spinner.next())
                            sys.stdout.flush()
                            sys.stdout.write('\b')
                      #      print('got good image of size {} and label of size {}'.format(in_.shape,label_vec.shape))
                        else:
                            print('something wrong w. image of size {} and label of size {}'.format(in_.shape,label_vec.shape))
                else:
                    print('got bad image:'+self.imagefiles[ind])
        else:  #
            for line in self.images_and_labels_list:
                imgfilename = line.split()[0]
                vals = line.split()[1:]
                self.n_labels = len(vals)
                label_vec = [int(i) for i in vals]
                label_vec = np.array(label_vec)
                self.n_labels = len(label_vec)
                if self.n_labels == 1:
                    label_vec = label_vec[0]
                good_img_files.append(imgfilename)
                good_label_vecs.append(label_vec)

        self.imagefiles = good_img_files
        self.label_vecs = good_label_vecs
        assert(len(self.imagefiles) == len(self.label_vecs))
        print('{} images and {} labels'.format(len(self.imagefiles),len(self.label_vecs)))
        self.n_files = len(self.imagefiles)
        print(str(self.n_files)+' good files in image dir '+str(self.images_dir))
        logging.debug('self.idx is :'+str(self.idx)+' type:'+str(type(self.idx)))

        #if images are being augmented then dont do this resize
        if self.augment_images == False:

            if self.new_size == None:   #the old size of 227 is not actually correct, original vgg/resnet wants 224
                print('uh i got no size so using 224x224')
                self.new_size = (224,224)
            top[0].reshape(self.batch_size, 3, self.new_size[0], self.new_size[1])
        else:
            self.new_size=(self.augment_crop_size[0],self.augment_crop_size[1])
            top[0].reshape(self.batch_size, 3, self.new_size[0], self.new_size[1])
#            top[0].reshape(self.batch_size, 3, self.augment_crop_size[0], self.augment_crop_size[1])

        logging.debug('reshaping labels to '+str(self.batch_size)+'x'+str(self.n_labels))
        top[1].reshape(self.batch_size, self.n_labels)

    def reshape(self, bottom, top):
        pass
        print('start reshape')
#        logging.debug('self.idx is :'+str(self.idx)+' type:'+str(type(self.idx)))
        if self.batch_size == 1:
            imgfilename, self.data, self.label = self.load_image_and_label()
        else:
            all_data = np.zeros((self.batch_size,3,self.new_size[0],self.new_size[1]))
            all_labels = np.zeros((self.batch_size,self.n_labels))
            for i in range(self.batch_size):
                imgfilename, data, label = self.load_image_and_label()
                all_data[i,...]=data
                all_labels[i,...]=label
                self.next_idx()
            self.data = all_data
            self.label = all_labels
        ## reshape tops to fit (leading 1 is for batch dimension)
 #       top[0].reshape(1, *self.data.shape)
 #       top[1].reshape(1, *self.label.shape)
        print('top 0 shape {} top 1 shape {}'.format(top[0].shape,top[1].shape))
        print('data shape {} label shape {}'.format(self.data.shape,self.label.shape))
##       the above just shows objects , top[0].shape is an object apparently

    def next_idx(self):
        if self.random_pick:
            self.idx = random.randint(0, len(self.imagefiles)-1)
        else:
            self.idx += 1
            if self.idx == len(self.imagefiles):
                print('hit end of labels, going back to first')
                self.idx = 0

    def forward(self, bottom, top):
        # assign output
        print('forward start')
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        # pick next input
        self.next_idx()
        print('forward end')

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image_and_label(self,idx=None):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        print('load_image_and_label start')
        if idx is None:
            idx = self.idx
        while(1):
            filename = self.imagefiles[idx]
            label_vec = self.label_vecs[idx]
            if self.images_dir:
                filename=os.path.join(self.images_dir,filename)
     #       print('the imagefile:'+filename+' index:'+str(idx))
            if not(os.path.isfile(filename)):
                print('NOT A FILE:'+str(filename))
                self.next_idx()   #bad file, goto next
                idx = self.idx
                continue
            print('calling augment_images with file '+filename)

#############start added code to avoid cv2.imread############
            im = Image.open(filename)
            if self.new_size:
                im = im.resize(self.new_size,Image.ANTIALIAS)
            in_ = np.array(im, dtype=np.float32)
            if in_ is None:
                logging.warning('could not get image '+filename)
                return None
#############end added code to avoid cv2.imread############

#            out_ = augment_images.generate_image_onthefly(in_, gaussian_or_uniform_distributions=self.augment_distribution,
#               max_angle = self.augment_max_angle,
#               max_offset_x = self.augment_max_offset_x,max_offset_y = self.augment_max_offset_y,
#               max_scale=self.augment_max_scale,
#             max_noise_level=self.augment_max_noise_level,noise_type='gauss',
#               max_blur=self.augment_max_blur,
#              do_mirror_lr=self.augment_do_mirror_lr,
 #              do_mirror_ud=self.augment_do_mirror_ud,
 #              crop_size=self.augment_crop_size,
 #              show_visual_output=self.augment_show_visual_output)
#            out_,unused = augment_images.generate_image_onthefly(in_,in_)
            logging.debug('here we go,trying to cv2.flip')
            out_ = cv2.flip(in_,0)
            logging.debug('there we went')

            #im = Image.open(filename)
            #if im is None:
            #    logging.warning('could not get image '+filename)
            #    self.next_idx()
            #    idx = self.idx
            #    continue
            #if self.new_size:
            #    im = im.resize(self.new_size,Image.ANTIALIAS)
            if out_ is None:
                logging.warning('could not get image '+filename)
                self.next_idx()
                idx = self.idx
                continue
            out_ = np.array(out_, dtype=np.float32)
            if len(out_.shape) != 3 or out_.shape[0] != self.new_size[0] or out_.shape[1] != self.new_size[1] or out_.shape[2]!=3:
                print('got bad img of size '+str(out_.shape) + '= when expected shape is 3x'+str(self.new_size))
                self.next_idx()  #goto next
                idx = self.idx
                continue
            break #got good img, get out of while


        print(str(filename) + ' has dims '+str(out_.shape)+' label:'+str(label_vec)+' idex'+str(idx))

#        in_ = in_[:,:,::-1]  #RGB->BGR - since we're using cv2 no need
        out_ -= self.mean
        out_ = out_.transpose((2,0,1))  #Row Column Channel -> Channel Row Column
#	print('uniques of img:'+str(np.unique(in_))+' shape:'+str(in_.shape))
        print('load_image_and_label end')
        return filename, out_, label_vec



def spinning_cursor():
    while True:
        for cursor in '|/-\\':
            yield cursor

