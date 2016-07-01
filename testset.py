import cv2
import numpy as np
import pandas as pd
import os
import glob
import math

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold

from numpy.random import permutation

class Testset:
    def __init__(self):
        

        self.test_image, self.test_id = self.load_test()

        # Init params
        self.train_ptr = 0
        self.test_ptr = 0
#        self.train_size = len(self.train_label)
        self.test_size = len(self.test_id)
        self.crop_size = 224
        self.scale_size = 256
        self.mean = np.array([104., 117., 124.])
        self.n_classes = 10    
        
    def load_test(self):
        print('Read test images')
        path = os.path.join('/home/spookfish/Projects/data/statefarm', 'imgs', 'test', '*.jpg')
        files = glob.glob(path)
        X_test = []
        X_test_id = []
        total = 0
        thr = math.floor(len(files)/10)
        for fl in files:
            flbase = os.path.basename(fl)
#            img = get_im(fl, img_rows, img_cols, color_type)
            X_test.append(fl)
            X_test_id.append(flbase)
            total += 1
            if total % thr == 0:
                print('Read {} images from {}'.format(total, len(files)))
    
        return X_test, X_test_id
        
    def next_batch(self, batch_size, phase):
        # Get next batch of image (path) and labels

        if phase == 'test':
            if self.test_ptr + batch_size < self.test_size:
                paths = self.test_image[self.test_ptr:self.test_ptr + batch_size]
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr + batch_size)%self.test_size
                paths = self.test_image[self.test_ptr:] + self.test_image[:new_ptr]
                self.test_ptr = new_ptr
        else:
            return None, None
        
        # Read images
        images = np.ndarray([batch_size, self.crop_size, self.crop_size, 3])
        for i in xrange(len(paths)):
            img = cv2.imread(paths[i])
            h, w, c = img.shape
            assert c==3
            
            img = cv2.resize(img, (self.crop_size, self.crop_size))
            img = img.astype(np.float32)
            img -= self.mean
#            shift = int((self.scale_size-self.crop_size)/2)
#            img_crop = img[shift:shift+self.crop_size, shift:shift+self.crop_size, :]
            images[i] = img

        return images

