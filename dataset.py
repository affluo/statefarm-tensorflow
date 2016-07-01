import cv2
import numpy as np
import pandas as pd
import os
import glob
import math

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold

from numpy.random import permutation

from rotate import *

class Dataset:
    def __init__(self, train_index, test_index):
        

        self.train_image, self.train_label, self.test_image , self.test_label = self.load_train(train_index, test_index)

        # Init params
        self.train_ptr = 0
        self.test_ptr = 0
        self.train_size = len(self.train_label)
        self.test_size = len(self.test_label)
        self.crop_size = 224
        self.scale_size = 256
        self.mean = np.array([104., 117., 124.])
        self.n_classes = 10
    
    def get_driver_data(self):
        dr = dict()
        path = os.path.join('/home/spookfish/Projects/data/statefarm', 'driver_imgs_list.csv')
        print('Read drivers data')
        f = open(path, 'r')
        line = f.readline()
        while (1):
            line = f.readline()
            if line == '':
                break
            arr = line.strip().split(',')
            dr[arr[2]] = arr[0]
        f.close()
        return dr
        
    def load_train(self,train_index, test_index):
    #    X_train = []
        y_train = []
        driver_id = []
        train_paths = []
    
        driver_data = self.get_driver_data()
    
        print('Read train images')
        for j in range(10):
#            print('Load folder c{}'.format(j))
            path = os.path.join('/home/spookfish/Projects/data/statefarm', 'imgs', 'train',
                                'c' + str(j), '*.jpg')
            files = glob.glob(path)
            for fl in files:
                flbase = os.path.basename(fl)
    #            img = get_im(fl, img_rows, img_cols, color_type)
    #            X_train.append(img)
                y_train.append(j)
                driver_id.append(driver_data[flbase])
                train_paths.append(fl)
        
        unique_drivers = sorted(list(set(driver_id)))
        print('Unique drivers: {}'.format(len(unique_drivers)))
        print(unique_drivers)
#        return y_train, driver_id, unique_drivers, train_paths
        
        y = sorted(zip(driver_id,train_paths,y_train))
        df = pd.DataFrame(y)
        
        unique_id = [0]
        c = 0
        
        for i in range(len(df)-1):
            if df[0][i+1] == df[0][i]:
                unique_id.append(c)
            else:
                c += 1
                unique_id.append(c)
            
        df['ID'] = pd.DataFrame(unique_id)
        df.columns = ['Driver', 'Path', 'Label', 'ID']
        
        dfsplit = pd.DataFrame(np.random.randn(len(df), 2))
        msk = np.random.rand(len(dfsplit)) < 0.8
        train_data = df[msk]
        test_data = df[~msk]
        
#        train_data = df[df['ID'].isin(train_index)]
#        test_data = df[df['ID'].isin(test_index)]
        
        train_data = train_data.iloc[np.random.permutation(len(train_data))]
        test_data = test_data.iloc[np.random.permutation(len(test_data))]
        
        train_label = train_data['Label'].tolist()
        train_image = train_data['Path'].tolist()
        test_label = test_data['Label'].tolist()
        test_image = test_data['Path'].tolist()
        
        return  train_image, train_label, test_image, test_label     
        
    def rotatedRectWithMaxArea(self,w, h, angle):
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle (maximal area) within the rotated rectangle.
        """
        if w <= 0 or h <= 0:
          return 0,0
      
        width_is_longer = w >= h
        side_long, side_short = (w,h) if width_is_longer else (h,w)
    
        # since the solutions for angle, -angle and 180-angle are all the same,
        # if suffices to look at the first quadrant and the absolute values of sin,cos:
        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
        if side_short <= 2.*sin_a*cos_a*side_long:
          # half constrained case: two crop corners touch the longer side,
          #   the other two corners are on the mid-line parallel to the longer line
          x = 0.5*side_short
          wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
        else:
          # fully constrained case: crop touches all 4 sides
          cos_2a = cos_a*cos_a - sin_a*sin_a
          wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a
    
        return wr,hr
        
    def next_batch(self, batch_size, phase):
        # Get next batch of image (path) and labels
        if phase == 'train':
            if self.train_ptr + batch_size < self.train_size:
                paths = self.train_image[self.train_ptr:self.train_ptr + batch_size]
                labels = self.train_label[self.train_ptr:self.train_ptr + batch_size]
                self.train_ptr += batch_size
            else:
                new_ptr = (self.train_ptr + batch_size)%self.train_size
                paths = self.train_image[self.train_ptr:] + self.train_image[:new_ptr]
                labels = self.train_label[self.train_ptr:] + self.train_label[:new_ptr]
                self.train_ptr = new_ptr
        elif phase == 'test':
            if self.test_ptr + batch_size < self.test_size:
                paths = self.test_image[self.test_ptr:self.test_ptr + batch_size]
                labels = self.test_label[self.test_ptr:self.test_ptr + batch_size]
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr + batch_size)%self.test_size
                paths = self.test_image[self.test_ptr:] + self.test_image[:new_ptr]
                labels = self.test_label[self.test_ptr:] + self.test_label[:new_ptr]
                self.test_ptr = new_ptr
        else:
            return None, None
        
        # Read images
        images = np.ndarray([batch_size, self.crop_size, self.crop_size, 3])
        for i in xrange(len(paths)):
            img = cv2.imread(paths[i])
            h, w, c = img.shape
            assert c==3
            
            if phase == 'train':
                angle = np.random.rand()*20 -10
                img = rotate(img, angle)
                
                img = cv2.resize(img, (self.scale_size, self.scale_size))
                img = img.astype(np.float32)
                img -= self.mean
    #            shift = int((self.scale_size-self.crop_size)/2)
                shift1 = np.random.randint(0,self.scale_size-self.crop_size+1)
                shift2 = np.random.randint(0,self.scale_size-self.crop_size+1)
                img_crop = img[shift1:shift1+self.crop_size, shift2:shift2+self.crop_size, :]
                images[i] = img_crop
            else:
                img = cv2.resize(img, (self.crop_size, self.crop_size))
                img = img.astype(np.float32)
                img -= self.mean
                images[i] = img
                
        # Expand labels
        one_hot_labels = np.zeros((batch_size, self.n_classes))
        for i in xrange(len(labels)):
            one_hot_labels[i][labels[i]] = 1
        return images, one_hot_labels

