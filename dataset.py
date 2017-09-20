from PIL import Image
from scipy import misc
import numpy as np
import os
from random import shuffle

class Dataset(object):
    def __init__(self, imageRootDir, newH=256, newW=256, augH=286, augW=286): 
        """
        imageRootDir:   The root directoy storing all the images
        newH:           The height of training images 
        newW:           The width of training images
        augH:           The resized image height for data augmentation
        augW:           The resized image width for data augmentation
        """
        def getList(inputDir):
            filelist = []
	    for rootdir, subdir, files in os.walk(inputDir):
                if "Images" not in rootdir:         # Only pick the Images Folder
                    continue
	        for filename in files:
	            extname = filename.split('.')[-1]
		    if extname == 'jpg' or extname == 'JPG' \
		        or extname == 'png' or extname == 'PNG' \
		        or extname == 'tif' or extname == 'TIF':
		        filelist.append(os.path.join(rootdir, filename))
            return filelist
	self.H = newH		# resize the images into new H
	self.W = newW		# resize the images inot new W
        self.augH = augH
        self.augW = augW
	self.ptr = 0
	self.filelist = getList(imageRootDir)		# List save all the image paths
	self.total_num = len(self.filelist)
        """
        print('{:10} = {:4d}\n{:10} = {:4d}\n{:10} = {:4d}\n{:10} = {:4d}\n{:10} = {:4d}\n'.format('self.H', self.H,\
                'self.W', self.W, 'self.augH', self.augH, 'self.augW', self.augW, 'self.total_num', self.total_num))
        """
    def next_batch(self, batch_size, is_test=True):
        if self.ptr + batch_size < self.total_num:
	    batch_list = self.filelist[self.ptr:self.ptr+batch_size]
	    self.ptr += batch_size
	else:
	    batch_list = self.filelist[self.ptr:] + self.filelist[:self.ptr+batch_size-self.total_num]
            shuffle(self.filelist)      # Shuffle the training list 
	    self.ptr = (self.ptr + bach_size) % self.total_num

        # Get the batch list file for image 
        image_batch_list = batch_list
        # Get the batch list file for labels
        label_batch_list = [path.replace('Images', 'labelImage').replace('.tif', '.png') \
	    for path in batch_list]			                                    
        # Create a zip file for both lists
        pairpath_batch_list = zip(image_batch_list, label_batch_list)
        # Load batch image and label
        batch_data = [self.load_image_label_pair(pairpath, is_test) for pairpath in pairpath_batch_list]

        return np.array(batch_data)
    def load_random_images(self, batch_size, is_test=True):
        index = np.random.permutation(self.total_num)[0]       # random pick examples for testing
        if index + batch_size < self.total_num:
	    batch_list = self.filelist[index:index+batch_size]
	else:
	    batch_list = self.filelist[index:] + self.filelist[:index+batch_size-self.total_num]
        # Get the batch list file for image 
        image_batch_list = batch_list
        # Get the batch list file for labels
        label_batch_list = [path.replace('Images', 'labelImage').replace('.tif', '.png') \
	    for path in batch_list]			                                    
        # Create a zip file for both lists
        pairpath_batch_list = zip(image_batch_list, label_batch_list)
        # Load batch image and label
        batch_data = [self.load_image_label_pair(pairpath, is_test) for pairpath in pairpath_batch_list]

        return np.array(batch_data)

    def load_image_label_pair(self, pathpair=None, is_test=True, flip=True):
        imgs = []
        if is_test: # Use the original images for testing
            for i in range(2):
                im = misc.imread(pathpair[i])
                im = misc.imresize(im, (self.H, self.W))
                imgs.append(im)
        else:
            h1 = int(np.ceil(np.random.uniform(1e-2, self.augH-self.H)))
            w1 = int(np.ceil(np.random.uniform(1e-2, self.augW-self.W)))
            randflip = np.random.random()
            for i in range(2):
                im = misc.imread(pathpair[i])
                im = misc.imresize(im, (self.augH, self.augW))
                im = im[h1:h1+self.H, w1:w1+self.W]
                if flip and randflip > 0.5:
                    im = np.fliplr(im)
                imgs.append(im)
        # normalize the image pixel into [-1. 1.]
        imgs = np.array(imgs)
        imgs = imgs / 127.5 - 1.
        imgs = np.transpose(imgs, (1, 2, 0))    # move channel to the last axis
        return imgs

    def load_image(self, imagepath):
        im = misc.imread(imagepath)
        im = misc.imresize(im, (self.H, self.W))
	im = np.expand_dims(im, axis=2)			# Make the image shape as (H, W, C)
        im = im / 127.5 - 1.
	return im
"""
if __name__ == "__main__":
    imageRootDir = '/home/gxdai/MMVC_LARGE/Guoxian_Dai/data/medicalImage/wustl/TrainingSet'
    print("imageRootDir={}".format(imageRootDir))
    dataset = Dataset(imageRootDir=imageRootDir)
    print("ENTER LOOP")
    for i in range(1000):
        print("Loop: {}".format(i))
        ims_labels = dataset.next_batch(10, is_test=True)
        print("************************")
        print(ims_labels.shape)
        print(np.amax(ims_labels))
"""
