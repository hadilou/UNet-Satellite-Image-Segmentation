# -*- coding: UTF-8 -*-
import numpy as np
import cv2
import tensorflow as tf
import os
import random
import argparse

train_path='/train'
validation_path='./validation'
train_annotation_path='./train_annotation--'
validation_annotation='./validation_annotation--'
	
def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_image(path,period,load_label=True):
	img_path = os.path.join(path,'.jpg')
	
	image = cv2.imread(img_path,flags=cv2.IMREAD_UNCHANGED)
	image = image[:,:,0:3]
	image = np.uint8(image)
	if period == train:
		root=train_path
	else:
		root=validation_path
	if load_label:
		label_path=os.path.splitext(root)[0]
		label_path=os.path.join(label_path,'%.png')
		label = cv2.imread(label_path)
		label = label[:,:,0]
		label = np.uint8(label)
		return image, label
	return image

def random_patch(image, label, patch_size):
	row = image.shape[0]
	col = image.shape[1]
	r = random.randint(0,row-patch_size)
	c = random.randint(0,col-patch_size)
	sub_image = image[r:r+patch_size,c:c+patch_size]
	sub_label = label[r:r+patch_size,c:c+patch_size]
	return sub_image, sub_label

def save_image(sub_image, sub_label, writer,augment=False):
	image = sub_image
	label = sub_label
	if augment:
		for i in range(2):
			for j in range(4):
				image_raw = sub_image.tostring()
				label_raw = sub_label.tostring()
				row = sub_image.shape[0]
				col = sub_image.shape[1]
				example = tf.train.Example(features=tf.train.Features(feature={
					'row': _int64_feature(row),
					'col': _int64_feature(col),
					'image_raw': _bytes_feature(image_raw),
					'label_raw': _bytes_feature(label_raw),
					}))
				writer.write(example.SerializeToString())
				image = np.rot90(image)
				label = np.rot90(label)
			image = np.fliplr(image)
			label = np.fliplr(label)
	else:
		image_raw = sub_image.tostring()
		label_raw = sub_label.tostring()
		row = sub_image.shape[0]
		col = sub_image.shape[1]
		example = tf.train.Example(features=tf.train.Features(feature={
			'row': _int64_feature(row),
			'col': _int64_feature(col),
			'image_raw': _bytes_feature(image_raw),
			'label_raw': _bytes_feature(label_raw),
			}))
		writer.write(example.SerializeToString())

def main(args):
	train_path='/train'
	validation_path='./validation'
	train_annotation_path='./train_annotation'
	validation_annotation='./validation_annotation'
	
	dataset_name='BDCI'
	dataset_name2 = 'dataset'
	dataset_name3 = 'BDCI-semi'
	dataset_path='./BDCI2017-jiage'
	dataset_path2 = './dataset'
	dataset_path3 = './BDCI2017-jiage-Semi'
	
	period = args.period
	
	if not (os.path.isdir('./TFRecord')):
		os.makedirs('./TFRecord')
	TFRecord_path='./TFRecord'
	
	save_path = os.path.join(TFRecord_path,'%s.tfrecord'%period)
	writer = tf.python_io.TFRecordWriter(save_path)

	patch_size = 256

	if period=='train':
		for image_path in (os.listdir(train_path)):
			label_path=os.path.splitext(image_path)[0]
			image,label= load_image(image_path,'train')
			save_image(image,label,writer,augment=True)
			print('Done Preparing traing dataset')

		# data_size = 5
		# sample_size = 256
		# path = os.path.join(dataset_path2,period)
		# for i in range(1,data_size+1):
		# 	image, label = load_image(path,i,tif=True)
		# 	row = image.shape[0]
		# 	col = image.shape[1]
		# 	num = np.int64(row/sample_size*col/sample_size)
		# 	for j in range(0,num):
		# 		sub_image,sub_label = random_patch(image,label,sample_size)
		# 		sub_label[sub_label==5]=2
		# 		sub_image=cv2.resize(sub_image,(patch_size,patch_size),interpolation=cv2.INTER_CUBIC)
		# 		sub_label=cv2.resize(sub_label,(patch_size,patch_size),interpolation=cv2.INTER_NEAREST)
		# 		save_image(sub_image,sub_label,writer,augment=True)
		# 		print('NO.%d patch in %s-%s-%d is saving...'%(j, dataset_name2, period,i))

	if period=='validation':
		for image_path in (os.listdir(validation_path)):
			label_path=os.path.splitext(image_path)[0]
			image,label= load_image(image_path,'validation')
			save_image(image,label,writer,augment=True)
			print('Done Preparing validation dataset')
	writer.close()



if __name__ == '__main__':
	parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
	parser.add_argument("--period", choices=['train', 'validation'], default="train", help="period")
	args = parser.parse_args()
	main(args)
