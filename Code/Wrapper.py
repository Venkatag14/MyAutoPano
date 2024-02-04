#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:

import argparse
import numpy as np
import cv2
from os import listdir
from os.path import dirname, abspath
import matplotlib.pyplot as plt
import random
from skimage.io import imread, imshow
from skimage import transform
import scipy
import sys

# Add any python libraries here

def LoadImagesFromFolder(folder):
	images = []
	for file in listdir(folder):
		tmp = cv2.imread(folder + "\\" + file)
		if tmp is not None:
			images.append(tmp)
	return images

def GetCorners(image_gray):
	# Here I found that maxCorners = 400 worked well, beyond which there were diminishing returns
	# Using low minimum quality level so that ANMS can do some work
	# Min distance set arbitrarily at 10
	corners = cv2.goodFeaturesToTrack(image=image_gray, 
										maxCorners=600, 
										qualityLevel=0.001, 
										minDistance=10)
	# corners = cv2.cornerHarris(image_gray, 5, 3, 0.01, )
	return np.intp(corners)


def ANMS(corners, N_best):
	# Find all local maxima using imregionalmax
	# Find x, y of all local maxima
	# Set r_i = inf for i = 1:N_strong
	corner_dict = {}

	N_strong = corners.shape[0]
	for i in range(N_strong):
		for j in range(N_strong):
			if j > i:
				xj, yj = corners[j,:,:].flatten()
				xi, yi = corners[i,:,:].flatten()
				strength_radius = (xj - xi)**2 + (yj - yi)**2

				if (xi, yi) not in corner_dict:
					corner_dict[(xi, yi)] = strength_radius
				elif strength_radius < corner_dict[(xi, yi)]:
					corner_dict[(xi, yi)] = strength_radius
	
	corners_sorted_by_radius = [x[0] for x in sorted(corner_dict.items(), key=lambda item: item[1], reverse=True)]

	return np.array(corners_sorted_by_radius[:N_best])

def GetFeatureDescriptors(image, corners):
	# Take a 41x41 patch centered around the keypoint
	# Are the x, y's correct and not ulta?
	h, w = image.shape
	feature_vectors = np.zeros((corners.shape[0], 64))
	for i, keypoint in enumerate(corners):
		x, y = keypoint
		# Grab a patch from image within bounds
		x_min = np.amax([0,x-20])
		x_max = np.amin([w,x+20])
		y_min = np.amax([0,y-20])
		y_max = np.amin([h,y+20])
		patch = image[y_min:y_max+1, x_min:x_max+1]

		# Apply Gaussian Blur
		patch = cv2.GaussianBlur(patch, ksize=(3,3), sigmaX=1.5, sigmaY=1.5)

		# Subsample
		patch = cv2.resize(patch, dsize=(8,8))

		feature_vectors[i] = patch.reshape(-1,)
	
	# Standardize vector
	feature_vectors = StandardizeAbout0(feature_vectors)

	return feature_vectors

def StandardizeAbout0(vector):
	vector = (vector  - np.mean(vector, axis=1, keepdims=True))/np.std(vector, axis=1, keepdims=True)

	return vector

def MatchingFeatures(Feature_vectors, corners):
	# Taking 2 feature vectors and 2 image best corners, finding best matched corners from it.
	matched_corners = []
	for i, f1 in enumerate(Feature_vectors[0]):
		# flattening feature vector
		difference = [] 

		for j, f2 in enumerate(Feature_vectors[1]):
			#flattening feature vector
			# finding square difference between feature vectors of both images
			Square_difference = np.array((f1-f2)**2)
			Square_difference = np.sum(Square_difference)
			difference.append(Square_difference)

		difference = np.array(difference)
		indices_sorted = np.argsort(difference) #returning array of indices which represents the sorted order of 'difference'

		match_ratio = difference[indices_sorted[0]]/difference[indices_sorted[1]] #finding match ratio between best match and second best match

		# append only if ratio is less than 0.75 (0.75 selected based on previous works, might have to change)
		if match_ratio < 0.4:
			matched_corners.append((corners[0][i], corners[1][indices_sorted[0]]))
	return matched_corners

def DrawMatches(images, matched_corners):
	
	#creating keypoints
	keypoints0 = [cv2.KeyPoint(float(x[0][0]), float(x[0][1]), 1) for x in matched_corners]
	keypoints1 = [cv2.KeyPoint(float(x[1][0]), float(x[1][1]), 1) for x in matched_corners]

	#creating matched pair indices
	matches = [cv2.DMatch(i, i, 0) for i in range(len(matched_corners))]

	# creating matching visualization using drawMatches
	matched_img = cv2.drawMatches(images[0], keypoints0, images[1], keypoints1, matches, None, flags=2)
	return matched_img


#creating different function to draw matches from ransac output as ransac output use 'pt' attribute to specify the point
def DrawMatchesRansac(images, matched_corners):
	keypoints0 = [cv2.KeyPoint(x[0].pt[0], x[0].pt[1], 1) for x in matched_corners]
	keypoints1 = [cv2.KeyPoint(x[1].pt[0], x[1].pt[1], 1) for x in matched_corners]

	# Creating matched pair indices
	matches = [cv2.DMatch(i, i, 0) for i in range(len(matched_corners))]

	# Creating matching visualization using drawMatches
	matched_img = cv2.drawMatches(images[0], keypoints0, images[1], keypoints1, matches, None, flags=2)
	return matched_img

def Homography(randompoints):
	#initializing known parameters with four random points
	A = []
	#forming the matrix with 4 random points
	for pts in randompoints:
		xs, ys = pts[0][0], pts[0][1]
		xd, yd = pts[1][0], pts[1][1]
		a = [[xs, ys, 1, 0, 0, 0, -xd*xs, -xd*ys, -xd],
			[0, 0, 0, xs, ys, 1, -yd*xs, -yd*ys, -yd]]
		A.append(a)
	A = np.array(A)
	A = A.reshape(8,9)

	#finding eigen values and vectors for loss function
	eigenvalues, eigenvectors = np.linalg.eig(np.dot(A.T, A))

	#finding eigen vector for minimum eigen value
	min_eigenvalue_index = np.argmin(eigenvalues)
	min_eigenvector = eigenvectors[:, min_eigenvalue_index]

	#assigning h_matrix
	h_matrix = min_eigenvector.reshape(3,3)
	return h_matrix

def dot_product(M, keypoint):
	keypoint = np.array([keypoint[0], keypoint[1], 1])
	keypoint = keypoint.T
	product = np.dot(M, keypoint)
	if product[2]!=0:
		product = product/product[2]
	else:
		product = product/0.000001
	product = product.T
	return product[0:2]





def Ransac(matched_corners):
	#creating empty lists
	inliers = []
	Inliers_count = []
	H_Matrix = []
	matched_corners = np.array(matched_corners)
	#defining number of iterations
	Nmax = 2000
	# defining keypoints of both images from matched_corners
	# keypoints0 = [cv2.KeyPoint(float(x[0][0]), float(x[0][1]), 1) for x in matched_corners]
	# keypoints1 = [cv2.KeyPoint(float(x[1][0]), float(x[1][1]), 1) for x in matched_corners]
	threshold = 5 #5 working
	#iterations
	for i in range(Nmax):
		
		# keypoints0 = [cv2.KeyPoint(float(x[0][0]), float(x[0][1]), 1) for x in matched_corners]
		# keypoints1 = [cv2.KeyPoint(float(x[1][0]), float(x[1][1]), 1) for x in matched_corners]
		#taking 4 random inidces
		keypoints0 = [x[0] for x in matched_corners]
		keypoints1 = [x[1] for x in matched_corners]
		four_indices = random.sample(range(0,len(matched_corners)),4)
		RandomPoints = matched_corners[four_indices]
		M = Homography(RandomPoints)

		#inliers in this iteration
		iter_inliers = []
		inlier_count = 0
		for i in range(0,len(keypoints0)):
			src_points = keypoints0[i]
			src_points_transformed = dot_product(M,keypoints0[i])
			dst_points = keypoints1[i]
			srcx, srcy = src_points_transformed[0], src_points_transformed[1]
			dstx, dsty = dst_points[0], dst_points[1]
			# ssd = np.linalg.norm(np.expand_dims(np.array(keypoints1[i]), 1) - dot_product(M, keypoints0[i]))

			ssd = ((dstx-srcx)**2+(dsty-srcy)**2)**0.5
			if ssd < threshold:
				inlier_count += 1
				iter_inliers.append((keypoints0[i],keypoints1[i]))

		#appending inliers and corresponding inlier count
		inliers.append(iter_inliers)
		H_Matrix.append(M)
		Inliers_count.append(inlier_count)
	
	#taking the inlier with largest inlier count(most effective h matrix)
	sort_index = np.argsort(Inliers_count)
	maximum_match_index = sort_index[-1]
	best_matched_pairs = inliers[maximum_match_index]
	best_H_matrix = H_Matrix[maximum_match_index]
	inlier_count = Inliers_count[maximum_match_index]

	return best_matched_pairs, best_H_matrix, inlier_count

def ImageStitch(images, h_matrix):

	#source image which is being warped
	srcimg = images[0]
	#destination image whose plane is used to warp the image
	dstimg = images[1]
	#height and width of both images
	height = [srcimg.shape[0], dstimg.shape[0]]
	width = [srcimg.shape[1], dstimg.shape[1]]
	#edges of both images, 1 is addedto each row as it get multiplied with 3*3 homography matrix
	src_edges = [[0, 0, 1],[0, height[0], 1],[width[0], 0, 1],[width[0], height[0], 1]]
	src_edges = np.array(src_edges)
	src_edges = src_edges.T
	dst_edges = [[0, 0],[0, height[1]],[width[1], 0],[width[1], height[1]]]
	dst_edges = np.array(dst_edges)
	dst_edges = dst_edges.T
	#transforming source image edges into destination image edges by multiplying the corner vector matrix with homography
	src_edges_warped = np.dot(h_matrix,src_edges)
	src_edges_warped = src_edges_warped/src_edges_warped[2,:]
	#taking only first 2 rows
	src_edges_warped = src_edges_warped[:2,:]
	#concatenaitng them to get minimum
	stitch_edges = np.concatenate((dst_edges, src_edges_warped), axis=1)
	#bottom left corner co-ordinates of the output image is minimum x,y co-ordinates between destination image and transformed source image
	blc = np.min(stitch_edges, axis=1)
	blc = blc.astype(np.int32)
	#top right corner co-ordinates of the output image is maximum x,y co-ordinates between destination image and transformed source image
	trc = np.max(stitch_edges, axis=1)
	trc = trc.astype(np.int32)

	t = [-blc[0], -blc[1]]
	#translation vector to translate the warped source image to the correct position for blending 
	translation_vector = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) 

	#warping and translating source image
	output = cv2.warpPerspective(srcimg, translation_vector.dot(h_matrix), (trc[0]-blc[0], trc[1]-blc[1]), flags = cv2.INTER_LINEAR)

	#adding destination image to the warped source image
	output[t[1]:height[1]+t[1],t[0]:width[1]+t[0]] = dstimg

	return output


def MyAutoPano(ImageFolderPath, ResultsPath):
	"""
	Read a set of images for Panorama stitching
	"""
	images = LoadImagesFromFolder(ImageFolderPath)
	iter = 0
	length = len(images)

	#while loop to run the process of stitching until full stitch created
	while len(images)>1:
		#creating best corners and feature vectors
		best_corners = []
		feature_vectors = []
		iter += 1
		print(iter)
		if iter >= length:
			break

		for i in range(0,len(images)):
			#Creating grayscale images
			images_gscale = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in images]
			img = images[i]
			img_gray = images_gscale[i]
			corner_img = np.copy(img)
			"""
			Corner Detection
			Save Corner detection output as corners.png
			"""
			corners= GetCorners(img_gray)

			f, axarr = plt.subplots(1,2)
			f.suptitle("Corner Detection using Shi-Tomasi Method", y=0.8)
			axarr[0].title.set_text("Input")
			axarr[0].set_xticks([])
			axarr[0].set_yticks([])
			axarr[1].title.set_text("Output")
			axarr[1].set_xticks([])
			axarr[1].set_yticks([])

			# Original 
			axarr[0].imshow(img)

			# Get image with corner overlay
			for coordinates in corners:
				x, y = coordinates.ravel()
				cv2.circle(corner_img, (x,y), 3, 255, -1)
			axarr[1].imshow(corner_img)
			f.savefig(ResultsPath + "\\" + f"corners_{i}{iter}.png", bbox_inches="tight") 
			# plt.show()

			"""
			Perform ANMS: Adaptive Non-Maximal Suppression
			Save ANMS output as anms.png
			"""
			num_features = 1000
			corners_anms = ANMS(corners, num_features)


			#appending it into a list
			best_corners.append(corners_anms)

			f, axarr = plt.subplots(1,2)
			f.suptitle("Corner Detection after ANMS", y=0.8)
			axarr[0].title.set_text("Input")
			axarr[0].set_xticks([])
			axarr[0].set_yticks([])
			axarr[1].title.set_text("Output")
			axarr[1].set_xticks([])
			axarr[1].set_yticks([])

			# Original 
			axarr[0].imshow(corner_img)

			# Get image with corner overlay
			corner_img_ANMS = np.copy(img)
			for coordinates in corners_anms:
				x, y = coordinates.ravel()
				cv2.circle(corner_img_ANMS, (x,y), 3, 255, -1)
			axarr[1].imshow(corner_img_ANMS)
			f.savefig(ResultsPath + "\\" + f"anms_{i}{iter}.png", bbox_inches="tight") 
			# plt.show()


			"""
			Feature Descriptors
			Save Feature Descriptor output as FD.png
			"""
			feature_descriptors = GetFeatureDescriptors(img_gray, corners_anms)
			#apending feature descriptors into a list
			feature_vectors.append(feature_descriptors)
			# x_dim = num_features // 5
			# y_dim = 5
			# f, axarr = plt.subplots(x_dim, y_dim)
			# f.suptitle("Feature Descriptors as 8x8 Images")
			# count = 0
			# for j in range(x_dim):
			# 	for k in range(y_dim):
			# 		axarr[j,k].set_xticks([])
			# 		axarr[j,k].set_yticks([])
			# 		axarr[j,k].imshow(feature_descriptors[count].reshape((8,8)))
			# 		count += 1
			# f.savefig(ResultsPath + "\\" + f"FD_{i}{iter}.png", bbox_inches="tight")

		#creating Homography matrix list, inlier count and Ransac match methods.
		H_matrixList = []
		Inlier_count = []
		Ransac_match = []

		#performing Ransac match and inlier count of image 1 with remaining images, 
		for i in range(1,len(images)):
			
			#creating image pairs, feature vector pairs and best corner pairs
			image_pair = [images[0],images[i]]
			feature_vector_pair = [feature_vectors[0],feature_vectors[i]]
			corner_pairs = [best_corners[0], best_corners[i]]
			"""
			Feature Matching
			Save Feature Matching output as matching.png
			"""
			matched_features = MatchingFeatures(feature_vector_pair, corner_pairs)
			drawmatches = DrawMatches(image_pair,matched_features)
			# cv2.imshow('featurematch', drawmatches)
			cv2.imwrite(ResultsPath + "\\" + f"matching{i}{iter}.png", drawmatches)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			#Some image pairs might not have feautres matched between them. 
			#creating a condition so that ransac won't return an error
			#continuing to next image pair if there are no features.
			if len(matched_features)>4:
				""" 
				Refine: RANSAC, Estimate Homography 
				"""
				ransac_match, h_matrix, inlier_count = Ransac(matched_features)
				drawmatches = DrawMatches(image_pair,ransac_match)
				cv2.imwrite(ResultsPath + "\\" + f"matching_ransac{i}{iter}.png", drawmatches)
				# cv2.imshow('ransac', drawmatches)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
			else:
				if iter == length-2:
					break
				else:
					Inlier_count.append(0)
					H_matrixList.append(0)
					Ransac_match.append(0)

					continue
			#appending inlier count and Homography
			Inlier_count.append(inlier_count)
			H_matrixList.append(h_matrix)
			Ransac_match.append(ransac_match)
		
		#condition to avoid exceptions when all the remaining images are not a match.
		if len(Inlier_count)>0:
			#taking image with highest inlier count and stitching the image with it.
			if max(Inlier_count)!=0:
				sort_index = np.argsort(Inlier_count)
				max_index = sort_index[-1]
				image_pair = [images[0], images[max_index+1]]
				h_matrix = H_matrixList[max_index]
				ransac_match = Ransac_match[max_index]
				"""
				Image Warping + Blending
				Save Panorama output as mypano.png
				"""
				output = ImageStitch(image_pair, h_matrix)
				cv2.imshow('output',output)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
				cv2.imwrite(ResultsPath + "\\" + f"mypartialpano{iter}.png", images[0])
				#stitching the image, removing the stitched image pair from images list 
				#and adding the stitched image to images[0] so that the stitched image can be blended again.
				images.pop(max_index+1)
				images.pop(0)
				images.append(output)
				images.reverse()
				best_corners.clear()
				feature_vectors.clear()
				H_matrixList.clear()
				Inlier_count.clear()
				Ransac_match.clear()
		

	output_img = images[0]
	cv2.imshow('output',output_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imwrite(ResultsPath + "\\" + f"mypano.png", images[0])
	return output_img



def main():
	# Add any Command Line arguments here
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--TrainPath', default="Set1", help='Number of best features to extract from each image, Default:100')

	Args = Parser.parse_args()
	key = Args.TrainPath

	# key = "Set3"
	basePath = dirname(dirname(abspath(__file__)))
	resultsPath = basePath + f"\\Code\\Results\\{key}"
	
	trainPath = basePath + "\\Data\\Train"
	testPath = basePath + "\\Data\\Test"
	trainFolderNames = {"Custom1": trainPath + "\\CustomSet1",
					    "Custom2": trainPath + "\\CustomSet2",
						"Set1":	trainPath + "\\Set1",
						"Set2": trainPath + "\\Set2",
						"Set3": trainPath + "\\Set3"}
	testFolderNames = {"TestSet1": testPath + "\\TestSet1",
					    "TestSet2": testPath + "\\TestSet2",
						"TestSet3":	testPath + "\\TestSet3",
						"TestSet4": testPath + "\\TestSet4"}

	panorama = MyAutoPano(testFolderNames[key], resultsPath)


if __name__ == "__main__":
	main()
