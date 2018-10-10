import cv2 as cv
import numpy as np


def detection(image):
	""" Detecting the words bounding boxes """
	# Preprocess image for word detection
	blurred = cv.GaussianBlur(image, (5, 5), 18)
	edgeImg = edgeDetect(blurred)
	ret, edgeImg = cv.threshold(edgeImg, 50, 255, cv.THRESH_BINARY)
	#bwImage = cv.morphologyEx(edgeImg, cv.MORPH_CLOSE, np.ones((15,15), np.uint8))
	# Return detected bounding boxes
	#return textDetect(bwImage, image, join)
	return edgeImg

def edgeDetect(im):
    """ 
    Edge detection 
    Sobel operator is applied for each image layer (RGB)
    """
    return np.max(np.array([sobelDetect(im[:,:, 0]),
                            sobelDetect(im[:,:, 1]),
                            sobelDetect(im[:,:, 2])]), axis=0)

def sobelDetect(channel):
    """ Sobel operator """
    sobelX = cv.Sobel(channel, cv.CV_16S, 1, 0)
    sobelY = cv.Sobel(channel, cv.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)
    sobel[sobel > 255] = 255
    return np.uint8(sobel)

def drawWord(binary_word,binary_img):
	height,width = binary_word.shape[:2]
	hist_word = np.zeros(width)
	for x in range(width):
		num = 0
		for y in range (height):
			if binary_word[y][x] != 0:
				num+=1
		hist_word[x]=num
		print(hist_word[x])
	binary_img = cv.cvtColor(binary_img,cv.COLOR_GRAY2BGR)
	
	list_pos = []
	start = 0


	for x in range(1,width):
		if x < start:
			continue
		if hist_word[x-1] == 0 and hist_word[x]!=0:
			cv.line(binary_img,(x,0),(x,height),(0,0,255),2) #head
			
			begin,end = x,x

			while hist_word[end] > 0:
				end += 1

			list_pos.append([begin,end])
			start = end

			cv.line(binary_img,(end,0),(end,height),(255,0,0),2) #tail






		# if hist_word[x-1] !=0 and hist_word[x] == 0:
		# 	cv.line(binary_img,(x,0),(x,height),(255,0,0),2) #tail
			
	return list_pos,binary_img

if __name__ == "__main__":
	src_img = cv.imread("1.jpg")
	width, height, _ = src_img.shape

	img = cv.cvtColor(src_img,cv.COLOR_BGR2RGB)
	
	#DETECT WORD
	binary = detection(img)
	
	binary_img = cv.morphologyEx(binary, cv.MORPH_CLOSE, np.ones((5,5), np.uint8))
	binary_word = cv.morphologyEx(binary, cv.MORPH_CLOSE, np.ones((15,15), np.uint8))
	
	list_pos,binary_img = drawWord(binary_word,binary_img)


	cv.imwrite('binary.jpg',binary_img)
	#imshow
	name_src = 'src'
	cv.namedWindow(name_src,cv.WINDOW_NORMAL)
	cv.resizeWindow(name_src, height, width)
	cv.imshow(name_src,binary_img)

	name_word = 'word'
	cv.namedWindow(name_word,cv.WINDOW_NORMAL)
	cv.resizeWindow(name_word, height, width)
	cv.imshow(name_word, binary_word)

	cv.waitKey(0)
