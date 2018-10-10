import cv2 as cv
import numpy as np
import os
import skimage 
from skimage import io

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
		#print(hist_word[x])
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


def show_display(img,cols,max_h,rank,list_range):
	cv.line(img,(cols-1,0),(cols - 1,max_h-1),(0,255,0),2)
	cv.putText(img, "{}".format(rank), (cols, int(max_h/2)),
				cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	print("pos " + str(rank) + " : " + str(cols) + "---", list_range[cols])
	
def detect_char(img, list_pos, MIN = 40,MAX = 85,min_cum = 3,min_rate_below = 1,max_rate_blow = 0.70):
	#path_img = os.path.join("detect_ocr","OCR","5.jpg")
	height,width = img.shape[:2]

	gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	ret,bin = cv.threshold(gray,150,255,cv.THRESH_BINARY)


	for _idx,pos in enumerate(list_pos):
		red_line, blue_line = pos[:2]
		
		crop_bin = bin[0:height, red_line:blue_line]
		crop_img = img[0:height,red_line:blue_line]
		# cv.imshow("cropped", crop_img)
		# 
		
		# #threshold
		
		
		h,w = crop_bin.shape[:2]
		
		for i in range(h):
			for j in range(w):
				crop_bin[i][j] = 0 if crop_bin[i][j] == 255 else 255
		
		list_range = []
		
		for i in range(w):
			num = 0
			for j in range(h):
				num += crop_bin[j][i]
			list_range.append(int(num/255))

		
		bin_color = cv.cvtColor(crop_bin,cv.COLOR_GRAY2BGR)
		begin = 0
		end = 0
		
		#MIN,MAX = 40,80
		
		#run out
		begin_new = 0
		end_new = len(list_range) - 1
		
		
		
		index_words = 0
		
		
		
		for idx in range(len(list_range)):
			if idx < begin_new:
				continue
			if idx > end_new:
				break
			#logic
			if idx == 0:
				while list_range[idx] <= min_cum:
					idx += 1
				
				index_words += 1
				
				show_display(bin_color,idx - 1,h,index_words,list_range)
				
				begin_new = idx
			
			#decrease MAX and MIN
			if index_words == 2:
				MAX = int(MAX*max_rate_blow)
				MIN = int(MIN*min_rate_below)
			#check range out of index
			if idx + MIN >= len(list_range):
				#index_words += 1
				#show_display(bin_color,w-1,h,index_words,list_range)
				#cv.line(bin_color,(w-3,0),(w-3,h-1),(0,255,0),1)
				break
			
			#estimate range words
			thresh_low = idx + MIN
			
			thresh_high = idx + MAX if idx + MAX <= len(list_range) - 1 else len(list_range)
			
			#fix range end
			if idx + MIN + MIN >= len(list_range) - 1 and 2*MIN < MAX:
				#print(thresh_low,'----',len(list_range) - MIN -1)
				thresh_high = len(list_range) - MIN - 1
				#num = min(list_range[thresh_low : len(list_range) - MIN - 1])
			
			
			#discard 
			if thresh_low >= thresh_high:
				break
			
			num = min(list_range[thresh_low:thresh_high])
			
			#cols  words
			xxx = list_range[thresh_low:thresh_high].index(num) + thresh_low
			
			index_words += 1
			show_display(bin_color,xxx,h,index_words,list_range)
			
			
			xxx += 1
			while list_range[xxx] <= min_cum:
				xxx += 1
				if xxx >= len(list_range) - 1:
					break
			begin_new = xxx
		drawing_window = 'binary'
		cv.namedWindow(drawing_window,cv.WINDOW_NORMAL)
		cv.resizeWindow(drawing_window, height, width)
		cv.imshow(drawing_window,bin_color)
		cv.waitKey(0)
		
	return


if __name__ == "__main__":
	src_img = cv.imread(os.path.join("test","1.jpg"))
	width, height = src_img.shape[:2]

	img = cv.cvtColor(src_img,cv.COLOR_BGR2RGB)
	
	#DETECT WORD
	binary = detection(img)
	
	binary_img = cv.morphologyEx(binary, cv.MORPH_CLOSE, np.ones((5,5), np.uint8))
	binary_word = cv.morphologyEx(binary, cv.MORPH_CLOSE, np.ones((15,15), np.uint8))
	
	list_pos,binary_img = drawWord(binary_word,binary_img)
	detect_char(img,list_pos)

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
