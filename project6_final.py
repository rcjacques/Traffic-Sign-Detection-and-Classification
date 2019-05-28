import cv2, argparse, copy, os
import numpy as np 

#=============================================
winSize = (64,64)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (4,4)
nbins = 9
derivAperture = 1
winSigma = -1.0
histogramNormType = 0
L2HysThreshold = 1
gammaCorrection = 1
nlevels = 64
signedGradients = False

hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels,signedGradients)

# main_folder = 'TSR/Training/'
# sub_folders = ['00001/','00014/','00017/','00019/','00021/','00035/','00038/','00045/']

# hog_des = []
# labels = []
# for j in range(1,62):
# 	images = [img for img in os.listdir(main_folder+str(j).zfill(5)+'/') if img.endswith(".ppm")]
# 	for i in images:
# 		img = cv2.imread(main_folder+str(j).zfill(5)+'/'+i)
# 		img = cv2.resize(img,(64,64))
# 		# cv2.imshow('img',img)
# 		# cv2.waitKey(1)
# 		descriptor = hog.compute(img)
# 		hog_des.append(descriptor)
# 		labels.append(int(str(j).zfill(5)))
		
# hog_des = np.squeeze(hog_des)

svm = cv2.ml.SVM_create()
# svm.setType(cv2.ml.SVM_C_SVC)
# svm.setKernel(cv2.ml.SVM_LINEAR)
# svm.setC(1)
# svm.setGamma(0)
# svm.train(hog_des,cv2.ml.ROW_SAMPLE,np.array(labels))
# svm.save('data.csv')
svm = svm.load('data.csv')
#=============================================

max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
def on_low_H_thresh_trackbar(val):
	global low_H
	global high_H
	low_H = val
	low_H = min(high_H-1, low_H)
	cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
	global low_H
	global high_H
	high_H = val
	high_H = max(high_H, low_H+1)
	cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
	global low_S
	global high_S
	low_S = val
	low_S = min(high_S-1, low_S)
	cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
	global low_S
	global high_S
	high_S = val
	high_S = max(high_S, low_S+1)
	cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
	global low_V
	global high_V
	low_V = val
	low_V = min(high_V-1, low_V)
	cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
	global low_V
	global high_V
	high_V = val
	high_V = max(high_V, low_V+1)
	cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)

def blue_blob_detector():
	params = cv2.SimpleBlobDetector_Params()

	# Change thresholds
	# params.minThreshold = 50
	# params.maxThreshold = 200

	params.minDistBetweenBlobs = 100
	# # Filter by Color
	# params.filterByColor = 1
	# params.blobColor = 255
	# # Filter by Area.
	params.filterByArea = True
	params.minArea = 100
	params.maxArea = 10000000000000
	# Filter by Circularity
	params.filterByCircularity = True
	params.minCircularity = 0
	params.maxCircularity = 0.99
	# Filter by Convexity
	params.filterByConvexity = True
	params.minConvexity = 0
	params.maxConvexity = 1
	detector = cv2.SimpleBlobDetector_create(params)
	return detector

def red_blob_detector():
	params = cv2.SimpleBlobDetector_Params()

	# Change thresholds
	# params.minThreshold = 50
	# params.maxThreshold = 200

	params.minDistBetweenBlobs = 100
	# # Filter by Color
	# params.filterByColor = 1
	# params.blobColor = 255
	# # Filter by Area.
	params.filterByArea = True
	params.minArea = 100
	params.maxArea = 10000000000000
	# Filter by Circularity
	params.filterByCircularity = True
	params.minCircularity = 0
	params.maxCircularity = 0.99
	# Filter by Convexity
	params.filterByConvexity = True
	params.minConvexity = 0
	params.maxConvexity = 1
	detector = cv2.SimpleBlobDetector_create(params)
	return detector

parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera devide number.', default=0, type=int)
args = parser.parse_args()

# cv2.namedWindow(window_capture_name)
# cv2.namedWindow(window_detection_name)
# cv2.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
# cv2.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
# cv2.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
# cv2.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
# cv2.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
# cv2.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
# detector = cv2.SimpleBlobDetector_create()
blue_detector = blue_blob_detector()

font = cv2.FONT_HERSHEY_SIMPLEX

video_name = 'p6_output_15.mp4'
fps = 15
size = (1628,1236)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_name,fourcc,fps,size)

f = 32640
# f = 34350
# f = 33200
fast_forward = True
rewind = False
# for f in range(33400,35500):
while f < 35500:
	keypress = cv2.waitKey(1)
	if keypress == ord('d'):
		if f+1 < 35500:
			f += 1
		# print(f-32640)
		rewind = False
		fast_forward = False
	elif keypress == ord('a'):
		if f-1 >= 32640:
			f -= 1
		# print(f-32640)
		rewind = False
		fast_forward = False
	elif keypress == ord('w'):
		fast_forward = True
		rewind = False
	elif keypress == ord('s'):
		fast_forward = False
		rewind = True
	else:
		if fast_forward:
			if f+1 <= 35500:
				f += 1
				print(f-32640)
		elif rewind:
			if f-1 >= 32640:
				f -= 1
				# print(f-32640)
		else:
			pass
	# print(f-32640)
	frame_name = "TSR/input/image.0"+str(f)+".jpg"
	frame = cv2.imread(frame_name)
	# print(frame.shape)
	frame = cv2.blur(frame,(5,5))
	black_frame = copy.copy(frame)
	black_frame[800:,:,:]=[0,0,0]
	frame_HSV = cv2.cvtColor(black_frame, cv2.COLOR_BGR2HSV)
	frame_RGB = cv2.cvtColor(black_frame, cv2.COLOR_RGB2HSV)
	mask_blue = cv2.inRange(frame_HSV, (52, 129, 0), (180, 255, 255))
	frame_blue = cv2.bitwise_and(frame,frame,mask=mask_blue)
	frame_blue_gray = cv2.cvtColor(frame_blue,cv2.COLOR_BGR2GRAY)
	_,contours, _ = cv2.findContours(frame_blue_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# print('contours ',len(contours))

	for cnt in contours:
		area = cv2.contourArea(cnt)
		approx = cv2.approxPolyDP(cnt,0.02*cv2.arcLength(cnt, True), True)
		if area > 300:# and len(approx) < 6:
			frame_blue = cv2.drawContours(frame_blue,[approx],0,(0,255,0),5)
			x,y,w,h = cv2.boundingRect(cnt)
			if w < 1.2*h:
				box_img = frame[y:y+h,x:x+w]
				box_img = cv2.resize(box_img,(64,64))
				box_des = hog.compute(box_img)
				# descriptors.append(box_des)
				# cv2.imshow('',box_img)
				# cv2.waitKey(0)
				box_des = np.array([np.squeeze(box_des)])
				# print(len(box_des))
				_,response = svm.predict(box_des)
				response = str(response)[2:-3]
				sign = cv2.imread('TSR/Display_Signs/'+response+'.ppm')
				if response == '45' or response == '38' or response == '35':
					# print(area,len(approx))
					# cv2.imshow('green',sign)
					# print(sign.shape)
					try:
						frame[y:y+h,x-w:x] = cv2.resize(sign,(w,h))
					except:
						pass
					cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
					# scale = 1
					# fontScale =  min(w,h)/(35/scale)
					# cv2.putText(frame,response,(x-int(3*w/2),y+h),font,fontScale,(0,255,0),4,cv2.LINE_AA)
				# else:
				# 	cv2.imshow('red',sign)
				# 	print(sign.shape)
				# 	try:
				# 		frame[y:y+h,x-w:x] = cv2.resize(sign,(w,h)) 
				# 	except:
				# 		pass            
				# 	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
				# 	scale = 1
				# 	fontScale =  min(w,h)/(35/scale)
				# 	cv2.putText(frame,response,(x-int(3*w/2),y+h),font,fontScale,(0,0,255),4,cv2.LINE_AA)
				# cv2.waitKey(0)

	# mask_red = cv2.inRange(frame_RGB, (low_H,low_S,low_V),(high_H,high_S,high_V))
	mask_red = cv2.inRange(frame_HSV, (0, 84, 0), (23, 255, 255))
	# mask_red = cv2.inRange(frame_HSV, (115, 0, 0), (180, 255, 104))

	frame_red = cv2.bitwise_and(frame,frame,mask=mask_red)
	frame_red_gray = cv2.cvtColor(frame_red,cv2.COLOR_BGR2GRAY)
	_,countours, _ = cv2.findContours(frame_red_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for cnt in countours:
		M = cv2.moments(cnt)
		
		area = cv2.contourArea(cnt)
		approx = cv2.approxPolyDP(cnt,0.02*cv2.arcLength(cnt, True), True)
		if area > 300:# and len(approx) < 6:
			frame_red = cv2.drawContours(frame_red,[approx],0,(0,255,0),5)
			x,y,w,h = cv2.boundingRect(cnt)
			if M['m00'] != 0:
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
				cv2.circle(frame_red,(cx,cy),5,(0,0,255),-1)
				centroid_dist = cy-y

			if w < 1.2*h and h < 1.2*w:
				box_img = frame[y:y+h,x:x+w]
				box_img = cv2.resize(box_img,(64,64))
				box_des = hog.compute(box_img)
				# descriptors.append(box_des)
				# cv2.imshow('',box_img)
				# cv2.waitKey(0)
				box_des = np.array([np.squeeze(box_des)])
				# print(len(box_des))
				_,response = svm.predict(box_des)
				response = str(response)[2:-3]
				sign = cv2.imread('TSR/Display_Signs/'+response+'.ppm')
				
				# finds triangles that are rightside up
				if centroid_dist > 2*h/3.5 and centroid_dist<2*h/2.5 and cx > 0.8*(x+w/2) and cx < 1.2*(x+w/2) and len(approx) <= 7:
					if response == '17' or response == '1' or response == '14':
						cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
						scale = 1
						fontScale =  min(w,h)/(35/scale)
						cv2.putText(frame,response,(x-int(3*w/2),y+h),font,fontScale,(0,255,0),4,cv2.LINE_AA)
						try:
							frame[y:y+h,x-w:x] = cv2.resize(sign,(w,h))
						except:
							pass
					# else:
						# scale = 1
						# fontScale =  min(w,h)/(35/scale)
						# cv2.putText(frame,response,(x-int(3*w/2),y+h),font,fontScale,(0,0,255),4,cv2.LINE_AA)
				# finds triangles that are inverted
				if centroid_dist > h/3.5 and centroid_dist<h/2.5 and cx > 0.8*(x+w/2) and cx < 1.2*(x+w/2) and len(approx) <= 4:
					if response == '19':
						# print(len(approx))
						cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
						# scale = 1
						# fontScale =  min(w,h)/(35/scale)
						# cv2.putText(frame,response,(x-int(3*w/2),y+h),font,fontScale,(0,255,0),4,cv2.LINE_AA)
						try:
							frame[y:y+h,x-w:x] = cv2.resize(sign,(w,h))
						except:
							pass
				# finds stop signs
				# if response == '21':
				# 	# cv2.imshow('green',sign)
				# 	# print(sign.shape)
				# 	# try:
				# 	# 	frame[y:y+h,x-w:x] = cv2.resize(sign,(w,h))
				# 	# except:
				# 	# 	pass
				# 	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
				# 	scale = 1
				# 	fontScale =  min(w,h)/(35/scale)
				# 	cv2.putText(frame,response,(x-int(3*w/2),y+h),font,fontScale,(0,255,0),4,cv2.LINE_AA)
			# print(np.shape(approx))



	mask_red_RGB = cv2.inRange(frame_RGB, (115, 0, 0), (180, 255, 104))
	mask_RGB = cv2.inRange(frame, (45, 45, 45), (255, 255, 255))
	frame_red_RGB = cv2.bitwise_and(frame,frame,mask=mask_red_RGB)
	frame_RGB = cv2.bitwise_and(frame,frame,mask=mask_RGB)
	frame_red_RGB = cv2.bitwise_and(frame_red_RGB,frame_red_RGB,mask = mask_RGB)
	frame_red_gray_RGB = cv2.cvtColor(frame_red_RGB,cv2.COLOR_BGR2GRAY)
	_,countours, _ = cv2.findContours(frame_red_gray_RGB, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for cnt in countours:
		M = cv2.moments(cnt)
		area = cv2.contourArea(cnt)
		approx = cv2.approxPolyDP(cnt,0.02*cv2.arcLength(cnt, True), True)
		if area > 500 and area < 2000 and len(approx) > 10:# and len(approx) < 6:
			frame_red_RGB = cv2.drawContours(frame_red_RGB,[approx],0,(0,255,0),5)
			x,y,w,h = cv2.boundingRect(cnt)
			if M['m00'] != 0:
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
				cv2.circle(frame_red,(cx,cy),5,(0,0,255),-1)
				centroid_dist = cy-y
			
			if w < 1.2*h and h < 1.2*w:
				box_img = frame[y:y+h,x:x+w]
				box_img = cv2.resize(box_img,(64,64))
				box_des = hog.compute(box_img)
				# descriptors.append(box_des)
				# cv2.imshow('',box_img)
				# cv2.waitKey(0)
				box_des = np.array([np.squeeze(box_des)])
				# print(len(box_des))
				_,response = svm.predict(box_des)
				response = str(response)[2:-3]
				if response == '21':
					# print(len(approx),area)
					sign = cv2.imread('TSR/Display_Signs/'+response+'.ppm')
					# cv2.imshow('green',sign)
					# print(sign.shape)
					try:
						frame[y:y+h,x-w:x] = cv2.resize(sign,(w,h))
					except:
						pass
					cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
					# scale = 1
					# fontScale =  min(w,h)/(35/scale)
					# cv2.putText(frame,response,(x-int(3*w/2),y+h),font,fontScale,(0,255,0),4,cv2.LINE_AA)
			# print(np.shape(approx))
	
	# cv2.imshow('frame',cv2.resize(frame_red,(0,0),fx=0.5,fy=0.5))
	comb = np.concatenate((frame,frame_blue),axis =1)
	# cv2.imshow('test',frame_red_RGB[:,:,2])
	cv2.imshow('frame',cv2.resize(frame,(0,0),fx=0.5,fy=0.5))

	# out.write(frame)
out.release()
	
	# cv2.waitKey(0)