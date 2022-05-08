##Loading the necessary packages 
import cv2
import pytesseract
from gtts import gTTS
import os
from matplotlib import pyplot as plt
# image_file = "images/no_parking.jpg"
image_file = "images/stop.jpg"
# image_file = "images/images.jpg"
# image_file = "images/no_parking.jpg"
# image_file = "text.jpg"
# image_file = "text.jpg"
# image_file = "text.jpg"

img = cv2.imread(image_file)


#							PREPROCESSING OF THE IMAGE


def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    height, width  = im_data.shape[:2]
    
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()

display(image_file) 

# Inverted images .. say negative of the image

inverted_image = cv2.bitwise_not(img)  # for inverting the image

#saving
cv2.imwrite("temp/inverted.jpg" , inverted_image)
display("temp/inverted.jpg") 

# rescaling

#Binarization (convert a image into black and white but for it we want image to be in greyscale)
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


gray_image = grayscale(img)


cv2.imwrite("temp/gray.jpg" , gray_image)
display("temp/gray.jpg")

# #now

# binarizethresh, im_bw = cv2.threshold(gray_image, 120, 150, cv2.THRESH_BINARY)
# cv2.imwrite("temp/bw_image.jpg" , im_bw)
# display("temp/bw_image.jpg") 


#Noise removal
def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)

no_noise = noise_removal(gray_image)


# no_noise = noise_removal(gray_image)
cv2.imwrite("image_with_border.jpg" , no_noise)
display("image_with_border.jpg")
# cv2.imwrite("temp/no_noise.jpg", no_noise)
# display("temp/no_noise.jpg") 

# Dilation and Erosion 
# for thinning and thickining the font
def thin_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)


eroded_image = thin_font(no_noise)
cv2.imwrite("image_with_border.jpg" , eroded_image)
display("image_with_border.jpg")



# cv2.imwrite("temp/eroded_image.jpg", eroded_image)
# display("temp/eroded_image.jpg")

def thick_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1) #expansion of the font
    image = cv2.bitwise_not(image)
    return (image)


dilated_image = thick_font(no_noise)
cv2.imwrite("image_with_border.jpg" , dilated_image)
# display("image_with_border.jpg")




# cv2.imwrite("temp/dilated_image.jpg", dilated_image)
# display("temp/dilated_image.jpg")


# Rotation / Deskewing (when the image is pivioted sideways)
# new = cv2.imread("temp/1.jpg")
# display("temp/1.jpg")
# import numpy as np

# def getSkewAngle(cvImage) -> float:
#     # Prep image, copy, convert to gray scale, blur, and threshold
#     newImage = cvImage.copy()
#     gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (9, 9), 0)
#     thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

#     # Apply dilate to merge text into meaningful lines/paragraphs.
#     # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
#     # But use smaller kernel on Y axis to separate between different blocks of text
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
#     dilate = cv2.dilate(thresh, kernel, iterations=2)

#     # Find all contours
#     contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key = cv2.contourArea, reverse = True)
#     for c in contours:
#         rect = cv2.boundingRect(c)
#         x,y,w,h = rect
#         cv2.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)

#     # Find largest contour and surround in min area box
#     largestContour = contours[0]
#     print (len(contours))
#     minAreaRect = cv2.minAreaRect(largestContour)
#     cv2.imwrite("temp/boxes.jpg", newImage)
#     # Determine the angle. Convert it to the value that was originally used to obtain skewed image
#     angle = minAreaRect[-1]
#     if angle < -45:
#         angle = 90 + angle
#     return -1.0 * angle
# # Rotate the image around its center
# def rotateImage(cvImage, angle: float):
#     newImage = cvImage.copy()
#     (h, w) = newImage.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     return newImage

# def deskew(cvImage):
#     angle = getSkewAngle(cvImage)
#     return rotateImage(cvImage, -1.0 * angle)
# fixed = deskew(dilated_image)
# cv2.imwrite("temp/rotated_fixed.jpg", fixed)
# display("temp/rotated_fixed.jpg")


# removing border
def remove_borders(image):
    contours, heiarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)


no_borders = remove_borders(no_noise)

cv2.imwrite("temp/no_borders.jpg", no_borders)
# display('temp/no_borders.jpg')

cv2.imwrite("image_with_border.jpg" , no_borders)

#Missing borders
# color = [255, 255, 255]
# top, bottom, left, right = [150]*4
# image_with_border = cv2.copyMakeBorder(no_borders, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
# cv2.imwrite("image_with_border.jpg", image_with_border)
# display("temp/image_with_border.jpg")


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import pytesseract
from matplotlib import pyplot as plt

# ans = []
ans2 = []
#Give location of the image to be read.
#"Example-images/ex24.jpg" image is being loaded here. 

args = {"image":"image_with_border.jpg", "east":"frozen_east_text_detection.pb","min_confidence":0.5, "width":320, "height":320}
# args = {'image':'2.jpg' , 'width':3120,'height':4160}

# args['image']="2.jpg"
# image_file = "2.jpg"
image = cv2.imread(args["image"])

# image = image.fromarray(args["image"])

# image = cv2.imread(image_file)


#Saving a original image and shape
orig = image.copy()
(origH, origW) = image.shape[:2]

# set the new height and width to default 320 by using args #dictionary.  
(newW, newH) = (args["width"], args["height"])

#Calculate the ratio between original and new image for both height and weight. 
#This ratio will be used to translate bounding box location on the original image. 
rW = origW / float(newW)
rH = origH / float(newH)

# resize the original image to new dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# construct a blob from the image to forward pass it to EAST model
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)


# load the pre-trained EAST model for text detection 
net = cv2.dnn.readNet(args["east"])

# We would like to get two outputs from the EAST model. 
#1. Probabilty scores for the region whether that contains text or not. 
#2. Geometry of the text -- Coordinates of the bounding box detecting a text
# The following two layer need to pulled from EAST model for achieving this. 
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]




#Forward pass the blob from the image to get the desired output layers
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)


## Returns a bounding box and probability score if it is more than minimum confidence
def predictions(prob_score, geo):
	(numR, numC) = prob_score.shape[2:4]
	boxes = []
	confidence_val = []

	# loop over rows
	for y in range(0, numR):
		scoresData = prob_score[0, 0, y]
		x0 = geo[0, 0, y]
		x1 = geo[0, 1, y]
		x2 = geo[0, 2, y]
		x3 = geo[0, 3, y]
		anglesData = geo[0, 4, y]

		# loop over the number of columns
		for i in range(0, numC):
			if scoresData[i] < args["min_confidence"]:
				continue

			(offX, offY) = (i * 4.0, y * 4.0)

			# extracting the rotation angle for the prediction and computing the sine and cosine
			angle = anglesData[i]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# using the geo volume to get the dimensions of the bounding box
			h = x0[i] + x2[i]
			w = x1[i] + x3[i]

			# compute start and end for the text pred bbox
			endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
			endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
			startX = int(endX - w)
			startY = int(endY - h)

			boxes.append((startX, startY, endX, endY))
			confidence_val.append(scoresData[i])

	# return bounding boxes and associated confidence_val
	return (boxes, confidence_val)


	# Find predictions and  apply non-maxima suppression
(boxes, confidence_val) = predictions(scores, geometry)
boxes = non_max_suppression(np.array(boxes), probs=confidence_val)


##Text Detection and Recognition 

# initialize the list of results
results = []

# loop over the bounding boxes to find the coordinate of bounding boxes
for (startX, startY, endX, endY) in boxes:
	# scale the coordinates based on the respective ratios in order to reflect bounding box on the original image
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	#extract the region of interest
	r = orig[startY:endY, startX:endX]

	#configuration setting to convert image to string.  
	configuration = ("-l eng --oem 1 --psm 8")

    ##This will recognize the text from the image of bounding box
	text = pytesseract.image_to_string(r, config=configuration)


	# temp = text
	# temp.rstrip('\n\x00')
	# ans.append(temp)
	# append bbox coordinate and associated text to the list of results 


	results.append(((startX, startY, endX, endY), text))


	#Display the image with bounding box and recognized text
orig_image = orig.copy()

# Moving over the results and display on the image
for ((start_X, start_Y, end_X, end_Y), text) in results:
	# display the text detected by Tesseract
	# print("{}\n".format(text))

	# Displaying text
	text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
	ans2.append(text)
	cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y),
		(0, 0, 255), 2)
	cv2.putText(orig_image, text, (start_X, start_Y - 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0, 255), 2)

plt.imshow(orig_image)
plt.title('Output')
plt.show()

# print(ans2)


textfile = open("text.txt", "w")
textfile.truncate(0)
textfile = open("text.txt", "w")
for element in ans2:
    textfile.write(element + "\n")
textfile.close()

fh = open("text.txt","r")
n = int(input("enter 1 to play sound and 0 if not. "))
if n==1 :
	print("now speaking")
	myText = fh.read().replace("\n","")
	language = 'en'
	output = gTTS(text=myText, lang =language, slow=False)
	output.save("output.mp3")
	# fh.close()
	os.system("start output.mp3")
print("Thank You")