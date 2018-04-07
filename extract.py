import numpy as np
from PIL import Image


f = open("output_image.txt", 'rb')
# read one byte
pixels = []
byte = f.read(1)
while byte:
	# convert the byte to an integer representation
	byte = ord(byte) 
	# now convert to string of 1s and 0s
	byte = bin(byte)[2:].rjust(8, '0')
	# now byte contains a string with 0s and 1s
	MSB = str(byte[7])+ str(byte[6]) + str(byte[5]) + str(byte[4]) + str(byte[3])+ str(byte[2]) + str(byte[1]) + str(byte[0])
	# LSB = str(byte[3])+ str(byte[2]) + str(byte[1]) + str(byte[0])	
	# print LSB,MSB
	pixels.append(int(MSB,2))
	# pixels.append(int(LSB,2))
	byte = f.read(1)

codebooks = np.loadtxt('codebooks.txt')

compressed_image = np.zeros([256,256])
print codebooks.shape
print np.mean(np.array(pixels))
j = 0
k = 0
for pixel in pixels:
	#print pixel
	pixel = pixel%12
	compressed_image[k:k+2,j:j+2] = codebooks[pixel].reshape([2,2])
	j = j+2
	if j >= 256:
		k = k + 2
		j = j - 256
		
img = Image.fromarray(compressed_image*255)
img = img.convert('RGB')		
img.save('regained.png')
	
