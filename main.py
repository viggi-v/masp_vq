import binascii
import numpy as np
import matplotlib.pylab as plt
from PIL import Image
from matplotlib import pyplot as MPL
import codebook as cb

for i in range(8,0,-1):

	block_size = 2

	codebooks,threads, width, height, original_image = cb.create_codebook(str(i)+".png",4,block_size)

	cb.plot_pca(np.array(threads),np.array(codebooks),"output"+str(i)+".png")
	
	if(i == 1):
		cs = cb.get_compressed_string(np.array(codebooks),threads,block_size, height)
		compressed_string = "".join(chr(int("".join(map(str,cs[i:i+8])),2)) for i in range(0,len(cs),8))
		print "Length of compressed string for image 8: ",len(compressed_string)
		text_file = open("output_image.txt", "w")
		text_file.write(compressed_string)
		text_file.close()
		np.savetxt('codebooks.txt', codebooks)
	#print compressed_string
	'''
	#np.array2string(codebook) + "@@@@@"
	'''
	compressed_image = np.zeros([width,height])
	
	k = 0
	j = 0
		
	print "compressing image number ",i
		
	for thread in threads:

		distances = [np.linalg.norm(centroid - thread) for centroid in codebooks]

		compressed_image[k:k+block_size,j:j+block_size] = codebooks[distances.index(min(distances))].reshape([block_size,block_size])
		
		bin_value = "{0:04b}".format(distances.index(min(distances)))
		
		#compressed_string += bin_value
		
		# not proud of this shit	
		j = j+block_size
		if j >= height:
			k = k+block_size
			j = j - height
	
	img = Image.fromarray(compressed_image*255)
	img = img.convert('RGB')		
	img.save('compressed_'+str(i)+'.png')

