import numpy as np
import matplotlib.pylab as plt
from PIL import Image
from matplotlib import pyplot as MPL

g_codebooks = []

histCount = 0

def to_grayscale(im, weights = np.c_[0.2989, 0.5870, 0.1140]):
    tile = np.tile(weights, reps=(im.shape[0],im.shape[1],1))
    return np.sum(tile * im, axis=2)

def threshold_otsu_from_array(arr, nbins=256):
   
    hist, bin_centers = np.histogram(arr, nbins)
    hist = hist.astype(float)
    # print "histo shape ",hist.shape
    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    # print "bin shape ", bin_centers.shape, " weight1 shape ", weight1.shape
    mean1 = np.cumsum(hist * bin_centers[0:256]) / weight1
    mean2 = (np.cumsum((hist * bin_centers[0:256])[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold

def get_virtual_centroid(np_threads):
	#print " shape of np_threads ", np_threads.shape
	mean = np_threads.mean(axis=0)
	
	width = np_threads.shape[1]
	
	threshold_vector = []
	
	for i in range(0, width):
		# print(np_threads[:,i])
		threshold_vector.append(threshold_otsu_from_array(np_threads[:,i]))
	threshold_vector = np.array(threshold_vector)
	#print threshold_vector, mean
	return(mean,np.subtract(2**threshold_vector, mean))
	
def split(threads, centroid, virtual_centroid):
	# do split
	left = []
	right = []
	for thread in threads:
		dist1 = np.linalg.norm(centroid - thread) # distance between centroid and thread
		dist2 = np.linalg.norm(virtual_centroid - thread)
		if dist1 > dist2:
			left.append(thread)
		else:
			right.append(thread)
	left = np.array(left)
	right = np.array(right)
	return(left,right)

def _create_codebook(level,max_quantization,np_threads):
	
	centroid, virtual_centroid = get_virtual_centroid(np_threads)
	
	if(level > max_quantization):
		g_codebooks.append(centroid)
		return
	
	left, right = split(np_threads, centroid, virtual_centroid)

	if left.shape[0] is 0 or right.shape[0] is 0:
		g_codebooks.append(centroid)
	else:
		_create_codebook(level+1, max_quantization,left)
		_create_codebook(level+1, max_quantization,right)

def create_codebook(image,max_quantization = 4, block_size = 4):
	
	im = plt.imread(image)

	image = to_grayscale(im)

	[width,height] = image.shape
	threads = []

	for i in range(0, width,block_size):
		for j in range(0, height,block_size):
			threads.append(image[i:i+block_size,j:j+block_size].ravel())

	np_threads = np.array(threads)

	_create_codebook(0,max_quantization,np_threads)

	return (g_codebooks,np_threads, width, height, image)

def PCA(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    from scipy import linalg as LA
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = LA.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(evecs.T, data.T).T, evals, evecs

def test_PCA(data, dims_rescaled_data=2):
    '''
    test by attempting to recover original data array from
    the eigenvectors of its covariance matrix & comparing that
    'recovered' array with the original data
    '''
    _ , _ , eigenvectors = PCA(data, dim_rescaled_data=2)
    data_recovered = np.dot(eigenvectors, m).T
    data_recovered += data_recovered.mean(axis=0)
    assert np.allclose(data, data_recovered)

def plot_pca(pixels, codebook, name):
    clr1 = '#0000FF'
    clr2 = '#FF0000'
    fig = MPL.figure()
    ax1 = fig.add_subplot(111)
    data_resc, data_orig,_s = PCA(pixels)
    ax1.plot(data_resc[:, 0], data_resc[:, 1], '.', mfc=clr1, mec=clr1)
    
    data_resc, data_orig,_s = PCA(codebook)
    ax1.plot(data_resc[:, 0], data_resc[:, 1], '.', mfc=clr2, mec=clr2)
    fig.savefig(name)

def get_compressed_string(codebook, threads, block_size, height):
	bits = np.ceil(np.log2(codebook.shape[0]))
	'''
	if(bits > 4):
		print "Error"
		return ''
	'''
	#compressed_image = np.zeros([width,height])
	compressed_string = ''
	
	k = 0
	j = 0
		
	#print "compressing image number ",i
		
	for thread in threads:

		distances = [np.linalg.norm(centroid - thread) for centroid in codebook]

		#compressed_image[k:k+block_size,j:j+block_size] = codebooks[distances.index(min(distances))].reshape([block_size,block_size])
		
		bin_value = "{0:08b}".format(distances.index(min(distances)))
		
		compressed_string += bin_value
		
		# not proud of this shit	
		j = j+block_size
		if j >= height:
			k = k+block_size
			j = j - height
	
	return compressed_string