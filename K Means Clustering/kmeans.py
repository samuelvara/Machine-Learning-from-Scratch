import numpy as np

#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: a list of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    '''
    p = generator.randint(0, n) #this is the index of the first center
    #############################################################################
    # TODO: implement the rest of Kmeans++ initialization. To sample an example
	# according to some distribution, first generate a random number between 0 and
	# 1 using generator.rand(), then find the the smallest index n so that the 
	# cumulative probability from example 1 to example n is larger than r.
    #############################################################################
    centers = [p]
    for _ in range(n_cluster-1):
        d = []
        for i in range(len(x)):
            curr = float('inf')
            for c in centers:
                curr = min(curr, np.sum((x[i] - x[c])**2))
            d.append(curr)
        
        base = sum(d)
        d = [x/base for x in d]
        cum_prob = 0
        r = generator.rand()
        for i in range(len(d)):
            cum_prob+=d[i]
            if cum_prob>r:
                break
        centers.append(i)

    # DO NOT CHANGE CODE BELOW THIS LINE
    return centers


# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)



class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''
    def __init__(self, n_cluster=0, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def find_kmean_obj(self, centroids, gammank, x):
        return np.sum([np.sum( (centroids[k] - x[gammank==k])**2) for k in range(self.n_cluster)])
    
    def find_Gammank(self, x, centroids):
        return np.argmin(np.sum(((x - np.expand_dims(centroids, axis=1))**2), axis=2), axis=0)

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array, 
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0), 
                  - number of times you update the assignment, an Int (at most self.max_iter)
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        ###################################################################
        # TODO: Update means and membership until convergence 
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        ###################################################################
        gammank = np.zeros(N) # Gamma nk with index denoting the membership
        centroids = np.array([x[c] for c in self.centers])
        print(centroids.shape, x.shape)
        # gamma objective = Sigma_n Sigma_k II[Gamma(n,k)==1]  (xn - centeroid)**2
        kmeanobj = self.find_kmean_obj(centroids, gammank, x)
        for iter in range(self.max_iter + 1):
            #From Alternate Minimization from Slide 14
            #Gammank_t+1 = argmin(Kmeanobj at Centeroid k)
            gammank = self.find_Gammank(x, centroids)
            new_kmeanobj = self.find_kmean_obj(centroids, gammank, x)
            if abs(new_kmeanobj - kmeanobj) <= self.e:
                break
            #New centroids from Slide 17 - (Sigma Gammank * xn) / Sigma Gammank
            new_centroids = np.array([np.mean(x[gammank==k], axis=0) for k in range(self.n_cluster)])
            # if Cluster is empty, then the new-centroid should just take on the previous centroid at the same index
            new_centroids[np.where(np.isnan(new_centroids))] = centroids[np.where(np.isnan(new_centroids))]
            centroids = new_centroids
        self.max_iter = iter
        return centroids, gammank, self.max_iter


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        ################################################################
        # TODO:
        # - assign means to centroids (use KMeans class you implemented, 
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        ################################################################
        centroids, assigned, _ = KMeans(self.n_cluster, self.max_iter, self.e, self.generator).fit(x, centroid_func)
        temp = [[] for i in range(self.n_cluster)]
        for i in range(N):
            temp[assigned[i]].append(y[i])
        centroid_labels = np.array([np.argmax(np.bincount(temp[i])) for i in range(self.n_cluster)])
        
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        ##########################################################################
        # TODO:
        # - for each example in x, predict its label using 1-NN on the stored 
        #    dataset (self.centroids, self.centroid_labels)
        ##########################################################################
        dist_from_cluster_centers = np.zeros([self.n_cluster, N])
        for i, centroid in enumerate(self.centroids):
            dist_from_cluster_centers[i] = np.sum((x-centroid)**2, axis=1)
        cluster_membership = np.argmin(dist_from_cluster_centers, axis=0)
        labels = []
        for i in range(len(x)):
            labels.append(self.centroid_labels[cluster_membership[i]])
        return np.array(labels)




def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    ##############################################################################
    # TODO
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    ##############################################################################
    #Since Code vector in Shape (?,3) we need to convert image to shape (?, 3) which 
    #means we have reshape into 2D array.
    new_image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    Gammank = KMeans().find_Gammank(new_image, code_vectors)
    required_image = code_vectors[Gammank]
    #Reshaping to (?, ?, 3)
    return required_image.reshape(image.shape[0], image.shape[1], image.shape[2])
