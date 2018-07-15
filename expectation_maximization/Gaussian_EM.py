import numpy as np
from scipy import random
import time
from matplotlib import patches
import matplotlib.pyplot as plt
from scipy import stats


class GMM:
    def __init__(self, K):
        self.K = K
        self.means = []  # randomly initialize, then update with the formula
        self.covariances = []  # randomly initialize, then update with the formula
        self.pis = []  # randomly initialize, then update with the formula
        self.gammas = []  # update using the formula

    def fit(self, data):
        """
        :params data: np.array of shape (..., dim)
                                  where dim is number of dimensions of point
        """
        self._initialize_params(data)
        # TODO: while is not converging
        for i in range(10):  # run 10 iterations for now
            print("Iteration #" + str(i + 1))
            self._E_step(data)
            self._M_step(data)

    def _initialize_params(self, data):
        self.gammas = np.empty(shape=[K, data.shape[0]], dtype=np.float64)
        # initializing means
        x_min = data.transpose()[0].min(axis=0)
        x_max = data.transpose()[0].max(axis=0)
        y_min = data.transpose()[1].min(axis=0)
        y_max = data.transpose()[1].max(axis=0)
        for i in range(self.K):
            self.means.append([random.uniform(x_min, x_max), random.uniform(y_min, y_max)])

        # initializing covariances
        for i in range(self.K):
            self.covariances.append(np.eye(data.shape[1]))

        # initializing pis
        for i in range(self.K):
            self.pis.append(1/self.K)

        self.means = np.asarray(self.means, dtype=np.float64)
        self.covariances = np.asarray(self.covariances, dtype=np.float64)
        self.pis = np.asarray(self.pis, dtype=np.float64)

    def _E_step(self, data):
        coefs = np.empty(shape=[self.K], dtype=np.float64)
        for k in range(self.K):
            coefs[k] = self.pis[k] * (1 / np.sqrt(2 * self.pis[k] * np.linalg.det(self.covariances[k])))

        for i in range(data.shape[0]):
            exponents = np.empty(shape=[self.K], dtype=np.float64)
            sum_exponents = 0
            for k in range(self.K):
                exponents[k] = stats.multivariate_normal(self.means[k], self.covariances[k]).cdf(data[i])
                sum_exponents += exponents[k]
            # TODO runtime warning here
            for k in range(self.K):
                self.gammas[k, i] = (coefs[k] * exponents[k]) / sum_exponents

    def _M_step(self, data):
        for k in range(K):
            sum_gamma = 0
            sum_gamma_x = [0, 0]
            sum_gamma_x_covariance = [0, 0]
            for i in range(data.shape[0]):
                sum_gamma += self.gammas[k, i]
                sum_gamma_x = sum_gamma_x + self.gammas[k, i] * data[i]
                diff = data[i] - self.means[k]
                product = np.dot(diff, np.transpose(diff))
                sum_gamma_x_covariance = sum_gamma_x_covariance + self.gammas[k, i] * product
            self.means[k] = sum_gamma_x/sum_gamma
            self.covariances[k] = sum_gamma_x_covariance / sum_gamma
            self.pis[k] = sum_gamma / data.shape[0]

    def predict(self, data):
        """
        :param data: np.array of shape (..., dim)
        :return: np.array of shape (...) without dims
                         each element is integer from 0 to k-1
        """
        prediction = np.empty(shape=[data.shape[0]], dtype=int)
        for i, _ in enumerate(data):
            probabilities = np.empty(shape=[self.K], dtype=np.float64)
            for k in range(self.K):
                probabilities[k] = stats.multivariate_normal(self.means[k], self.covariances[k]).cdf(data[i])
            prediction[i] = probabilities.tolist().index(probabilities.max())
        return prediction

    def get_means(self):
        return self.means.copy()

    def get_covariances(self):
        return self.covariances.copy()

    def get_pis(self):
        return self.pis.copy()


def get_ellipse_from_covariance(matrix, std_multiplier=2):
    values, vectors = np.linalg.eig(matrix)
    maxI = np.argmax(values)
    large, small = values[maxI], values[1 - maxI]
    return (std_multiplier * np.sqrt(large),
            std_multiplier * np.sqrt(small),
            np.rad2deg(np.arccos(vectors[0, 0])))


if __name__ == "__main__":
    path = "/Users/mavagyan/MachineLearning/ML-ISTC-Unsupervised/GMM"
    cluster_data = np.genfromtxt(path + "/dense_clusters.csv", dtype=float, delimiter=',', skip_header=True)
    normalization_factor = 10000
    cluster_data = np.divide(cluster_data, normalization_factor)
    plt.clf()
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=3, color='blue')

    # number of clusters
    K = 3
    gmm = GMM(K)
    gmm.fit(cluster_data)
    print(gmm.predict(cluster_data))

    mean = gmm.get_means()
    sigma = gmm.get_covariances()
    pi = gmm.get_pis()

    # Plot ellipses for each of covariance matrices.
    for k in range(len(sigma)):
        h, w, angle = get_ellipse_from_covariance(sigma[k])
        e = patches.Ellipse(mean[k], w, h, angle=angle)
        e.set_alpha(np.power(pi[k], .3))
        e.set_facecolor('red')
        plt.axes().add_artist(e)
    plt.show()
    plt.savefig('covariances_{}_{}'.format(cluster_data, "Gaussian-EM" + str(time.time())))

