from numpy import (array, unravel_index, nditer, linalg, random, subtract, max,
                   power, exp, zeros, ones, arange, outer, meshgrid, dot,
                   logical_and, mean, cov, argsort, linspace, transpose,
                   einsum, prod, nan, sqrt, hstack, diff, argmin, multiply,
                   nanmean, nansum, tile, array_equal)
from sklearn.manifold import TSNE
import numpy as np
from numpy.linalg import norm
from collections import defaultdict, Counter
from warnings import warn
from sys import stdout
from time import time
from datetime import timedelta
import pickle
import math
import os


# for unit tests
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from numpy.testing import assert_array_equal
import unittest


def _build_iteration_indexes(data_len, num_iterations,
                             verbose=False, random_generator=None,
                             use_epochs=False, client_handler=None, cardId=None):
    """Returns an iterable with the indexes of the samples
    to pick at each iteration of the training.

    If random_generator is not None, it must be an instance
    of numpy.random.RandomState and it will be used
    to randomize the order of the samples."""
    if use_epochs:
        iterations = [] 
        for i in range(num_iterations):  
            iterations_per_epoch = arange(data_len)
            if random_generator:
                random_generator.shuffle(iterations_per_epoch)
            iterations.append(iterations_per_epoch)
    else:
        iterations = arange(num_iterations) % data_len
        if random_generator:
            random_generator.shuffle(iterations)
    if verbose:
        return _wrap_index__in_verbose(iterations, client_handler=client_handler, cardId=cardId)
    else:
        return iterations


def _wrap_index__in_verbose(iterations, client_handler=None, cardId=None):
    """Yields the values in iterations printing the status on the stdout."""
    m = len(iterations)
    digits = len(str(m))
    progress = '\r [ {s:{d}} / {m} ] {s:3.0f}% - ? it/s'
    progress = progress.format(m=m, d=digits, s=0)
    stdout.write(progress)
    beginning = time()
    stdout.write(progress)
    for i, it in enumerate(iterations):
        yield it
        sec_left = ((m-i+1) * (time() - beginning)) / (i+1)
        time_left = str(timedelta(seconds=sec_left))[:7]
        progress = '\r [ {i:{d}} / {m} ]'.format(i=i+1, d=digits, m=m)
        progress += ' {p:3.0f}%'.format(p=100*(i+1)/m)
        if client_handler: 
            client_handler.emitTaskEnvLayoutComputationProgress({"progress": 100*(i+1)/m, "cardId": cardId})
        progress += ' - {time_left} left '.format(time_left=time_left)
        stdout.write(progress)


def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array.
    """
    return sqrt(dot(x, x.T))


def asymptotic_decay(learning_rate, t, max_iter):
    """Decay function of the learning process.
    Parameters
    ----------
    learning_rate : float
        current learning rate.

    t : int
        current iteration.

    max_iter : int
        maximum number of iterations for the training.
    """
    return learning_rate / (1+t/(max_iter/2))


def generate_rr_projection(data, relevance, metadata, num_of_epochs=1, w_s=0.2, w_r=0.8, step = 8, verbose=True, sigma=1.5, learning_rate=.7, activation_distance='euclidean', 
                  topology='circular', neighborhood_function='gaussian', random_seed=10, plot_split=False):
    from sklearn.cluster import KMeans
    data_size = data.shape[0] # 
    embedding_size = data.shape[1]
    num_of_layers = get_best_number_of_layers(step, data_size)
    _relevance = np.exp(relevance)
    relevance = (_relevance - np.min(_relevance))/(np.max(_relevance) - np.min(_relevance))
    som = CircularSom(step,  num_of_layers, embedding_size, sigma=1.5, learning_rate=.7, activation_distance='euclidean', 
                  topology='circular', neighborhood_function='gaussian', random_seed=10)
    if verbose:
        print("logging: start som training")
    som.train(data, relevance, data_size*num_of_epochs, w_s, w_r, verbose=True)
    if verbose:
        print("logging: som training finished")
    circle_pos = get_grid_position_som(som, data, relevance)
    data_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=20).fit_transform(data)
    if verbose:
        print("logging: start data processing")
    data_dict = [{"pos_x": data_embedded[i][0], "pos_y": data_embedded[i][1], "circle_x":circle_pos[i][0], "circle_y":circle_pos[i][1], "embedding": doc.embedding, "page_content": doc.page_content, "title": doc.metadata["TI"] or "None", "embedding": doc.embedding, "relevance": relevance[i], "metadata": metadata[i]}  for i, (doc, score) in enumerate(res)]
    df = pd.DataFrame(data_dict)

    embeddings = df.embedding.to_list()
    kmeans = KMeans(n_clusters=8, random_state=0, n_init="auto").fit(embeddings)
    df["category_num"] = kmeans.labels_
    return som, df

def get_grid_position_som(som, data, relevance, ids_same_order, sort=True, split=False): 
    pos_res = {}
    radius = .3
    som.initialize_resource_map()
    im = 0
    modified_data = zip(data, relevance, ids_same_order)
    if sort: 
        modified_data = sorted(zip(data, relevance, ids_same_order), key=lambda t: t[1], reverse=True)
    for x, r, i in modified_data:  # scatterplot
        w = som.winner(x, r)
        loc = som._grid[w]
        if not split:
            pos_res[i] = [loc[0], loc[1]]
        else: 
            xloc = loc[0]
            yloc = loc[1]
            
            if xloc > 0: 
                xloc += radius*2
            else:
                xloc -= radius*2
                
            if yloc > 0:
                yloc += radius*2
            else:
                yloc -= radius*2
            pos_res[i] = [xloc, yloc]
    return pos_res
    
def get_best_number_of_layers(step: int, data_length: int, skip=1):
    """Get the best number of layers for the circular SOM.
    Parameters
    ----------
    step : int
        The number of neurons in the first layer.

    data_length : int
        The number of samples in the dataset.

    skip : int
        The number of layers to skip.

    Returns
    -------
    int
        The best number of layers for the circular SOM.
    """
    num_layers = 1
    num_neurons = step*(num_layers+skip)
    while num_neurons < data_length:
        num_layers += 1
        num_neurons += step * (num_layers + skip)
    return num_layers + 1

def generate_circular_grid(step, layer, radius=1, skip=1):
    π = math.pi
    points = []
    weights_radius = [] 
    points_map = {} 
    global_Index = 0
    num_of_points_per_layer = {}
    # num_points_in_total = 5*(2**layer - 1)
    for l in range(layer):
        _num_in_current_layer = step + step*(l+skip)
        num_of_points_per_layer[l] = _num_in_current_layer
        _angle_interval = 2*π / _num_in_current_layer
        _radius_in_current_layer = (l + 1 + skip) * radius
        for i in range(_num_in_current_layer):
            _angle = _angle_interval * i - π / _num_in_current_layer
            point = [
                _radius_in_current_layer * math.cos(_angle),
                _radius_in_current_layer * math.sin(_angle),
            ]
            points.append(point) 
            weights_radius.append(_radius_in_current_layer)
            points_map[tuple(point)] = {
                "coords": point, 
                "layer": l,
                "angle": _angle,
                "radius": _radius_in_current_layer,
                "index": global_Index
            }
            global_Index += 1

    weights_radius = np.exp(1 / array(weights_radius))
    weights_radius = (weights_radius - np.min(weights_radius)) / (np.max(weights_radius) - np.min(weights_radius))

    return array(points), weights_radius, points_map, num_of_points_per_layer


class CircularSom(object):
    """"""

    def __init__(self, step, layer, input_len, sigma=1.0, learning_rate=0.5,
                 decay_function=asymptotic_decay,
                 neighborhood_function='gaussian', topology='circular',
                 activation_distance='euclidean', random_seed=None):
        """Initializes a Self Organizing Maps.

        Parameters 
        ----------  

        """
        self._random_generator = random.RandomState(random_seed)

        self._learning_rate = learning_rate
        self._sigma = sigma
        self._input_len = input_len
        self._latest_x = None
        # random initialization
        
        self.topology = topology
        self._grid, self._weights_radius, self._grid_map, self._num_of_points_per_layer = generate_circular_grid(step, layer)
        self._grid.astype(float)
        # self._total_cells = int(step * layer + step*(layer**2 - layer) / 2)
        self._total_cells = len(self._grid)
        self._weights = self._random_generator.rand(
            self._total_cells, input_len)*2-1
        self._weights /= linalg.norm(self._weights, axis=-1, keepdims=True)

        self._activation_map = zeros((self._total_cells, 1))
        # self._resource_map = ones((self._total_cells, ))
        self.initialize_resource_map()
        self._decay_function = decay_function

        neig_functions = {'gaussian': self._gaussian}

        # self.neighborhood = neig_functions[neighborhood_function]
        self.neighborhood = self._gaussian

        distance_functions = {'euclidean': self._euclidean_distance,
                              'cosine': self._cosine_distance,
                              'manhattan': self._manhattan_distance,
                              'chebyshev': self._chebyshev_distance}

        if isinstance(activation_distance, str):
            if activation_distance not in distance_functions:
                msg = '%s not supported. Distances available: %s'
                raise ValueError(msg % (activation_distance,
                                        ', '.join(distance_functions.keys())))

            self._activation_distance = distance_functions[activation_distance]
        elif callable(activation_distance):
            self._activation_distance = activation_distance

    def initialize_resource_map(self):
        self._resource_map = ones((self._total_cells, ))

    def get_weights(self):
        """Returns the weights of the neural network."""
        return self._weights

    def get_euclidean_coordinates(self):
        """Returns the position of the neurons on an euclidean
        plane that reflects the chosen topology in two meshgrids xx and yy.
        Neuron with map coordinates (1, 4) has coordinate (xx[1, 4], yy[1, 4])
        in the euclidean plane.

        Only useful if the topology chosen is not rectangular.
        """
        return self._grid

    def update_weights_radius_with_relevance(self, relevance):
        """Updates the weights radius with the relevance score of the data.""" 
        num_worked = 0
        new_weights_relevance = []
        new_relevance = np.sort(relevance)[::-1]
        for l in range(len(self._num_of_points_per_layer)):
            num_of_points = self._num_of_points_per_layer.get(l)
            if num_of_points is not None and num_of_points > 0:
                target_values = new_relevance[num_worked:num_worked+num_of_points]
                if len(target_values) > 0:
                    new_weights_relevance += [np.mean(new_relevance[num_worked:num_worked+num_of_points])]*num_of_points
                else:
                    new_weights_relevance += [0] * num_of_points
                num_worked += num_of_points
        self._weights_radius = np.array(new_weights_relevance)

    def train(self, data, relevance_score, num_iteration, w_s=.4, w_r=.6, random_order=False, client_handler=None, cardId=None, verbose=False, use_epochs=False, report_error=False, use_sorted=False):
        """Trains the SOM picking samples at random from data
        
        Parameters
        ---------- 
        relevance_score: array-like, shape = [n_samples] range: [0, 1] 
        """
        self._check_iteration_number(num_iteration)
        self._check_input_len(data)
        self.epoch_cur = 1 
        self._wr = w_r 
        self._ws = w_s
        random_generator = None
        if random_order:
            random_generator = self._random_generator
        iterations = _build_iteration_indexes(len(data), num_iteration,
                                              verbose, random_generator,
                                              use_epochs, client_handler=client_handler, cardId=cardId)
        if use_epochs:
            def get_decay_rate(iteration_index, data_len):
                return int(iteration_index / data_len)
        else:
            def get_decay_rate(iteration_index, data_len):
                return int(iteration_index)
        self.update_weights_radius_with_relevance(relevance_score)
        if use_sorted:                
            sorted_modified_data = sorted(zip(data, relevance_score), key=lambda t: t[1], reverse=True)
            data = [x for x, r in sorted_modified_data]
            relevance_score = [r for x, r in sorted_modified_data]
        if use_epochs: 
            for epoch_num, iteration_idx in enumerate(iterations): 
                for t, iteration in enumerate(iteration_idx):
                    decay_rate = get_decay_rate(t, len(data))
                    self.update(data[iteration], relevance_score[iteration], self.winner(data[iteration], relevance_score[iteration]),
                                decay_rate, num_iteration)
                if report_error: 
                    self.dual_quantization_error(data, relevance_score)
                    self.start_a_new_epoch(verbose)
        else: 
            for t, iteration in enumerate(iterations):
                decay_rate = get_decay_rate(t, len(data))
                if iteration == 0: 
                    if report_error: 
                        self.dual_quantization_error(data, relevance_score)
                    self.start_a_new_epoch(verbose)
                self.update(data[iteration], relevance_score[iteration], self.winner(data[iteration], relevance_score[iteration]),
                            decay_rate, num_iteration)
        # if verbose:
        #     print('\n quantization error:', self.dual_quantization_error(data))

    def dual_quantization_error(self, data, relevance, sort=True): 
        self.initialize_resource_map()
        total_loss = 0
        modified_data = zip(data, relevance)
        if sort: 
            modified_data = sorted(zip(data, relevance), key=lambda t: t[1], reverse=True)
        for x, r in modified_data:  # scatterplot
            w, loss = self.winner(x, r, True)
            total_loss += loss
        print(f"\n dual quantization error at epoch-{self.epoch_cur}: ", total_loss/len(data))
    
    def start_a_new_epoch(self, verbose=False):
        """"""
        # print("Start epoch: ", self.epoch_cur)
        self.epoch_cur += 1
        self.initialize_resource_map()


    def update(self, x, r, win, t, max_iteration):
        """Updates the weights of the neurons.
        
        Parameters
        """
        eta = self._decay_function(self._learning_rate, t, max_iteration)
        # sigma and learning rate decrease with the same rule
        sig = self._decay_function(self._sigma, t, max_iteration)
        # improves the performances
        g = self.neighborhood(win, sig)*eta
        # gr = self._gaussian_radiance(win)*.01
        self._weights += einsum('i, ik->ik', g, x-self._weights)
        # self._weights_radius += einsum('i, i->i', gr, r-self._weights_radius)


    def winner(self, x, rscore, return_distance=False): 
        self._activate(x, rscore) 
        _winner = unravel_index(self._activation_map.argmin(), self._activation_map.shape)
        self._resource_map[_winner] -= 1
        if self._resource_map[_winner] < 0:
            self._latest_x = {
                "x": x,
                "r": rscore,
            }
            # report error 
            print(f"Resource map is not updated correctly: {_winner} - {self._resource_map[_winner]}")
            raise ValueError("Resource map is not updated correctly")

        if return_distance:
            return _winner, self._activation_map[_winner]
        return _winner
        
    
    def _activate(self, x, rscore):
        """Updates matrix activation_map, in this matrix
        the element i,j is the response of the neuron i,j to x."""
        self._activation_map = self._euclidean_distance(x, rscore, self._weights, self._weights_radius) 
        resource_map_mask  = np.log(1/(self._resource_map + 1e-11))*1e5
        self._activation_map += resource_map_mask
        return 
    
    def _euclidean_distance(self, x, r, w, wr):
        raw_dist = linalg.norm(subtract(x, w), axis=-1) 
        normalized_dist = (raw_dist - np.min(raw_dist))/(np.max(raw_dist) - np.min(raw_dist))
        self._normalized_dist = normalized_dist
        raw_radius_dist = np.exp(np.power(subtract(r, wr), 2))
        normalized_radius_dist = (raw_radius_dist - np.min(raw_radius_dist))/(np.max(raw_radius_dist) - np.min(raw_radius_dist))
        self._normalized_radius_dist = normalized_radius_dist
        return (self._ws*normalized_dist + self._wr*normalized_radius_dist)

    def quantization_error(self, data):
        """Returns the quantization error computed as the average
        distance between each input sample and its best matching unit."""
        self._check_input_len(data)
        return norm(data-self.quantization(data), axis=1).mean()

    def quantization(self, data):
        """Assigns a code book (weights vector of the winning neuron)
        to each sample in data."""
        self._check_input_len(data)
        winners_coords = argmin(self._distance_from_weights(data), axis=1)
        return self._weights[unravel_index(winners_coords,
                                           self._weights.shape[:2])]

    def _distance_from_weights(self, data):
        """Returns a matrix d where d[i,j] is the euclidean distance between
        data[i] and the j-th weight.
        """
        input_data = array(data)
        weights_flat = self._weights.reshape(-1, self._weights.shape[2])
        input_data_sq = power(input_data, 2).sum(axis=1, keepdims=True)
        weights_flat_sq = power(weights_flat, 2).sum(axis=1, keepdims=True)
        cross_term = dot(input_data, weights_flat.T)
        return sqrt(-2 * cross_term + input_data_sq + weights_flat_sq.T)
        


    def _check_iteration_number(self, num_iteration):
        if num_iteration < 1:
            raise ValueError('num_iteration must be > 1')

    def _check_input_len(self, data):
        """Checks that the data in input is of the correct shape."""
        data_len = len(data[0])
        if self._input_len != data_len:
            msg = 'Received %d features, expected %d.' % (data_len,
                                                          self._input_len)
            raise ValueError(msg)



    def _gaussian(self, c, sigma):
        """Returns a Gaussian centered in c."""
        d = 2*sigma*sigma
        ax = exp(-np.sum(power(self._grid-self._grid[c], 2)/d, axis=-1))
        return ax  # the external product gives a matrix

    def _gaussian_radiance(self, c):
        # d = 2*self._weights_radius[c]*self._weights_radius[c]
        dist = np.round(np.linalg.norm(self._grid, axis=1) - np.linalg.norm(self._grid[c]))
        # ax = exp(-dist/d)
        ax = np.where(dist == 0, 1, 0)
        return ax

    def _cosine_distance(self, x, w):
        num = (w * x).sum(axis=2)
        denum = multiply(linalg.norm(w, axis=2), linalg.norm(x))
        return 1 - num / (denum+1e-8)

    

    def _manhattan_distance(self, x, w):
        return linalg.norm(subtract(x, w), ord=1, axis=-1)

    def _chebyshev_distance(self, x, w):
        return max(subtract(x, w), axis=-1)


    def _plot(self, radius=.5): 
        """"""
        from matplotlib.patches import Circle
        import matplotlib.pyplot as plt 
        import numpy as np
        
        grid = self._grid 

        f = plt.figure(figsize=(10,10))
        ax = f.add_subplot(111)

        ax.set_aspect('equal')
        for c in grid: 
            circle = Circle((c[0], c[1]), radius, fc='orange') 
            ax.add_patch(circle)
        grid_x_min, grid_x_max = np.min(grid[:, 0]), np.max(grid[:, 0])
        grid_y_min, grid_y_max = np.min(grid[:, 1]), np.max(grid[:, 1])
        # Set the limits of the plot
        ax.set_xlim(grid_x_min - 2, grid_x_max + 2)
        ax.set_ylim(grid_y_min - 2, grid_y_max + 2) 

        plt.show()