import numpy as np
from scipy.ndimage import convolve
from scipy.signal import argrelmax
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster._dbscan_inner import dbscan_inner
from sklearn.neighbors import KDTree


class VortexCorrelationClustering(ClusterMixin, BaseEstimator):
    """Estimate vortex-shaped clusters on spatio-temporal data.
    
    VoCC - Vortex Correlation Clustering. VoCC is a clustering approach for spatio-temporal data which clusters data points along vortex-shaped objects. 

    Parameters
    ---
    radii: list[int]
        A list of radii for possible circle candidates. The value must be given in measurement units of the original data space.

    cell_size: float
        The distance between the edges of two cells used for the gridded execution of the Circle Hough Transformation. This also has implications to the size of the internal accumulator.

    sectors: int
        Depicts the number of sectors the Circle Hough Transformation mask is divided into. Enables to enforce a certain circle coverage over these sectors.

    circle_coverage_rate: float 
        Defining a threshold of sectors including relevant data, to enforce an at least `circle_coverage_rate` completed circle.

    qth_threshold: float
        The algorithm is using an relative maximum algorithm to select the candidates, which will cancel out all smaller relative maxima with values lower than this threshold.
    
    min_points: float
        A RDBSCAN algortihm is used, as described in the underlying paper, to merge multiple vortex candidates into a single one.
        The ratio `min_points` * len(candidates) defines the minimum points parameter passed to the underlying RDBSCAN. 
    """
    vortices_ = None
    labels_ = None

    def __init__(
        self,
        radii,
        cell_size,
        sectors,
        circle_coverage_rate,
        qth_threshold,
        min_points,
        depth_boundaries = None,
        cylindrical = False
    ):
    
        super().__init__()
        self.radii = radii
        self.cell_size = cell_size
        self.sectors = sectors
        self.circle_coverage_rate = circle_coverage_rate
        self.qth_threshold = qth_threshold
        self.min_points = min_points
        self.depth_boundaries = depth_boundaries
        self.cylindrical = cylindrical
        self.vortices_ = None
        self.labels_ = None
        self.accumulator_ = None

    
    def fit(self, X, y=None):
        """Perform VortexCorrelationClustering clustering from moving objects.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, 4) with the features positionX, positionY,
            movementX, and movementY.
        y : Ignored
            Not used, present here for API consistency by convention.
        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """

        assert X.shape[1] == 5, 'The input array must be structured as (n_samples, 5) for the propoerties position X, position Y, movement X, movement Y along the second dimension, and the Depth D'

        X, Y, U, V, D = X.T

        self.min_positions = np.array([np.min(X), np.min(Y)])

        if self.depth_boundaries is None:
            D_binned = np.ones_like(D, dtype = np.int32)
        else:
            D_binned = np.digitize(D, bins = self.depth_boundaries)

        # Normalize the depth index to a 0-based index
        D_binned = D_binned - D_binned.min()
        self.D_binned = D_binned
        normal_vectors = _bin_image(X, Y, U, V, D_binned, self.cell_size)

        self.accumulator_ = _create_parameter_space(normal_vectors, self.radii, self.sectors, self.circle_coverage_rate, self.depth_boundaries, self.cylindrical)
        
        threshold = np.max(np.abs(np.quantile(self.accumulator_, [1 - self.qth_threshold, self.qth_threshold])))
        vortices_direction_1 = _extract_vortices_from_parameter_space(self.accumulator_, threshold, self.radii, 'a')
        vortices_direction_2 = _extract_vortices_from_parameter_space(-self.accumulator_, threshold, self.radii, 'c')

        grouped_vortices_1 = _merge_votex_candidates(vortices_direction_1, self.radii, self.min_points)
        grouped_vortices_2 = _merge_votex_candidates(vortices_direction_2, self.radii, self.min_points)
        vortices = grouped_vortices_1 + grouped_vortices_2


        labels = np.full_like(X, fill_value=-1)


        for depth_index in range(D_binned.max() + 1):
            mask = D_binned == depth_index
            if np.sum(mask) < 1:
                continue
            X_d = X[mask]
            Y_d = Y[mask]

            mask_indices = np.atleast_1d(np.argwhere(mask).squeeze())

            particle_tree = KDTree(np.array([X_d,Y_d]).T)
            for vortex_id, vortex in enumerate(vortices):

                backprojected_positions = vortex.get_circle_positions() * self.cell_size + np.array([np.min(X), np.min(Y)])
                backprojected_radii = vortex.get_circle_radii() * self.cell_size

                circle_depth_mask = vortex.pixels[:, 3] == depth_index

                if np.sum(circle_depth_mask) < 1:
                    continue

                backprojected_positions = backprojected_positions[circle_depth_mask]
                backprojected_radii = backprojected_radii[circle_depth_mask]

                influenced_position_indices = particle_tree.query_radius(backprojected_positions, backprojected_radii)
                influenced_position_indices = np.concatenate(influenced_position_indices)
                
                
                
                if self.cylindrical:
                    labels[influenced_position_indices] = vortex_id
                else:
                    labels[mask_indices[influenced_position_indices]] = vortex_id

        self.labels_ = labels
        self.vortices_ = vortices

        return self


class Vortex():
    def __init__(self, x_bin: int, y_bin: int, radius: int, pixels: np.ndarray, flag, depth_level):
        self.x_bin = x_bin 
        self.y_bin = y_bin
        self.radius = radius 
        self.pixels = pixels
        self.flag = flag
        self.depths_level = depth_level
    
    def __repr__(self) -> str:
        return f'[{self.x_bin},{self.y_bin}] -> {self.radius}; #pixels {self.pixels.shape[0] if isinstance(self.pixels, np.ndarray) else len(self.pixels)}'

    def get_circle_positions(self):
        try:
            pixel_positions = self.pixels[:, :2]
        except:
            print(self.pixels)
            raise
        return pixel_positions

    
    def get_circle_radii(self):
        return self.pixels[:, 2]
    

    def get_depth_layers(self):
        return np.unique(self.pixels[:, 3])


def _bin_image(X, Y, U, V, D_binned, cell_size):
    bins_X = np.arange(np.min(X), np.max(X), cell_size)
    bins_Y = np.arange(np.min(Y), np.max(Y), cell_size)

    X_indices = np.digitize(X, bins = bins_X)
    Y_indices = np.digitize(Y, bins = bins_Y)
    
    movement = np.zeros(shape = (bins_X.shape[0] +1, bins_Y.shape[0]+1, D_binned.max() + 1, 2))
    normal = np.zeros(shape = (bins_X.shape[0] +1, bins_Y.shape[0]+1, D_binned.max() + 1, 2))

    for d_i in range(D_binned.max() + 1):
        mask = D_binned == d_i
        U_d = U[mask]
        V_d = V[mask]
        X_d = X_indices[mask]
        Y_d = Y_indices[mask]

        for x_i, x in enumerate(bins_X):
            for y_i, y in enumerate(bins_Y):
                mask = (X_d == x_i) & (Y_d == y_i)
                if np.sum(mask) < 1:
                    continue
                U_x_y = U_d[mask]
                V_x_y = V_d[mask]
                U_x_y = np.median(U_x_y)
                V_x_y = np.median(V_x_y)

                movement[x_i, y_i, d_i, 0] = U_x_y
                movement[x_i, y_i, d_i, 1] = V_x_y
    

    # Normalize the length of the movement
    norm = np.linalg.norm(movement, axis = -1)[..., np.newaxis]
    movement = np.divide(movement, norm, where = norm > 0)

    normal[..., 0] = - movement[..., 1]
    normal[..., 1] = movement[..., 0]

    return normal

def _create_parameter_space(normal_vectors, radii, sectors, circle_coverage_rate, depth_boundaries, cylindrical):
    accumulator = []
    for radius in radii:
        XX, YY = np.meshgrid(np.arange(2 * radius +1), np.arange(2* radius +1 ))

        distances_from_center =(XX-radius)**2 + (YY-radius)**2
        activation_mask = (distances_from_center < radius ** 2).astype(np.float32)
        

        angle_mask = np.arctan2(XX-radius, YY-radius)
        direction_mask_x = np.cos(angle_mask) * activation_mask
        direction_mask_y = np.sin(angle_mask) * activation_mask
        regularization = np.sum(direction_mask_x != 0)


        sector_angle = 2*np.pi/sectors
        positive_angle_mask = angle_mask + np.pi 
        
        sector_accumulator = []
        for sector_index in range(sectors):
            activation_mask_sector = np.zeros_like(activation_mask)
            activation_mask_sector[(positive_angle_mask >= sector_index * sector_angle) & 
                                    (positive_angle_mask <  (sector_index+1) * sector_angle)] = 1.0

            ht_mask_x = activation_mask * activation_mask_sector * direction_mask_x 
            ht_mask_y = activation_mask * activation_mask_sector * direction_mask_y 
            depth_dependent_convolution = []
            for d_i in range(normal_vectors.shape[2]):
                depth_dependent_convolution.append(convolve(normal_vectors[..., d_i, 0], ht_mask_x) + convolve(normal_vectors[..., d_i, 1], ht_mask_y))
            
            if cylindrical:
                sector_vector_i = np.sum(np.array(depth_dependent_convolution), axis = 0, keepdims=True)
            else:
                sector_vector_i = np.array(depth_dependent_convolution)

            sector_accumulator.append(sector_vector_i)
        
        # qth-quantile selection of the sector vectors
        sector_accumulator = np.array(sector_accumulator).reshape(sectors, sector_accumulator[0].shape[0], normal_vectors.shape[0], normal_vectors.shape[1])

        positive_vlaues = np.quantile(np.where(sector_accumulator > 0, sector_accumulator, 0), 1 - circle_coverage_rate, axis = 0)
        negative_vlaues = np.quantile(np.where(sector_accumulator < 0, sector_accumulator, 0), circle_coverage_rate, axis = 0)

        accumulator_r_slice = np.where(positive_vlaues > np.abs(negative_vlaues), positive_vlaues, negative_vlaues)
        accumulator_r_slice /= regularization
        accumulator.append(accumulator_r_slice)
    accumulator = np.array(accumulator)
    return accumulator
    
def _extract_vortices_from_parameter_space(accumulator, threshold, radii, flag):
    vortices = []

    for depth_index in range(accumulator.shape[1]):

        accumulator_high_values = np.maximum(accumulator[:, depth_index], np.full_like(accumulator[:, depth_index], fill_value=threshold))

        accumulator2d = np.max(accumulator_high_values, axis= 0)
        accumulator2d_indices = np.argmax(accumulator_high_values, axis= 0)

        local_maxima = argrelmax(accumulator2d)
        indices = np.array(local_maxima)
        for index in indices.T:
            vortices.append(Vortex(index[0],index[1], radii[accumulator2d_indices[index[0], index[1]]], np.empty(shape = (0,4)), flag, depth_index))

    return vortices

def _merge_votex_candidates(vortices, radii, rdbscan_min_points):
    # Remove vortex candidates that are completely enclosed by bigger candidates.
    for vortex in vortices:
        radius_index = np.argwhere(radii == vortex.radius).squeeze()
        if radius_index + 1 == len(radii):
            continue 

        for vortex_prime in vortices:
            if vortex == vortex_prime:
                continue
            if vortex_prime.radius <= vortex.radius:
                continue

            distance = np.linalg.norm([vortex.x_bin - vortex_prime.x_bin, vortex.y_bin - vortex_prime.y_bin])

            if vortex_prime.radius >= distance + vortex.radius:
                vortices.remove(vortex)
                break
    
    if len(vortices) < 1:
        return []

    X = np.empty(shape = (len(vortices), 4))
    for vortex_i, vortex in enumerate(vortices):
        X[vortex_i] = [vortex.x_bin, vortex.y_bin, vortex.radius, vortex.depths_level]

    labels = _radius_eps_DBSCAN(X, max(5,int(rdbscan_min_points * len(vortices))))

    vortices_grouped = []
    for l in np.unique(labels):
        if l < 0:
            continue
        mask = labels == l

        if np.sum(mask) == 1:
            vortex = vortices[np.argwhere(mask).squeeze()]
            pixel = [[vortex.x_bin, vortex.y_bin, vortex.radius, vortex.depths_level]]
            vortex.pixels = np.array(pixel).reshape(1, 4)
            vortices_grouped.append(vortex)
        elif np.sum(mask) > 1:
            indices = np.argwhere(labels == l).squeeze()

            if len(indices) < 1:
                continue

            pixels = []
            inner_x_y_r = np.empty(shape= (len(indices), 4))
            for index_i, index in enumerate(indices):
                vortex = vortices[index]
                inner_x_y_r[index_i] = [vortex.x_bin, vortex.y_bin, vortex.radius, vortex.depths_level]
                pixels.append(vortex.pixels)

            if len(pixels) < 1:
                continue


            new_x_bin = np.around(np.mean(inner_x_y_r[:, 0]), decimals= 0)
            new_y_bin = np.around(np.mean(inner_x_y_r[:, 1]), decimals= 0)

            vortex_group = Vortex(new_x_bin, new_y_bin, np.max(inner_x_y_r[:, 2]), inner_x_y_r, vortex.flag, np.atleast_1d(np.unique(inner_x_y_r[:, 3])))
            vortices_grouped.append(vortex_group)

    return vortices_grouped

def _radius_eps_DBSCAN(X: np.ndarray, min_samples : int):
    """
    X is an array (n, 4) with x, y, radius, and depth_index
    """
    assert X.shape[1] == 4, "Please use three dimensions for the radius DBSCAN"

    neighborhoods = np.empty(shape = (X.shape[0]), dtype = object)
    for depth_index in np.unique(X[:, 3]):
        depth_mask_outer = np.abs(X[:, 3] - depth_index) <= 1
        depth_mask_inner = X[:, 3] == depth_index
        depth_mask_outer_indices = np.atleast_1d(np.argwhere(depth_mask_outer).squeeze())


        X_outer_depth = X[depth_mask_outer]
        X_inner_depth = X[depth_mask_inner]
        

        tree = KDTree(X_outer_depth[:, :2])
        
        neighborhoods_in_depth_layer = np.empty(shape = (depth_mask_inner.sum()), dtype = object)
        
        for r in np.unique(X_inner_depth[:, 2])[::-1]:
            r_mask = X_inner_depth[:, 2] == r
            neighbors_per_r = tree.query_radius(X_inner_depth[r_mask, :2],r)

            # map the indices back to the original array
            neighbors_per_r = [depth_mask_outer_indices[neighbors] for neighbors in neighbors_per_r]

            """
            # We dont need this, there is something else (probably in dbscan_inner that makes it bidirectional)
            for i in range(neighbors_per_r.shape[0]):
                print(np.argwhere(r_mask).squeeze())
                
                looking_for = np.argwhere(r_mask).squeeze()
                if neighbors_per_r.shape[0] > 1:
                    looking_for = looking_for[i]
                additional_neighbors = []
                for j in range(neighborhoods.shape[0]):
                    if neighborhoods[j] is None:
                        continue
                    if looking_for in neighborhoods[j]:
                        additional_neighbors.append(j)
                neighbors_per_r[i] = np.concatenate([neighbors_per_r[i], additional_neighbors]).astype(np.intp)
            """

            neighborhoods_in_depth_layer[r_mask] = np.array(neighbors_per_r + [None], dtype=object)[:-1]
        
        # Find all vortices that are at the same x, y location but are in a different depth layer
        # Assign the neighbor indices to the same index in the neighborhoods array
        
        neighborhoods[depth_mask_inner] = neighborhoods_in_depth_layer

    n_neighbors = np.array([len(neighbors) for neighbors in neighborhoods])
    labels = np.full(X.shape[0], -1, dtype=np.intp)

    # A list of all core samples found.
    core_samples = np.asarray(n_neighbors >= min_samples, dtype=np.uint8)
    dbscan_inner(core_samples, neighborhoods, labels)
    return labels
