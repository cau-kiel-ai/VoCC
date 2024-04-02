from cartopy.crs import PlateCarree
from cartopy.feature import COASTLINE, LAND
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

depth_layers = np.array([4.7679895e-01, 1.5707064e+00, 2.8554180e+00, 4.3355064e+00,
       6.0156703e+00, 7.9007444e+00, 9.9957113e+00, 1.2305715e+01,
       1.4836073e+01, 1.7592300e+01, 2.0580124e+01, 2.3805506e+01,
       2.7274673e+01, 3.0994144e+01, 3.4970764e+01, 3.9211739e+01,
       4.3724689e+01, 4.8517693e+01, 5.3599339e+01, 5.8978794e+01,
       6.4665886e+01, 7.0671165e+01, 7.7006004e+01, 8.3682709e+01,
       9.0714615e+01, 9.8116249e+01, 1.0590342e+02, 1.1409344e+02,
       1.2270523e+02, 1.3175960e+02, 1.4127936e+02, 1.5128963e+02,
       1.6181805e+02, 1.7289508e+02, 1.8455423e+02, 1.9683240e+02,
       2.0977017e+02, 2.2341211e+02, 2.3780708e+02, 2.5300858e+02,
       2.6907489e+02, 2.8606949e+02, 3.0406100e+02, 3.2312344e+02,
       3.4333621e+02, 3.6478378e+02, 3.8755551e+02, 4.1174518e+02,
       4.3745007e+02, 4.6477032e+02, 4.9380743e+02, 5.2466309e+02,
       5.5743713e+02, 5.9222620e+02, 6.2912128e+02, 6.6820563e+02,
       7.0955292e+02, 7.5322528e+02, 7.9927124e+02, 8.4772443e+02,
       8.9860266e+02, 9.5190729e+02, 1.0076231e+03, 1.0657189e+03,
       1.1261479e+03, 1.1888505e+03, 1.2537540e+03, 1.3207762e+03,
       1.3898268e+03, 1.4608092e+03, 1.5336232e+03, 1.6081664e+03,
       1.6843364e+03, 1.7620319e+03, 1.8411539e+03, 1.9216071e+03,
       2.0033002e+03, 2.0861470e+03, 2.1700664e+03, 2.2549822e+03,
       2.3408240e+03, 2.4275269e+03, 2.5150303e+03, 2.6032795e+03,
       2.6922241e+03, 2.7818181e+03, 2.8720193e+03, 2.9627896e+03,
       3.0540945e+03, 3.1459023e+03, 3.2381846e+03, 3.3309153e+03,
       3.4240706e+03, 3.5176292e+03, 3.6115713e+03, 3.7058794e+03,
       3.8005366e+03, 3.8955283e+03, 3.9908408e+03, 4.0864612e+03,
       4.1823779e+03, 4.2785801e+03, 4.3750576e+03, 4.4718013e+03,
       4.5688027e+03, 4.6660532e+03, 4.7635449e+03, 4.8612715e+03,
       4.9592251e+03, 5.0573999e+03, 5.1557891e+03, 5.2543877e+03,
       5.3531895e+03, 5.4521890e+03, 5.5513809e+03, 5.6507607e+03,
       5.7503232e+03, 5.8500645e+03, 5.9499790e+03, 6.0500630e+03],
      dtype=np.float32)

def create_synthetic_outlier_set(seed):
    np.random.seed(seed)    
    image = np.zeros(shape= (500, 300))
    normals = np.zeros(shape = (500, 300, 2))
    movement = np.zeros(shape = (500, 300, 2))

    image[0,0] = 1
    image[0,-1] = 1

    circle_centers_radius = [[230, 140, 70], [80, 240, 20]]

    points = 100
    for cx, cy, r in circle_centers_radius:

        random_values = np.random.random(size=(points, 2))

        random_angles = (random_values[:, 0] * 2 - 1) *np.pi
        random_distances = random_values[:, 1] * r


        x_positions = np.cos(random_angles) * random_distances + cx
        y_positions = np.sin(random_angles) * random_distances + cy

        x_positions = np.around(x_positions).astype(np.int32)
        y_positions = np.around(y_positions).astype(np.int32)

        image[x_positions, y_positions] = 1

        orientation_angle = random_angles + np.random.normal(scale= np.pi / 4,size = points)

        normals[x_positions, y_positions, 0] = np.cos(orientation_angle) 
        normals[x_positions, y_positions, 1] = np.sin(orientation_angle) 

    points = 100
    flow_start = np.array([300, 20])
    flow_end = np.array([480, 150])

    vector = flow_end - flow_start
    vector_perpendicular = np.array([-vector[1], vector[0]])
    vector_perpendicular = vector_perpendicular / np.linalg.norm(vector_perpendicular)

    positions = flow_start + vector * np.random.random(size = (points, 1)) + vector_perpendicular * np.random.normal(size = (points, 1), scale = 10)

    x_positions = np.around(positions[:, 0]).astype(np.int32)
    y_positions = np.around(positions[:, 1]).astype(np.int32)
    image[x_positions, y_positions] = 1

    angles = np.random.normal(scale= np.pi / 8,size = points)


    normals[x_positions, y_positions, 0] = np.cos(angles) * vector_perpendicular[0] - np.sin(angles) * vector_perpendicular[1] 
    normals[x_positions, y_positions, 1] = np.sin(angles) * vector_perpendicular[0] + np.cos(angles) * vector_perpendicular[1]

    movement[..., 0] = - normals[:,:, 1]
    movement[..., 1] = normals[:,:, 0]

    X,Y = np.argwhere(image).T
    U, V = movement[X, Y].T

    return np.vstack([X,Y,U,V]).T

def create_benchmarking_dataset(dimension_x : int, dimension_y:int, n_circles: int, 
                                n_points_per_circle: int, snr: float, possible_radii: np.ndarray,
                                r_inner_free_radius:float = 0, orientation_noise_std:float = np.pi/8,
                                add_flow = False, flow_start = None, flow_end = None):

    if add_flow:
        points = 300
        vector = flow_end - flow_start
        vector_perpendicular = np.array([-vector[1], vector[0]])
        vector_perpendicular = vector_perpendicular / np.linalg.norm(vector_perpendicular)

        positions = flow_start + vector * np.random.random(size = (points, 1)) + vector_perpendicular * np.random.normal(size = (points, 1), scale = 4)

        X = positions[:, 0]
        Y = positions[:, 1]
        angles = np.random.normal(scale= np.pi / 8,size = points)
        U = - (np.sin(angles) * vector_perpendicular[0] + np.cos(angles) * vector_perpendicular[1])
        V = np.cos(angles) * vector_perpendicular[0] - np.sin(angles) * vector_perpendicular[1] 

        coordinates_flow = np.vstack([X,Y,U,V, np.full_like(X, fill_value=-2)]).T
    

    r_max = possible_radii[-1]
    circles = []
    tries = 0
    while len(circles) < n_circles:

        circle = np.random.randint(low = (r_max,r_max,0), high=(dimension_x-r_max, dimension_y-r_max, possible_radii.shape[0]), size= (3,))
        circle[2] = possible_radii[circle[2]]
        distances_to_all_circles = [np.linalg.norm(c[:2] - circle[:2]) - 1.4* (c[2] + circle[2]) for c in circles]
        
        if add_flow:
            distances_to_flow = np.linalg.norm(flow_start + (circle[:2] -flow_start).dot(vector) / vector.dot(vector) * vector - circle[:2])
        else:
            distances_to_flow = max(dimension_x, dimension_y)

        if not np.all(np.array(distances_to_all_circles) > 0) or distances_to_flow < circle[2]*2:
            tries += 1
            if tries > 1_000:
                break
            continue


        circles.append(circle)
        tries = 0
    circles = np.array(circles)
    coordinates = np.empty(shape = (len(circles) * n_points_per_circle, 5)) # X, Y, U, V, L
    for circle_i, circle in enumerate(circles):
        random_values = np.random.random(size=(n_points_per_circle, 2))

        random_angles = (random_values[:, 0] * 2 - 1) *np.pi
        random_distances = np.sqrt(random_values[:, 1]) * (circle[2] - r_inner_free_radius) + r_inner_free_radius


        x_positions = np.cos(random_angles) * random_distances + circle[0]
        y_positions = np.sin(random_angles) * random_distances + circle[1]
        
        orientation_angle = random_angles + np.random.normal(scale= orientation_noise_std,size = n_points_per_circle)

        if np.random.random(1) >.5:
            movement_angle = orientation_angle + np.pi/2
        else:
            movement_angle = orientation_angle - np.pi/2
        
        coordinates[circle_i * n_points_per_circle: (circle_i + 1) * n_points_per_circle, 0] = x_positions
        coordinates[circle_i * n_points_per_circle: (circle_i + 1) * n_points_per_circle, 1] = y_positions
        coordinates[circle_i * n_points_per_circle: (circle_i + 1) * n_points_per_circle, 2] = np.cos(movement_angle)
        coordinates[circle_i * n_points_per_circle: (circle_i + 1) * n_points_per_circle, 3] = np.sin(movement_angle)
        coordinates[circle_i * n_points_per_circle: (circle_i + 1) * n_points_per_circle, 4] = circle_i


    noise_samples = []

    while len(noise_samples) < snr * coordinates.shape[0]:
        xyuv = np.random.uniform(low = (0,0,-1,-1,0), high = (dimension_x, dimension_y, 1,1,1),size= (5,))


        distances_to_centers = np.array([np.linalg.norm(xyuv[:2] - c[:2]) for c in circles])
        if np.any(distances_to_centers < circles[:, 2]):
            continue

        xyuv[2:4] /= np.linalg.norm(xyuv[2:4])
        xyuv[4] = -1 # Noise

        noise_samples.append(xyuv)
        
    if add_flow:
        coordinates = np.vstack([coordinates, noise_samples, coordinates_flow])
    else:
        coordinates = np.vstack([coordinates, noise_samples])

    return coordinates

def plot_benchmark(coordinates, predicted_labels):
    X,Y,U,V,L = coordinates.T
    fig = plt.figure(figsize = (10,12))
    ax = fig.add_subplot(2,1,1)

    colors = plt.cm.gist_ncar(np.linspace(0, 1, np.unique(L).shape[0]))
    ax.quiver(X,Y,U,V)

    ax = fig.add_subplot(2,1,2)
    ax.plot(X, Y, '.', color = 'grey', markersize = 2)

    for vortex_i, col in enumerate( colors[1:]):
        indices = np.argwhere(predicted_labels == vortex_i).squeeze()
        ax.plot(X[indices], Y[indices],'o', markerfacecolor = col, markeredgecolor = 'k', markersize = 6, zorder = 2, linewidth = 0)

def plot_benchmark_quiver(coordinates, predicted_labels):
    X,Y,U,V,L = coordinates.T
    fig = plt.figure(figsize = (10,6))
    colors = plt.cm.gist_ncar(np.linspace(0, 1, len(np.unique(L))))

    ax = fig.add_subplot(1,1,1)

    for vortex_i, col in enumerate(colors[1:]):
        indices = np.argwhere(predicted_labels == vortex_i).squeeze()
        ax.quiver(X[indices], Y[indices], U[indices], V[indices], color = col, zorder = 5, scale = 50, edgecolor = 'k', linewidth = 1)

    indices = np.argwhere(predicted_labels == -1).squeeze()
    ax.quiver(X[indices], Y[indices], U[indices], V[indices], scale = 60, color = 'grey', alpha = .5, zorder = 1)

    ax.set_xticks([])
    ax.set_yticks([])

def calc_rates(true_labels, predicted_labels):
    TP = np.sum((predicted_labels >= 0) & (true_labels >= 0 ))
    TN = np.sum((predicted_labels < 0) & (true_labels < 0 ))

    FP = np.sum((predicted_labels >= 0) & (true_labels < 0 ))
    FN = np.sum((predicted_labels < 0) & (true_labels >= 0 ))
    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}


def create_real_world_image(coordinates, labels, output_name, text = False, ax = None, font_size = 16):
    unique_labels = set(labels)
    colors = plt.cm.gist_ncar(np.linspace(0, 1, len(unique_labels)))

    if ax is None:
        fig = plt.figure(figsize= (10,5))
        ax = fig.add_subplot(1,1,1, projection = PlateCarree())
    else:
        output_name = None # Disable the saving because this is only part of a bigger plot
    
    ax.add_feature(COASTLINE)
    ax.add_feature(LAND, facecolor = 'darkgrey')

    for k, col in zip(unique_labels, colors):
        xy = coordinates[:,:2][labels == k]
        if k == -1:
            ax.plot(xy[:, 0], xy[:, 1], '.',color='grey', markersize=0.1)
        if k > -1:
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor = 'k', markersize=6,zorder=2, alpha = .3)
            if text:
                ax.text(xy[:, 0].mean(), xy[:, 1].mean(), str(int(k)), fontsize = font_size, c = 'w')

    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
    gl.top_labels, gl.right_labels = False, False
    gl.xlabel_style, gl.ylabel_style = {'fontsize': font_size}, {'fontsize': font_size}

    lat_min = -45
    lat_max = -31
    lon_min = 5
    lon_max = 35
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])

    if output_name is not None:
        plt.savefig(f'../fig/{output_name}.png', dpi = 300)

    return ax