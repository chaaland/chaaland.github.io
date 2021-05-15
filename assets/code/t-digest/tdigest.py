from pathlib import Path

import pandas as pd
import numpy as np

TAU = 2 * np.pi

def scale_fn(q, delta):
  return delta / TAU * np.arcsin(2 * q - 1)

def inverse_scale_fn(k, delta):
  return 0.5 * (np.sin(TAU / delta * k) + 1)

class Cluster:
  def __init__(self, centroid, weight=1):
    self._mean = centroid
    self._n = weight
  
  @property
  def weight(self):
    return self._n

  @property
  def centroid(self):
    return self._mean

  def __add__(self, other: Cluster):
    new_weight = self.weight + other.weight
    new_mean = (self.centroid * self.weight + other.centroid * other.weight) / new_weight
    new_cluster = Cluster(new_mean, new_weight)

    return new_cluster
  
  def __repr__(self):
    centroid_arg = f"centroid={self._centroid}"
    n_points_arg = f"n_points={self._n_points}"

    return f"{self.__class__.__name__}({centroid_arg}, {n_points_arg})"


class TDigest:
  def __init__(self, delta):
    self._delta = delta
    self._clusters = []
  
  def update(self, centroids):
    pass
  
  def cdf(self, x):
    pass

  @property
  def centroids(self):
    return [c for c in self._centroids]

  def clusters(self):
    return [c.centroid for c in self._centroids]

  def __add__(self, other_digest):
    digest_centroids = [self.centroids, other_digest.centroids]
    data = list(chain(self.C.values(), other_digest.C.values()))
    new_digest = TDigest(self.delta, self.K)

    if len(data) > 0:
        for c in pyudorandom.items(data):
            new_digest.update(c.mean, c.count)

    return new_digest

def simple_stats_aggregator(files):
  max_val = -np.inf
  min_val = np.inf
  nnz = 0

  for f in files:
    df = pd.read_csv(f)
    max_val = max(max_val, df.max())
    min_val = min(min_val, df.min())
    nnz += df[df==0].sum(axis=0)

  summary = {
    "max": max_val,
    "min": min_val,
    "nnz": nnz,
  }

  return summary

def plot_splash_image():
  np.random.seed(3)
  n = 100
  eps = 0.2 * np.random.randn(n)
  x_train = np.random.randn(n)
  y_train = np.square(x_train) + eps
  small_tree = RegressionTree(min_sample=5, max_depth=3)
  med_tree = RegressionTree(min_sample=5, max_depth=5)
  big_tree = RegressionTree(min_sample=5, max_depth=11)

  small_tree.fit(x_train[:, np.newaxis], y_train)
  med_tree.fit(x_train[:, np.newaxis], y_train)
  big_tree.fit(x_train[:, np.newaxis], y_train)

  x_vals = np.linspace(-3, 3, 1000)[:, np.newaxis]
  y_small_pred = small_tree.predict(x_vals)
  y_med_pred = med_tree.predict(x_vals)
  y_big_pred = big_tree.predict(x_vals)

  plt.scatter(x_train, y_train, color="k", alpha=0.7)
  plt.step(x_vals, y_small_pred)
  plt.step(x_vals, y_med_pred)
  plt.step(x_vals, y_big_pred)

  fname = Path("..", "..", "images", "numpy-regression-trees", "splash-image.png")
  plt.savefig(fname)




def cluster_points(points, delta):
  n_points = len(points)
  percentile_increment = 1 / n_points

  sorted_points = np.sort(points)
  q_partitions = []
  centroids = []

  k_lower = scale_fn(0, delta)
  data_clusters = [[]]
  for j, pt in enumerate(sorted_points):
      percentile = (j + 1) * percentile_increment
      k_upper = scale_fn(percentile, delta)
      if k_upper - k_lower < 1:
          data_clusters[-1].append(pt)
      else:
          cluster_center = np.mean(data_clusters[-1])
          centroids.append(cluster_center)
          data_clusters.append([pt])
          q_right = inverse_scale_fn(k_upper, delta)
          q_partitions.append(q_right)
          k_lower = k_upper

  return data_clusters


def clustering_with_scale_function_animation():
    np.random.seed(2718)
    n_points = 25
    points = 2 * np.random.randn(n_points) + 0.5
    sorted_points = np.sort(points)

    delta = 10
    
    def plot_cluster_and_k_function(i):
        data_clusters = [[]]
        k_lower = scale_fn(0, delta)
        colors = {
            0: 'b',
            1: 'g',
            2: 'y',
            3: 'orange',
            4: 'r',
        }

        q_partitions = []
        percentile_increment = 1 / n_points
        centroids = []
        for j, pt in enumerate(sorted_points[:i + 1]):
            percentile = (j + 1) * percentile_increment
            k_upper = scale_fn(percentile, delta)
            if k_upper - k_lower < 1:
                data_clusters[-1].append(pt)
            else:
                centroids.append(np.mean(data_clusters[-1]))
                data_clusters.append([pt])
                q_right = inverse_scale_fn(k_upper, delta)
                q_partitions.append(q_right)
                k_lower = k_upper

        fig = plt.figure()
        ax = plt.subplot(5,2,5)
        plt.yticks([])
        n_colors = len(colors)
        for j, c in enumerate(data_clusters):
          ax.scatter(c, np.zeros_like(c), alpha=0.3, color=colors[j % n_colors], s=75)
        
        for j, center in enumerate(centroids):
          ax.scatter(center, 0, color=colors[j % n_colors], s=100, marker="x")

        if i == n_points - 1:
          c = colors[(len(data_clusters) - 1) % n_colors]
          ax.scatter(np.mean(data_clusters[-1]), 0, color=c, s=100, marker="x")

        unclustered_points = sorted_points[i+1:]
        ax.scatter(unclustered_points, np.zeros_like(unclustered_points), alpha=0.3, color="k", s=75)
        x_min = sorted_points[0]
        x_max = sorted_points[-1]
        plt.xlim([x_min - 0.05 * np.abs(x_min), x_max + 0.05 * np.abs(x_max)])

        ax = plt.subplot(122)
        k_min = scale_fn(0, delta)
        k_max = scale_fn(1.0, delta)
        plt.xlim([-0.05, 1.05])
        plt.ylim([k_min, k_max])

        pct_vals = np.linspace(0, 1, 500)
        ax.plot(pct_vals, scale_fn(pct_vals, delta))
        q_partitions = np.array(q_partitions)

        cluster_quantile_boundaries = q_partitions - percentile_increment / 2
        cluster_boundary_scale_values = scale_fn(cluster_quantile_boundaries, delta)
        ax.scatter(cluster_quantile_boundaries, cluster_boundary_scale_values, s=50, alpha=0.8, color='k', marker="+")

        data_quantiles = []
        cnt = 1
        for j, c in enumerate(data_clusters):
          quantiles = np.arange(cnt, cnt + len(c)) / n_points
          ax.scatter(quantiles, scale_fn(quantiles, delta), color=colors[j], alpha=0.5, s=30)
          cnt += len(c)

        plt.tight_layout()

        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image

    gif_dir = Path("..", "..", "gifs", "t-digest")
    gif_dir.mkdir(parents=True, exist_ok=True)
    gif_file = str(gif_dir / "cluster-building-with-scale-function.gif")
    imageio.mimsave(gif_file, [plot_cluster_and_k_function(i) for i in range(n_points)], fps=2) 
  

def approximate_cdf():
  np.random.seed(2718)
  n_points = 25
  delta = 10
  points = 2 * np.random.randn(n_points) + 0.5

  data_clusters = cluster_points(points, delta)

  img_dir = Path("..", "..", "images", "t-digest")
  img_dir.mkdir(exist_ok=True, parents=True)
  img_file = str(img_dir / "simple_approximate_histogram.png")

  plt.subplot(211)
  # plot the empirical histogram
  plt.hist(points, bins=100, histtype="step", cumulative=True, density=True)

  # plot the approximate histogram by assuming the cluster centroid is the median
  data_fraction_per_cluster = [len(c) / n_points for c in data_clusters]
  right_endpoint_cluster_percentiles = np.cumsum(np.row_stack([0] + data_fraction_per_cluster))
  centroid_percentiles = (right_endpoint_cluster_percentiles[1:] + right_endpoint_cluster_percentiles[:-1]) / 2

  x = np.array([np.mean(c) for c in data_clusters])
  plt.plot(x, centroid_percentiles, "o-", label="Approximate")
  plt.title("CDF(x) vs x")
  plt.ylabel("CDF")
  plt.legend()
  plt.xlim([np.min(points), np.max(points)])
  plt.xticks([])

  plt.subplot(514)
  plt.scatter(points, np.zeros_like(points))
  plt.xlabel("x")
  plt.yticks([])

  plt.savefig(img_file)


if __name__ == "__main__":
  import matplotlib
  matplotlib.use("TkAgg")
  import matplotlib.pyplot as plt
  import imageio

  # clustering_with_scale_function_animation()
  approximate_cdf()
  # profiling_example()
  # plot_2d_example()
  # plot_3d_example()
  #   plot_splash_image()
    