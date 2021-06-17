from pathlib import Path

import pandas as pd
import numpy as np


IMG_DIR = Path("..", "..", "images", "t-digest")
IMG_DIR.mkdir(exist_ok=True, parents=True)
GIF_DIR = Path("..", "..", "gifs", "t-digest")
GIF_DIR.mkdir(parents=True, exist_ok=True)

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

  def update(self, *clusters):
    new_weight = self.weight + sum(c.weight for c in clusters)
    new_mean = (self.centroid * self.weight + sum(c.centroid * c.weight for c in clusters)) / new_weight

    self._n = new_weight
    self._mean = new_mean

  def __add__(self, other: "Cluster"):
    new_weight = self.weight + other.weight
    new_mean = (self.centroid * self.weight + other.centroid * other.weight) / new_weight
    new_cluster = Cluster(new_mean, new_weight)

    return new_cluster
  
  def __repr__(self):
    centroid_arg = f"centroid={self.centroid}"
    weight_arg = f"weight={self.weight}"

    return f"{self.__class__.__name__}({centroid_arg}, {weight_arg})"


class TDigest:
  def __init__(self, delta: float, points: np.array):
    """
    """
    self.delta = delta
    self.clusters = self.cluster_points(points)
  
  def cluster_points(self, points: np.array):
    total_weight = len(points)
    percentile_increment = 1 / total_weight

    sorted_points = np.sort(points)

    k_limit = scale_fn(0, self.delta) + 1
    q_limit = inverse_scale_fn(k_limit, self.delta)

    left_cluster_index = 0
    right_cluster_index = 0
    data_clusters = []

    for j in range(total_weight):
        percentile = (j + 1) * percentile_increment
        if percentile > q_limit:
          right_cluster_index = j + 1

          cluster_points = sorted_points[left_cluster_index:right_cluster_index]
          cluster_centroid = np.mean(cluster_points)
          cluster_weight = right_cluster_index - left_cluster_index
          cluster = Cluster(centroid=cluster_centroid, weight=cluster_weight)
          data_clusters.append(cluster)
          
          left_cluster_index = right_cluster_index
          k_limit = scale_fn(percentile, self.delta) + 1
          q_limit = inverse_scale_fn(k_limit, self.delta)

    return data_clusters

  def update(self, *tdigests):
    # what to do about current centroid??
    self.clusters += [c for digest in tdigests for c in digest._clusters]
    self.clusters.sort(key=lambda c: c.centroid)

    total_weight = sum(c.weight for c in self.clusters)

    k_limit = scale_fn(0, self.delta) + 1
    q_limit = inverse_scale_fn(k_limit, self.delta)

    left_cluster_index = 0
    right_cluster_index = 0
    data_clusters = []

    percentile = 0
    for j, cluster in enumerate(self.clusters):
      percentile += cluster.weight / total_weight
      if percentile > q_limit:
        right_cluster_index = j

        clusters_to_merge = self.clusters[left_cluster_index:right_cluster_index]
        merged_cluster_weight = sum(c.weight for c in clusters_to_merge)
        merged_cluster_centroid = sum(c.centroid * c.weight / merged_cluster_weight for c in clusters_to_merge)
        cluster = Cluster(centroid=merged_cluster_centroid, weight=merged_cluster_weight)
        data_clusters.append(cluster)
        
        left_cluster_index = right_cluster_index
        k_limit = scale_fn(percentile, self.delta) + 1
        q_limit = inverse_scale_fn(k_limit, self.delta)

    self._clusters = data_clusters
  
  def cdf(self, x):
    total_weight = sum(c.weight for c in self.clusters)

    prev_percentile = 0
    percentile = 0
    prev_weight = 0
    for i, cluster in enumerate(self.clusters):
      if cluster.weight == 1:
        percentile += 1 / total_weight
        prev_weight = 0
      else:
        percentile += (cluster.weight / 2 + prev_weight / 2) / total_weight
        prev_weight = cluster.weight

      if x < cluster.centroid:
        if i == 0:
          return 0.
        delta_x = abs(cluster.centroid - prev_cluster.centroid)
        delta_y = abs(percentile - prev_percentile)
        m = delta_y / delta_x

        return prev_percentile + m * abs(x - prev_cluster.centroid)

      prev_percentile = percentile
      prev_cluster = cluster

    return 1.


def plot_splash_image():
  pass


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

    gif_file = str(GIF_DIR / "cluster-building-with-scale-function.gif")
    imageio.mimsave(gif_file, [plot_cluster_and_k_function(i) for i in range(n_points)], fps=2) 
  

def approximate_cdf():
  np.random.seed(2718)
  n_points = 25
  delta = 10
  points = 2 * np.random.randn(n_points) + 0.5

  data_clusters = cluster_points(points, delta)

  img_file = str(IMG_DIR / "simple-approximate-histogram.png")

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


def randomly_cluster_points(points, n_clusters):
  sorted_points = np.sort(points)
  clusters = []
  curr_cluster = [sorted_points[0]]

  boundaries = np.random.choice(np.arange(points.size), size=n_clusters) + 1
  boundaries = np.sort(boundaries)
  boundaries = np.hstack([boundaries, points.size])

  cnt = 0
  for i, p in enumerate(sorted_points[1:]):
    if i < boundaries[cnt]:
      curr_cluster.append(p)
    else:
      clusters.append(curr_cluster)
      curr_cluster = [p]
      cnt += 1
  clusters.append(curr_cluster)
      
  return [c for c in clusters if len(c) > 0]

def arbitrary_clustering_examples():
  np.random.seed(2718)
  n_points = 25
  delta = 10
  points = 2 * np.random.randn(n_points) + 0.5 
  colors = {
      0: 'b',
      1: 'g',
      2: 'y',
      3: 'orange',
      4: 'r',
  }

  n_colors = len(colors)

  for i in range(3):
    plt.subplot(3, 1, i + 1)
    clusters = randomly_cluster_points(points, n_clusters=1 + 3)
    # print(sum(len(c) for c in clusters))
    for j, c in enumerate(clusters):
      plt.scatter(c, np.zeros_like(c), alpha=0.3, color=colors[j % n_colors], s=75)
    plt.yticks([])
  fname = str(IMG_DIR / "strongly-ordered-clusters.png") 
  plt.tight_layout()
  plt.savefig(fname)


def plot_scale_functions():
  q = np.linspace(0.0001, 0.999, 1000)
  for delta in [5, 10, 15]:

    plt.subplot(211)
    plt.plot(q, delta / TAU * np.arcsin(2 * q - 1), label=rf"$\delta={delta}$")
    plt.title(rf"$k_1(q)$ vs $q$")
    # plt.xlabel(r"$q$")
    plt.ylabel(r"$k_1(q)$")
    leg = plt.legend()
    leg.get_frame().set_linewidth(0.0)
    plt.tick_params(bottom=False, labelbottom=False)
    plt.tight_layout()

    plt.subplot(212)
    plt.plot(q, delta/ (4 * np.log(1 / delta) + 21) * np.log(q/(1-q)), label=rf"$\delta={delta}$")
    plt.title(rf"$k_2(q)$ vs $q$")
    plt.xlabel(r"$q$")
    plt.ylabel(r"$k_2(q)$")
    leg = plt.legend()
    leg.get_frame().set_linewidth(0.0)
    plt.tight_layout()

  fname = str(IMG_DIR / "scale-functions.png") 
  plt.savefig(fname)


def plot_weakly_ordered_cluster():
  np.random.seed(3141)
  # plt.subplot(3,1,2)
  plt.figure(figsize=(8,2))
  x = np.sort(np.random.randn(20))
  plt.scatter(x[:10], np.zeros_like(x[:10]), color='b', alpha=0.5, s=65)
  plt.scatter(x[10:], np.zeros_like(x[10:]), color='g', alpha=0.5, s=65)
  plt.scatter(np.mean(x[:10]), 0, color='b', s=100, marker="x")
  plt.scatter(np.mean(x[10:]), 0, color='g', s=100, marker="x")

  x = np.random.randn(10) + 1.5
  plt.scatter(x, np.zeros_like(x), color='y', alpha=0.4, s=65)
  plt.scatter(np.mean(x), 0, color='y', s=100, marker="x")
  plt.yticks([])

  fname = str(IMG_DIR / "weakly-ordered-cluster.png") 
  plt.tight_layout()
  plt.savefig(fname)


def plot_cdf_examples():
  x1 = np.random.randn(1000)
  x2 = np.random.randn(1000) + 3
  x3 = np.random.randn(1000) + 7

  x = np.hstack([x1, x2, x3])
  delta1 = 25
  delta2 = 50
  delta3 = 75

  t1 = TDigest(delta1, x)
  t2 = TDigest(delta2, x)
  t3 = TDigest(delta3, x)

  plt.figure(figsize=(16,10))

  plt.subplot(121)
  plt.hist(x, density=True, bins=100)


  plt.subplot(322)
  centroids = [c.centroid for c in t1.clusters]
  q_approx = [t1.cdf(c) for c in centroids]
  plt.plot(centroids, q_approx, alpha=0.7)
  plt.scatter(centroids, q_approx)
  plt.hist(x, density=True, bins=100, cumulative=True, histtype="step")
  plt.xlim([np.min(x), np.max(x)])
  plt.xticks([])
  plt.title(rf"Approx CDF ($\delta$ = {delta1})")


  plt.subplot(324)
  centroids = [c.centroid for c in t2.clusters]
  q_approx = [t2.cdf(c) for c in centroids]
  plt.plot(centroids, q_approx, alpha=0.7)
  plt.scatter(centroids, q_approx)
  plt.hist(x, density=True, bins=100, cumulative=True, histtype="step")
  plt.xlim([np.min(x), np.max(x)])
  plt.xticks([])
  plt.title(rf"Approx CDF ($\delta$ = {delta2})")


  plt.subplot(326)
  centroids = [c.centroid for c in t3.clusters]
  q_approx = [t3.cdf(c) for c in centroids]
  plt.plot(centroids, q_approx, alpha=0.7)
  plt.scatter(centroids, q_approx)
  plt.hist(x, density=True, bins=100, cumulative=True, histtype="step")
  plt.xlim([np.min(x), np.max(x)])
  plt.title(rf"Approx CDF ($\delta$ = {delta3})")

  fname = str(IMG_DIR / "approximate-cdf.png") 
  plt.tight_layout()
  plt.savefig(fname)


if __name__ == "__main__":
  import matplotlib
  matplotlib.use("TkAgg")
  import matplotlib.pyplot as plt
  import imageio

  # clustering_with_scale_function_animation()
  # arbitrary_clustering_examples()
  plot_weakly_ordered_cluster()
  # plot_scale_functions()

  # plot_cdf_examples()
  # profiling_example()
  # plot_2d_example()
  # plot_3d_example()
  # plot_splash_image()
    