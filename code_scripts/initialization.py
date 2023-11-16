import numpy as np
import random
from itertools import combinations # https://docs.python.org/3/library/itertools.html#itertools.combinations

def perform_rsa(boundaries, num_dim, number_pts, min_dist_func = None,
                dist_func = np.linalg.norm):
  """
  Function: perform random sequential addition (RSA) to generate n RSA points

  Input:
    - boundaries (list of tuples): each tuple in array has the form
      (i_low_lim, i_high_lim) where i_low_lim and i_high_lim represents the
      lower and upper boundary of the ith dimension, respectively

    - num_dim (int): an explicit parameter ensuring that the number of
      dimensions in boundaries is as intended (will throw error otherwise)
    - number_pts (integer): number of points to generate
    - min_dist_func (func): minimum distance between randomly generated points
      (could be a constant or pointer to function)
    - dist_func (pointer to function): distance function/metric to use to find
      distance between points

  Output:
    - all_rsa_pts (float array with dim (number_pts x num_dim)):
      each list is a point generated by RSA
  """

  # check user knows what dimensions they're inputting
  if len(boundaries) != num_dim:
      raise Exception("num dim doesnt agree")

  # for our purposes, we only deal with 2D or 3D
  if num_dim == 2:
    min_dist_func = min_dist_2D
    # if min_dist_func != min_dist_2D:
    #   print("Are you sure you don't mean to use min_dist_2D since num_dim=2?")
  elif num_dim == 3:
    min_dist_func = min_dist_3D

  # generate an array to contain all the points at once
  # first column is x coordinate of ith point/row
  # second column is y coordinate of ith point/row
  # third column ...
  all_rsa_pts = np.empty(shape=(number_pts,num_dim))
  center = np.empty(shape=(num_dim))
  # for the first point/row, assign valid random values based on boundaries
  for i in range(0, num_dim):
    all_rsa_pts[0,i] = random.uniform(boundaries[i][0],boundaries[i][1])
    center[i] = (boundaries[i][0] + boundaries[i][1]) / 2

  # the last valid point created is the first one, initialized above.
  # while the next point created is not valid, continue to recreate it, once
  # it is valid, then update index_last_valid_pt by 1 and repeat until
  # index_last_valid_pt = number_pts-1
  # which entails that the last row/element/point in the all_rsa_pts is valid
  # i.e. all points are now valid
  index_last_valid_pt = 0
  # generate the next random pt
  while index_last_valid_pt < number_pts-1:
    cur_pt_idx = index_last_valid_pt+1

    # select a slice from the initialized array (a numpy "view")
    cur_rand_pt = all_rsa_pts[cur_pt_idx,:]

    for i in range(0,num_dim):
      # generate the value for dimension i
      cur_rand_pt[i] = random.uniform(boundaries[i][0],boundaries[i][1])

    # ensure that point's distance to already-present points is acceptable,
    # range over all the valid points so far
    acceptable_dist = True
    for other_pt_idx in range(0,index_last_valid_pt+1):

      # never compare the point against itself, otherwise infinite loop
      if other_pt_idx == cur_pt_idx:
        continue

      # slice out the other point
      other_pt = all_rsa_pts[other_pt_idx,:]

      if dist_func(other_pt - cur_rand_pt) < min_dist_func(cur_rand_pt,center):
          # stop early, if one point is too close, then this cur_rand_pt is not
          # acceptable
          acceptable_dist = False
          break

    # add the point if its valid
    if acceptable_dist:
      # a valid point has been selected, advance to the next index
      index_last_valid_pt += 1

  return all_rsa_pts, center

def min_dist_2D(pt, center):
  return 0.146 * np.linalg.norm(pt - center) ** (1/3)

def min_dist_3D(pt, center):
  return 0.146 * np.linalg.norm(pt - center) ** (2/3)


def create_all_pairs_from_simplex(simplex):
  # if we let length of simplex = s, then to get all pairs we have
  # sC2 which is read as "s choose 2" in combinatorics
  return combinations(simplex,2)

def create_adjacency_matrix_from_simplices(simplices,number_pts):
  """
  Input: 
  - simplices (np.ndarray): contains all the points defining the vertices of the 
  triangulation which in turn defines nearest neighbors (output from 
  Delaunay(set of points))
  - number_pts (int): number of points in the consideration

  Output: 
  - adjacency matrix (np.ndarray): dimensions (number_pts x number_pts),
  with a 1 at (row_idx, col_idx) and (col_idx,row_idx) then the point 
  represented by row_idx and the point represented by col_idx are neighbors in
  the Delunay Triagnulation
  """
  # initialize adjacency matrix (avoid doing empty this time,
  # since if they are NOT neighbors, then we want zeros)
  adjacency_mtx = np.zeros(shape=(number_pts,number_pts))

  # range over all the simplices
  for simplex in simplices:

      # look at each neighbor pair within the simplex
      for neighbor_pair in create_all_pairs_from_simplex(simplex):
        point_1_idx = neighbor_pair[0]
        point_2_idx = neighbor_pair[1]

        # assign 1 to the entry corresponding to point_1 and point_2
        adjacency_mtx[point_1_idx,point_2_idx] = 1
        # with our defintion of "neighbors"
        # the adjacency matrix is symmetrical. if point_1 is neighbor of point_2
        # then that means point_2 is a neighbor of point_1 (note for kNN this 
        # may not be true in general, so neighbors must be defined carefully)
        adjacency_mtx[point_2_idx,point_1_idx] = 1

  return adjacency_mtx

def get_point_nearest_center(all_pts,tumor_center):
  min_dist_from_center = np.inf
  best_idx = None
  for idx, pt in enumerate(all_pts):
    # calculate distance between point and center
    if np.linalg.norm(pt - tumor_center) < min_dist_from_center:
      best_idx = idx

  return best_idx

def create_initial_states(number_pts, num_dim, tumor_center, tri):
  """
  DESCRIPTION: create the array describing the state for each initial point. the
  ith index in the tri.points attribute is defined by the state at the ith 
  element in the cell_states_array. see note below on states

  NOTE, STATES ARE DEFINED AS FOLLOWS:
  - 0: empty cellular automaton cell (non-tumorous biological cells)
  - 1: cancerous, proliferative cellular automaton cell
  - 2: cancerous, non-proliferative, non-necrotic cellular automaton cell
  - 3: cancerous, necrotic cellular automaton cell

  INPUT: 
  - number_pts (int): number of cellular automaton cells
  - num_dim (array): number of dimensions for the cells. i.e. 2d or 3d
  - tumor_center (array): point indicating the center of the tumor

  OUTPUT: 
  - cell_states_array (ndarray): size (1 x number_pts), where the ith entry 
  corresponds to the ith point's state
  """

  cell_states_array = np.zeros(shape=(number_pts))

  all_pts = tri.points

  # assign the first progenitor cell
  pt_idx_nearest_to_center = get_point_nearest_center(all_pts,tumor_center)

  cell_states_array[pt_idx_nearest_to_center] = 1

  return cell_states_array