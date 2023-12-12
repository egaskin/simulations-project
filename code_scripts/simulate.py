import copy
import numpy as np

def simulate_system(num_dim,t,init_states,a,b,tri,R_max,p_0,adjacency_mtx,center_of_tumor_pt,center_of_tumor_idx):
  all_results = [init_states]

  if num_dim != len(center_of_tumor_pt) or num_dim != len(tri.points[0]):
    error_string = f"""
    ERROR: Dimensions DO NOT MATCH
    center_of_tumor_pt = {center_of_tumor_pt}
    tri.points[0] = first_pt = {tri.points[0]}
    BUT
    num_dim = {num_dim}
    """
    raise Exception(error_string)

  # for our purposes, we only deal with 2D or 3D
  if num_dim == 2:
    calc_delta_p_func = calc_delta_p_2D
    calc_delta_n_func = calc_delta_n_2D
  elif num_dim == 3:
    calc_delta_p_func = calc_delta_p_3D
    calc_delta_n_func = calc_delta_n_3D
  #Get the coordinates of the centers of all cells in the triangulation
  all_points = tri.points
  # #In the initial state, only one cell is proliferative, find its index and coordinates
  # center_of_tumor_idx = np.where(init_states==1)[0][0]
  # center_of_tumor = tumor_center
  #Range over time steps
  for timestep in range(t):
    #Calculate R_t
    R_t = calc_Rt(center_of_tumor_pt,all_points,init_states,adjacency_mtx)
    check_no_prolif_cells = (R_t == np.inf)

    if check_no_prolif_cells:
      break # exit timesteps early

    # CURRENT TIMESTEP PARAMETERS:
    #Create a copy of the current_states. This will be updated in this time step
    new_states = copy.deepcopy(init_states)
    #Calculate delta_n
    delta_n = calc_delta_n_func(a,R_t)
    #If it can spread, calculate delta_p
    delta_p = calc_delta_p_func(b,R_t)
    # calculate necrotic region radius
    necrotic_radius = R_t - delta_p - delta_n

    #Iterate over all cells in the current time step
    for cell_idx, cell in enumerate(init_states):

      #Get the coordinates of the current cell
      cell_pt = all_points[cell_idx]
      #Calculate the distance of the current cell from tumor center r_i
      r_i = np.linalg.norm(center_of_tumor_pt - cell_pt)

      #If cell is non-proliferative, check if it will turn necrotic
      if ((cell == 2) & (r_i < necrotic_radius)):
        new_states[cell_idx] = 3
        #Else if the cell is proliferative, check if it can spread
      elif cell == 1:
        #Check probabilistically if it is allowed to spread.
        if np.random.random() < (calc_prob_division(p_0,r_i,R_max)):
          #Find neighbors of this cell in NEW BOARD
          neighbor_idxs = np.where(adjacency_mtx[cell_idx,:]==1)[0]
          # neighbor_states = new_states[neighbor_idxs]
          neighbor_xy = all_points[neighbor_idxs]
          #Iterate over neighbors and check if any neighbor is within delta_p distance
          divide = False
          for nbr_idx, nbr_pt in zip(neighbor_idxs,neighbor_xy):
            # print("np.linalg.norm(cell_pt - nbr_pt)",np.linalg.norm(cell_pt - nbr_pt),
            #       "delta_p",delta_p)
            if np.linalg.norm(cell_pt - nbr_pt) < delta_p:
              #If it is, check what the state of the neighbor is
              nbr_cell_state = init_states[nbr_idx]
              if nbr_cell_state == 0:#neighbor is a non-tumor cell
                new_states[nbr_idx] = 1#Infect
                divide = True
                break
          if not divide:
            new_states[cell_idx] = 2#It cannot infect any cell, make current cell non-proliferative <- Check the logic here
    # save the updated lattice for next timestep
    init_states = new_states
    # save the result after updating all the cells from previous timestep
    all_results.append(init_states)

  if check_no_prolif_cells:
      print(f"No more proliferative cells left, stopped at timestep={timestep}")
  return all_results

def calc_delta_p_3D(b,R_t):
  delta_p = b*(R_t** (2/3))
  return delta_p

def calc_delta_n_3D(a,R_t):
  delta_n = a*(R_t** (2/3))
  return delta_n

def calc_delta_p_2D(b,R_t):
  delta_p = b*(R_t** (1/2))
  return delta_p

def calc_delta_n_2D(a,R_t):
  delta_n = a*(R_t** (1/2))
  return delta_n

def calc_prob_division(p_0,r_i,R_max):
  p_d = p_0*(1-(r_i/R_max))
  return p_d

def get_proliferative_cell_idxs(cur_state_array):
  idxs = np.where(cur_state_array == 1 )[0]
  return idxs#"BLAH"

def calc_Rt(tumor_center, all_points, cur_state_array, adjacency_mtx):
  # print("cur_state_array",cur_state_array,"\n\n")
  # get all the proliferative cell indices and points
  proliferative_cell_idxs = get_proliferative_cell_idxs(cur_state_array)
  proliferative_cell_pts = all_points[proliferative_cell_idxs]

  # edge points are proliferative cells with at least one healthy cell (state 0) as a neighbor
  edge_pts = []
  for cell_idx, cell_pt in zip(proliferative_cell_idxs, proliferative_cell_pts):
    neighbor_idxs = np.where(adjacency_mtx[cell_idx,:]==1)[0]

    # print("neighbor_idxs",neighbor_idxs,"\n\n")

    if 0 in cur_state_array[neighbor_idxs]:
      edge_pts.append(cell_pt)
  edge_pts = np.array(edge_pts)
  # print("edge_pts",edge_pts,"\n\n")
  N_p = len(edge_pts)

  # initialize variable to track running sum
  sums = 0

  for cell_pt in edge_pts:
    r_i = np.linalg.norm(cell_pt - tumor_center)

    # handle the edge case where the center is the only proliferative cell
    # in the boundary proliferative cells
    if r_i == 0 and len(edge_pts) == 1:
      r_i = 1
      N_p = 1

    # print("tumor_center",tumor_center)
    sums += r_i

  if N_p == 0:
    return np.inf

  return sums/N_p