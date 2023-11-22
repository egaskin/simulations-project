from code_scripts import initialization, visualization, simulate

if __name__ == "__main__":

    """ GENERATE RSA AND INITALIZE STATES"""
    # RSA inputs
    x_lim = 10
    y_lim = x_lim
    boundaries = [(-x_lim, x_lim), (-y_lim, y_lim)]
    num_dim = 2
    number_pts = 1000

    # perform the RSA
    small_set, RSA_center = perform_rsa(boundaries, num_dim, number_pts)
    x_vals = small_set[:,0]
    y_vals = small_set[:,1]

    # create Voronoi and Delaunay plot with overlay
    tri = Delaunay(small_set)
    init_states, center_of_tumor_idx, center_of_tumor_pt = create_initial_states(number_pts, num_dim, RSA_center=RSA_center, tri=tri)

    adjacency_mtx = create_adjacency_matrix_from_simplices(simplices=tri.simplices,number_pts=number_pts)


    """ INITIALIZE MODEL PARAMETERS """
    a_2d = 0.9
    # b_2d = 1.1
    b_2d = 2
    p_0_2d = 0.6
    R_max_2d = 37.5

    a_3d=0.42
    b_3d = 0.11
    p_0_3d = 0.192
    R_max_3d = R_max_2d

    """ RUN MODEL """
    all_timesteps = simulate_system(num_dim=2,t=100,init_states=init_states,a=a_2d,
                                b=b_2d,tri=tri,boundary=boundaries,R_max=R_max_2d,
                                p_0=p_0_2d,adjacency_mtx = adjacency_mtx,
                                center_of_tumor_pt = center_of_tumor_pt,
                                center_of_tumor_idx = center_of_tumor_idx)
    
    """ SHOW RESULTS """

    for idx, timestep_states in enumerate(all_timesteps):
        # plot every 10th timestep, starting with first
        if idx % 10 == 0:
            plot_current_timestep(cur_timestep_states=timestep_states,
                                  all_rsa_points=small_set)