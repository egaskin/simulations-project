import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi, voronoi_plot_2d


def plot_current_timestep(states_to_labels_dict,cur_timestep_states, all_rsa_points):
  cell_labels = [states_to_labels_dict[state_val] for state_val in cur_timestep_states]
  fig, ax = plt.subplots(1,figsize=(8, 4))
  x_vals = all_rsa_points[:,0]
  y_vals = all_rsa_points[:,1]

  color_map = {"necrotic-core": "dimgray", "proliferative": "lightcoral", "non-proliferative": "firebrick", "non-tumorous": "cornflowerblue"}
  vor = Voronoi(all_rsa_points)
  v_fig = voronoi_plot_2d(vor, show_vertices = False, ax = ax, show_points = False, line_colors = 'steelblue')

  for r in range(len(vor.point_region)):
      region = vor.regions[vor.point_region[r]]
      if not -1 in region:
          polygon = [vor.vertices[i] for i in region]
          plt.fill(*zip(*polygon), color=color_map[cell_labels[r]])

  ax.plot(x_vals, y_vals, 'ok',markersize=1)

  # Create custom legend for cell categories
  legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10)
                    for label, color in color_map.items()]

  plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',handles=legend_elements, title='Cell Types')
  plt.tight_layout()
  plt.show()