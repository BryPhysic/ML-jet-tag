from Functions.functions import NPZDatasetManager,ClassDistributionPlotter,BalancedDatasetSampler
# call del clas NPZDatasetManager for manage the npz files
manager = NPZDatasetManager("Data")
# select de npz file for load the data
my_dataset = manager.load_npz_by_index(0)
(my_dataset['jetClass'])
#load the x and y del dataset
var_x = my_dataset['jetImage']
var_y = my_dataset['jetClass']

plotter = ClassDistributionPlotter(var_y)

#First create a Dictionary for the data 
#dict of particle
particles = {
    "gluons": [1, 0, 0, 0, 0],
    "quarks": [0, 1, 0, 0, 0],
    "Ws":     [0, 0, 1, 0, 0],
    "Zs":     [0, 0, 0, 1, 0],
    "tops":   [0, 0, 0, 0, 1]
}
particle_dict = {
    "gluons": 0,
    "quarks": 1,
    "Ws":     2,
    "Zs":     3,
    "tops":   4
}

class_names = ["Gluons", "Quarks", "Ws", "Zs", "Tops"]
# Map labels to class names
plotter.map_labels(particle_dict)
# Plot the distribution
#plotter.plot_distribution() 


balanced_sampler = BalancedDatasetSampler(var_x, var_y)
balanced_sampler.balance_dataset(shuffle=False)
images = balanced_sampler.balanced_images
labels = balanced_sampler.balanced_labels

plotter = ClassDistributionPlotter(labels)
# Map labels to class names
plotter.map_labels(particle_dict)
# Plot the distribution
plotter.plot_distribution()