import numpy as np

class SimulationManager():
    """
    A general purpose manager class, which manages several distinct simulation runs. It allows
    for easily obtaining mean, and std values of an attribute.
    """

    def __init__(self, class_type, n_instances, *args, **kwargs):
        """
        Initializes the manager.

        Arguments:
            class_type (type): The class type. The class needs to have a 'run' function.
            n_instances (int): The number of distinct simulations.
        """
        
        self.class_type = class_type
        self.n_instances = n_instances
        
        self.args = args
        self.kwargs = kwargs
    
    def calc_attribute(self, attribute, run=True, show_progress=True):
        """ Returns the mean and std of the given attribute. """

        values = []
        for i in range(self.n_instances):
            
            instance = self.class_type(*self.args, **self.kwargs)
            if show_progress:
                print(f"{i / self.n_instances * 100}% {instance}                        ", end="\r")
            if run:
                instance.run()

            values.append(attribute(instance))

        return np.mean(values), np.std(values)
