import random
import Fitness
import Functions
import copy
import numpy as np
from collections import deque
from operator import attrgetter

'''' 
Piyush Edit: Feel free to comment out the os section. It is just for my section.
'''
import os
if os.path.isfile('population.txt'):
    os.remove('population.txt')
if os.path.isfile('fitness_file.txt'):
    os.remove('fitness_file.txt')
if os.path.isfile('population_each_gen.txt'):
    os.remove('population_each_gen.txt')
if os.path.isfile('population_mut_gen.txt'):
    os.remove('population_mut_gen.txt')



class GP_Tree():
    """Genetic-Programming Syntax Tree created for performing
the common GP operations on trees. This tree is traversed
and represented according to Depth-First-search (DFS) traversal.

Paramaters and functions used to describe the tree are described
as follows:"""

    # Node class of the tree which contains the terminal or the function with its children.
    class _Node:

        def __init__(self, data):

            # Data is the function or the terminal constant
            self.data = data

            # Parent of every node will also be provided (Not implemented yet).
            self.parent = None

            # Its size would be equal to the arity of the function. For the terminals, it would be none
            self.children = None

        # This is overriden to define the representation of every node. The function is called recursively
        # to build the whole representation of the tree.
        def __str__(self):

            # "retval" is the final string that would be returned. Here the content of the node is added.
            retval = str(self.data) + "=>"

            # Content of the children of the node is added here in retval.
            if (self.children != None):
                for child in self.children:
                    retval += str(child.data) + ", "

            retval += "END\n"  # After every node and its children, string is concatenated with "END"

            # Recursive calls to all the nodes of the tree.
            if (self.children != None):
                for child in self.children:
                    retval += child.__str__()

            return retval

    def __init__(self, function_set, terminal_set, num_features, min_depth, max_depth, fitness_metric):
        '''The constructor accepts the max and min depth values of
        the tree.'''

        self.function_set = function_set
        # A list of functions to be used. Custom functions can be created.'

        self.terminal_set = terminal_set
        # List of floating point or zero arity functions acting as the terminals
        # of the tree

        self.num_features = num_features
        # Specifies the num of features in the input file

        self.min_depth = min_depth
        # Specifies the minimum depth of the tree.

        self.max_depth = max_depth
        # Specifies the maximum depth of the tree.

        self.fitness_metric = fitness_metric
        # The fitness function metric to be used to calculate fitness.

        ###################################################################
        # Other parameters :
        ###################################################################

        self.root = None
        # This is the root of the tree. It is from here, that the tree is traversed by every member function.

        self.fitness = None
        # The fitness value of the syntax tree

        self.number_of_terminals = 0
        self.number_of_functions = 0
        # These parameters are required for calculating the "terminal_ratio" in generation methods.

        self._add_features_in_terminal_set(prefix="X")
        # Features are added in the final terminal_set

    # this returns the string representation of the root which builds the representation of the whole tree recursively.
    def __str__(self):
        return self.root.__str__()


    @property
    def terminal_ratio(self):
        # Returns the ratio of the number of terminals to the number of all the functions in the tree.
        return self.number_of_terminals / float(self.number_of_terminals + self.number_of_functions)

    # Adds the number of arguments as specified in num_features in the syntax tree. The arguments is prefixed
    # with "X" followed by the index number. Eg: X0, X1, X2 ....
    def _add_features_in_terminal_set(self, prefix):

        temp_list = []
        for i in range(self.num_features):
            feature_str = "{prefix}{index}".format(prefix=prefix, index=i)
            temp_list.append(feature_str)

        temp_list.extend(self.terminal_set)
        self.terminal_set = temp_list

    #####################################################################################
    #                            Tree Generation Methods                                #
    #####################################################################################

    # The main tree generation function. Recursive function that starts building the tree from the root and returns the root of the constructed tree.
    def _generate(self, condition, depth, height):

        node = None  # The node that would be returned and get assigned to the root of the tree.
        # See functions: 'generate_full' and 'generate_grow' for assignment to the root of the tree.

        # Condition to check if currently function is to be added. If the condition is false, then the terminal
        # is not yet reached and a function should be inserted.
        if (condition(depth, height) == False):
            node_data = random.choice(self.function_set)  # Randomly choosing a function from the function set
            node_arity = Functions.get_arity(
                node_data)  # Getting the arity of the function to determine the node's children

            node = GP_Tree._Node(node_data)  # Creating the node.
            self.number_of_functions += 1

            node.children = []  # Creating the empty children list
            for _ in range(node_arity):
                child = self._generate(condition, depth + 1, height)
                child.parent = node
                node.children.append(child)  # Children are added recursively.

        else:  # Now the terminal should be inserted
            node_data = random.choice(self.terminal_set)  # Choosing the terminal randomly.
            node = GP_Tree._Node(node_data)  # Creating the terminal node
            self.number_of_terminals += 1
            node.children = None  # Children is none as the arity of the terminals is 0.

        return node  # Return the node created.

    def generate_full(self):
        # The method constructs the full tree. Note that only the function 'condition' is different in the
        # 'generate_grow()' and 'generate_full()' methods.

        def condition(depth, height):
            return depth == height

        height = random.randint(self.min_depth, self.max_depth)
        self.root = self._generate(condition, 0, height)

    def generate_grow(self):
        # The method constructs a grown tree.

        def condition(depth, height):
            return depth == height or (depth >= self.min_depth and random.random() < self.terminal_ratio)

        height = random.randint(self.min_depth, self.max_depth)
        self.root = self._generate(condition, 0, height)

    def generate_half_and_half(self):
        # Half the time, the expression is generated with 'generate_full()', the other half,
        # the expression is generated with 'generate_grow()'.

        # Selecting grow or full method randomly.
        method = random.choice((self.generate_grow, self.generate_full))
        # Returns either a full or a grown tree
        method()

    #####################################################################################
    #    Tree Traversal Methods: Different ways of representing the tree expression     #
    #####################################################################################

    # Depth-First-Traversal of the tree. It first reads the node, then the left child and then the right child.
    def tree_expression_DFS(self):

        expression = []  # expression to be built and returned

        # Recursive function as an helper to this function.
        self._tree_expression_DFS_helper(self.root, expression)

        return expression

    # Helper recursive function needed by the function "tree_expression_DFS()".
    def _tree_expression_DFS_helper(self, node, expression):

        expression.append(node.data)  # Expression to be built.

        if (node.children != None):
            for child in node.children:  # Adding children to the expression recursively.
                self._tree_expression_DFS_helper(child, expression)

        return

        # Breadth-First-Traversal of the tree. It first reads the left child, then the node itself and then the right child.

    def tree_expression_BFS(self):
        q = deque()  # BFS is implemented using a queue (FIFO)
        expression = []  # Expression to be built and returned.

        # Adding root to the queue
        node = self.root
        q.append(node)

        while (q):
            popped_node = q.popleft()
            if (popped_node.children != None):
                for child in popped_node.children:
                    q.append(child)

            expression.append(popped_node.data)

        return expression

    #####################################################################################
    #                            Tree Evaluation Methods                                #
    #####################################################################################

    # This function evaluates the syntax tree and returns the value evaluated by the tree.
    def evaluate(self, X_Data):

        # X_Data : shape = [n_samples, num_features]
        # Training vectors, where n_samples is the number of samples and num_features is the number of features.

        # Return: Y_Pred: shape = [n_samples]
        # Evaluated value of the n_samples.

        Y_Pred = []

        for features in X_Data:
            if features.size != self.num_features:
                print("features.size "+str(features.size))
                print("num_features "+str(self.num_features))
                raise ValueError("Number of input features in X_Data is not equal to the parameter: 'num_features'.")

            Y_Pred.append(self._evaluate_helper(self.root, features))

        return np.array(Y_Pred)

    # Helper function for the func: "evaluate()". This makes recursive calls for the evaluation.
    def _evaluate_helper(self, node, X_Data):

        # Terminal nodes
        if (node.children == None):

            if isinstance(node.data, str):
                feature_name = node.data
                index = int(feature_name[1:])
                return X_Data[index]

            else:
                return node.data

        args = []  # Will contain the input arguments i.e the children of the function in the tree.

        for child in node.children:
            args.append(self._evaluate_helper(child, X_Data))  # Evaluation by the recursive calls.

        func = Functions.get_function(node.data)  # Get the function from the alias name
        return func(*args)  # Return the computed value

    #####################################################################################
    #                            Tree Fitness Measure Method                            #
    #####################################################################################

    # Function to evaluate the fitness of the tree using the specified metric for it.
    def evaluate_fitness(self, X_Data, Y_Data, Weights=None):

        # X_Data: np-array. Shape = (num_features, n_samples). Specifies the features of the samples.
        # Y_Data: np-array. Shape = (n_samples). These are the target values of n_samples.
        # Weights: np-array. Shape = (n_sample). Weights applied to individual sample

        Y_Pred = self.evaluate(X_Data)
        # Predicted values from the tree.
        fitness_function = Fitness.get_fitness_metric(self.fitness_metric)
        fitness = fitness_function(Y_Data, Y_Pred, Weights)
        self.fitness = Fitness.get_fitness_sign(self.fitness_metric) * fitness
        
        return self.fitness

#####################################################################################################
''' The Definition of the functions that are related to the population of the syntax trees.'''
#####################################################################################################

#####################################################################################
#                            Population Initialization                              #
#####################################################################################

def initialize_population(function_set, terminal_set, num_features, min_depth, max_depth, fitness_metric,
                          population_size):
    '''This function initializes the population of the trees. This needs to be changed according to the covering mechanism.
    Currently, it takes a parameter 'population_size' and returns the population of randomly created trees.'''

    population = []
    for _ in range(population_size):
        tree = GP_Tree(function_set, terminal_set, num_features, min_depth, max_depth, fitness_metric)
        tree.generate_half_and_half()
        population.append(tree)

    return population


#####################################################################################
#           Population Evaluation: Calculating fitness of each individual           #
#####################################################################################

def evaluate_population(population, X_Data, Y_Data, Weights=None):
    # The whole population is evaluated with the input X_Data and the fitness of each individual tree is calculated.
    
    fitness_list = []
    for individual_tree in population:
        fitness_list = fitness_list + [individual_tree.evaluate_fitness(X_Data, Y_Data, Weights)]
    return fitness_list


#####################################################################################
#               Parent selection method for applying genetic operators              #
#####################################################################################

def tournament_selection(population, k, tournament_size):
    '''This function selects the best individual (based on fitness values) among 'tournament_size' randomly chosen
    individual trees, 'k' times. The list returned contains references to the syntax tree objects.

    population: A list of syntax trees to select from.
    k: The number of individuals to select.
    tournament_size: The number of individual trees participating in each tournament.
    returns: A list of selected individual trees.
    '''
    selected = []
    for _ in range(k):
        selected_aspirants = [random.choice(population) for i in range(tournament_size)]
        selected.append(max(selected_aspirants, key=attrgetter("fitness")))
    return selected


#####################################################################################
#                            GP Cross-over: One-point crossover                     #
#####################################################################################
'''
Copied from GP_tree.py extracs: starting here
'''
def crossover_onepoint(first_parent, second_parent):
    """ This method performs the one point crossover in the tree. Randomly select in each individual tree and exchange
    each subtree with the point as root between each individual."""

    # These lists will contain all the nodes that are present in the trees.
    subtree1_nodes = all_nodes(first_parent)
    subtree2_nodes = all_nodes(second_parent)

    if len(subtree1_nodes) > 1 and len(subtree2_nodes) > 1:
        # Select the random node index to make a slice.
        slice_index1 = random.randint(1, len(subtree1_nodes) - 1)
        slice_index2 = random.randint(1, len(subtree2_nodes) - 1)

        # Select nodes using the slice_index.
        slice_node1 = subtree1_nodes[slice_index1]
        slice_node2 = subtree2_nodes[slice_index2]

        # "ancestor_node1" and "ancestor_node2" are parent nodes of the sliced node. These will be needed while
        # applying actual crossover.
        ancestor_node1 = slice_node1.parent
        ancestor_node2 = slice_node2.parent

        # Finding the selected node to slice in its parent node
        for i in range(len(ancestor_node1.children)):
            if ancestor_node1.children[i] == slice_node1:
                ancestor_node1.children[i] = slice_node2  # Putting the subtree selected from the other tree.
                slice_node2.parent = ancestor_node1  # Making the parent reference
                break  # Can come out of the loop from here.

        # Same thing happening for the other tree.
        for i in range(len(ancestor_node2.children)):
            if ancestor_node2.children[i] == slice_node2:
                ancestor_node2.children[i] = slice_node1
                slice_node1.parent = ancestor_node2
                break

    return first_parent, second_parent

def all_nodes(tree):
    """This function returns a list containing all the nodes present in the tree. This uses the func:
    "all_nodes_helper" for making the recursive calls."""
    all_nodes = []
    all_nodes_helper(tree.root, all_nodes)
    return all_nodes


def all_nodes_helper(node, all_nodes):
    """Helper function for creating the list containing all the nodes. Used by the func: "all_nodes" for making the recursive
    calls"""

    all_nodes.append(node)  # Appending the node in the list.

    if node.children is None:
        return  # Base-case.

    for child in node.children:
        all_nodes_helper(child, all_nodes)  # Recursive calls for the other nodes.

'''
Copied from GP_tree.py extracs: Ending here
'''

#####################################################################################
#                                 GP Mutation                                       #
#####################################################################################

def mutation_NodeReplacement(parent):
    '''Replaces a randomly chosen node from the individual tree by a randomly chosen node with the same number
    of arguments from the attribute: "arity" of the individual node. It takes the input "population" and selects
    the parent to mutate.'''

    # Root of the parent.
    root_parent = parent.root

    # List to store all the nodes in the parent chosen. This list is populated using the function "mutation_helper".
    all_nodes_ = []

    # Populating the list "all_nodes".
    mutation_helper(root_parent, all_nodes_)

    mutation_point = random.choice(all_nodes_)  # Choosing the mutation point

    if (mutation_point.children == None):  # Case 1: Mutation point is a terminal.
        new_terminal_data = random.choice(parent.terminal_set)
        mutation_point.data = new_terminal_data

    else:  # Case 2: Mutation point is a function
        mutation_point_arity = Functions.get_arity(mutation_point.data)

        while True:  # Finding the same arity function.
            new_function_data = random.choice(parent.function_set)
            if (Functions.get_arity(new_function_data) == mutation_point_arity):
                mutation_point.data = new_function_data
                break

    offspring = parent
    return offspring


def mutation_Uniform(parent):
    '''Randomly select a mutation point in the individual tree, then replace the subtree at that point
    as a root by the "random_subtree_root" that was generated using one of the initialization methods.'''

    # Root of the parent.
    root_parent = parent.root

    # List to store all the nodes in the parent chosen. This list is populated using the function "mutation_helper".
    all_nodes_ = []

    # Populating the list "all_nodes".
    mutation_helper(root_parent, all_nodes_)

    mutation_point = random.choice(all_nodes_)  # Choosing the mutation point randomly.
    ancestor_mutation_point = mutation_point.parent  # Saving the parent node of the mutation point

    if (ancestor_mutation_point == None):  
        # If root is chosen as the Mutation-point, perform mutation through
        # Node-Replacement method. This has been suggested by Ryan.
        offspring = mutation_NodeReplacement(parent)
        return (offspring,1)

    # Performing the uniform mutation.
    for i in range(len(ancestor_mutation_point.children)):
        if (ancestor_mutation_point.children[i] == mutation_point):

            # Generating a random subtree to attach to the mutation point.
            mut_subtree = GP_Tree(function_set, terminal_set, num_features=1, min_depth=1, max_depth=2,
                                  fitness_metric="Root_Mean_Square_Error")
            mut_subtree.generate_half_and_half()
            random_subtree_root = mut_subtree.root

            # Fitting the random subtree in the parent
            ancestor_mutation_point.children[i] = random_subtree_root
            del (mutation_point)
            random_subtree_root.parent = ancestor_mutation_point
            break

    offspring = parent

    return (offspring,0)


def mutation_helper(node, all_nodes_):
    ''' Helper function to store all the nodes in a tree and return the list of the stored nodes. Used by mutation
    methods.'''

    if (node.children == None):
        all_nodes_.append(node)
        return

    all_nodes_.append(node)

    for child in node.children:
        mutation_helper(child, all_nodes_)



if __name__ == "__main__":

    function_set = ("add", "mul", "sub", "div", "cos", "sqrt", "absolute", "sin", "tan")
    terminal_set = (5.0, 8.5, 2.0, 0.0)
    gen = 1
    population_size = 100
    ratio_mutation = 1.0
    ratio_crossover = 0.0
    tournament_size = 5

    population = initialize_population(function_set, terminal_set, num_features=13, min_depth=1, max_depth=3,
                                       fitness_metric="Root_Mean_Square_Error", population_size=population_size)

    
    # Writing original population into a file
    # with open('population_init.txt', '+a') as pop_init_f:
    #     pop_init_f.write('**********************\n')
    #     pop_init_f.write(str(len(population)))
    #     for l in range(0,len(population)):
    #         pop_init_f.write(str(population[l])+'\n')
    # pop_init_f.close()

    # Open files
    Y_Data = np.loadtxt(open("Regression(2X+5)_Test_Y.txt"), delimiter="\t")
    X_Data = np.loadtxt(open("Regression(2X+5)_Test_X.txt"), delimiter="\t")

    # Evaluating the whole population
    fitness_list = evaluate_population(population, X_Data, Y_Data)

    # Make a list of fitness to save in a file later
    fitness_file = [fitness_list]


    # Iterate over generations
    for g in range(0,gen):

        population_old = population
        population = []
        population_picked_mutation = []
        root_mut = []
        population_origin = []

        # To make sure that new population always contains no more than given size
        while len(population) < population_size:

            # Select an offspring
            selected_parent = tournament_selection(population_old, k=1, tournament_size=tournament_size)

            # Random value to decide fate of the offspring
            random_value = random.random()
            random_value = 0.6

            # Population obtained by crossover
            if random_value <= ratio_crossover:

                # To obtain donor parents
                donor_parent = tournament_selection(population_old, k = 1, tournament_size = tournament_size)

                # Carryout crossover
                offsprings_crossover = crossover_onepoint(selected_parent[0],donor_parent[0])

                # Add to the new population list
                population.append(offsprings_crossover[0])

                # To annotate the origin of the individual for population.txt file.
                population_origin.append('crossover')

            # Population obtained by mutation
            elif random_value > ratio_crossover and random_value <= (ratio_crossover + ratio_mutation):

                # To write in a file population.txt
                population_picked_mutation.append(selected_parent[0])

                # Carryout mutation
                offspring_mutation = mutation_Uniform(copy.deepcopy(selected_parent[0]))

                # Add the offspring to the population
                population.append(offspring_mutation[0])

                # Write each mutation generation in a file
                with open('population_mut_gen.txt', '+a') as pop_mut_f:
                    pop_mut_f.write('**********************\n')
                    if(offspring_mutation[1])==0:
                        pop_mut_f.write('Uniform\n')
                    else:
                        pop_mut_f.write('Node-Replacement\n')
                    pop_mut_f.write('Original \n')
                    pop_mut_f.write(str(selected_parent[0]))
                    pop_mut_f.write('Progeny \n')
                    pop_mut_f.write(str(offspring_mutation[0])+'\n')
                pop_mut_f.close()

                if(offspring_mutation[1])==0:
                    population_origin.append('mutation_Uniform')
                else:
                    population_origin.append('mutation_NodeReplacement')


            # Population obatined by direct selection
            else:
                population = population + selected_offspring
                population_origin.append('selection')

        fitness_list = evaluate_population(population, X_Data, Y_Data)
        fitness_file = fitness_file + fitness_list
        # print('selection effect:'+str(len(population)))
        # print('set:'+str(len(set(population))))

        max_fitness = max(fitness_list)

        # Write the fittest population in a file
        with open('population.txt', '+a') as pop_f:
            pop_f.write(str(g)+'\n')
            for pos, value in enumerate(fitness_list):
                if value == max_fitness:
                    pop_f.write("%s\n" % population_origin[pos])
                    pop_f.write("%s\n" % population[pos])
                    pop_f.write("Original \n")
                    pop_f.write("%s\n" % population_picked_mutation[pos])
                    pop_f.write("root \n")
                    # pop_f.write("%s\n" % root_mut[pos])
        pop_f.close()

        # Write all the fitness values in a file
        with open('fitness_file.txt', '+a') as fit_f:
            for fit in fitness_file:
                fit_f.write("%s\n" % fit)
        fit_f.close()

        # Write all the individual of a generation in a file
        with open('population_each_gen.txt','+a') as pop_gen_f:
            pop_gen_f.write(str(g)+'\n')
            pop_gen_f.write('**********************\n')
            pop_gen_f.write(str(len(population)))
            for l in range(0,len(population)):
                pop_gen_f.write(str(population[l])+'\n')
        pop_gen_f.close()

        with open('population_mut_gen.txt', '+a') as pop_mut_f:
            pop_mut_f.write(str(g)+' gen \n')
        pop_mut_f.close()


   ####################################################################
    ## TESTING AND A PLAYGROUND: DISREGARD FOR FINAL SUBMISSION
    ####################################################################

    # Parent Selection for mutation.
    # selected_offspring_temp = tournament_selection(population, k=1, tournament_size=5)
    # print("Selected Parent:")
    # print(selected_offspring_temp[0])
    # # Perform mutation
    # child=mutation_Uniform(selected_offspring_temp[0])
    # print(child[0])