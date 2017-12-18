#    example which maximizes the sum of a list of integers

#    each of which can be 0 or 1

import  ann
import  pandas as pd
import  numpy as np
import random
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier#import the classifier
from deap import base

from deap import creator

from deap import tools

pandas_data=pd.read_csv('sql_eigen.csv')
sql_eigen=pandas_data.fillna(np.mean(pandas_data))

data =sql_eigen.iloc[:,0:85]
# data.iloc[:,84][data.iloc[:,84]>200]=91
data['age'][data['age']>200]=91
data2=data.drop(['hr_cov', 'bpsys_cov', 'bpdia_cov', 'bpmean_cov', 'pulse_cov', 'resp_cov', 'spo2_cov'],axis=1)

label=sql_eigen['class_label']

dataMat1=np.array(data2)
labelMat=np.array(label)

data01 = ann.preprocess(dataMat1)
dataMat = ann.preprocess1(data01)
dataMat=np.array(dataMat)


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator

#                      define 'attr_bool' to be an attribute ('gene')

#                      which corresponds to integers sampled uniformly

#                      from the range [0,1] (i.e. 0 or 1 with equal

#                      probability)

toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers

#                         define 'individual' to be an individual

#                         consisting of 100 'attr_bool' elements ('genes')

toolbox.register("individual", tools.initRepeat, creator.Individual,

                 toolbox.attr_bool, 78)

# define the population to be a list of individuals

toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# the goal ('fitness') function to be maximized
######calculate the fitness value#######

individual=toolbox.individual()
individual = np.array(individual)
index = np.where(individual == 1)
for i in index:
    data_in = dataMat[:, i]
datain = np.array(data_in)

neronum=len(index)
clf=MLPClassifier(hidden_layer_sizes=(neronum,), activation='tanh',
                      shuffle=True,solver='sgd',alpha=1e-6,batch_size=1,
                      learning_rate='adaptive')

skf = StratifiedShuffleSplit(n_splits=5)
dataMat=datain
scores=[]
for train, test in skf.split(dataMat, labelMat):
    print("%s %s" % (train, test))
    train_in = dataMat[train]
    test_in = dataMat[test]
    train_out = labelMat[train]
    test_out = labelMat[test]
    clf.fit(train_in, train_out)
    predict_prob = clf.predict_proba(test_in)
    test=np.sum((predict_prob[:,1]-test_out)**2)
    score=clf.score(test_in,test_out)
    scores.append(score)
fitnessvalue=np.mean(scores)
#######calculate the fitness value#########
def evalOneMax(individual):
    individual = np.array(individual)
    index = np.where(individual == 1)

    for i in index:
        data_in = dataMat[:, i]
    datain = np.array(data_in)

    return sum(individual),


# ----------

# Operator registration

# ----------

# register the goal / fitness function

toolbox.register("evaluate", evalOneMax)

# register the crossover operator

toolbox.register("mate", tools.cxOnePoint)

# register a mutation operator with a probability to

# flip each attribute/gene of 0.05

toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# operator for selecting individuals for breeding the next

# generation: each individual of the current generation

# is replaced by the 'fittest' (best) of three individuals

# drawn randomly from the current generation.

toolbox.register("select", tools.selRoulette)


# ----------



def main():
    random.seed(64)

    # create an initial population of 300 individuals (where

    # each individual is a list of integers)

    pop = toolbox.population(n=300)

    # CXPB  is the probability with which two individuals

    #       are crossed

    #

    # MUTPB is the probability for mutating an individual

    CXPB, MUTPB = 0.5, 0.2

    print("Start of evolution")

    # Evaluate the entire population

    fitnesses = list(map(toolbox.evaluate, pop))

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of

    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations

    g = 0

    # Begin the evolution

    while max(fits) < 100 and g < 1000:

        # A new generation

        g = g + 1

        print("-- Generation %i --" % g)

        # Select the next generation individuals

        offspring = toolbox.select(pop, len(pop))

        # Clone the selected individuals

        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring

        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB

            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children

                # must be recalculated later

                del child1.fitness.values

                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB

            if random.random() < MUTPB:
                toolbox.mutate(mutant)

                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        fitnesses = map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring

        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats

        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)

        mean = sum(fits) / length

        sum2 = sum(x * x for x in fits)

        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fits))

        print("  Max %s" % max(fits))

        print("  Avg %s" % mean)

        print("  Std %s" % std)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]

    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))


if __name__ == "__main__":
    main()