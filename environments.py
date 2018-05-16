import copy as cp
import random
import numpy as np
import scipy.stats
import itertools
import operator
import functools
import math
random.seed(1234)
np.random.seed(1234)
#import gym



class State():
    def next_state(self,action):
        raise NotImplementedError()

    def reward(self,action,next_state):
        raise NotImplementedError()

class Problem(State):
    name = 0
    number_of_visits=1
    actions={}
    untried_actions={}
    
    def define_actions(self, actions):
        self.actions = actions
        self.untried_actions = cp.deepcopy(actions)

    def next_state(self,action):
        i = states.index(self)
        if action == 'U':
            if self == states[6] or self == states[7] or self == states[8]:
                next_state=self
            elif self == states[0] or self == states[3]:
                next_state = np.random.choice((states[i + 3], states[i + 1], states[i]), p=(0.6, 0.2, 0.2))
            elif self == states[2] or self == states[5]:
                next_state = np.random.choice((states[i + 3], states[i - 1], states[i]), p=(0.6, 0.2, 0.2))
            else:
                next_state = np.random.choice((states[i + 3], states[i - 1], states[i+1]), p=(0.6, 0.2, 0.2))
        elif action=='R':
            if self == states[2] or self == states[5] or self == states[8]:
                next_state=self
            elif self == states[6] or self == states[7]:
                next_state = np.random.choice((states[i-3], states[i + 1], states[i]), p=(0.2, 0.6, 0.2))
            elif self == states[0] or self == states[1]:
                next_state = np.random.choice((states[i+3], states[i + 1], states[i]), p=(0.2, 0.6, 0.2))
            else:
                next_state = np.random.choice((states[i + 3], states[i + 1], states[i-3]), p=(0.2, 0.6, 0.2))
        elif action =='L':
            if self == states[6] or self == states[3] or self == states[0]:
                next_state = self
            elif self == states[7] or self == states[8]:
                next_state = np.random.choice((states[i - 3], states[i], states[i - 1]), p=(0.2, 0.2, 0.6))
            elif self == states[1] or self == states[2]:
                next_state = np.random.choice((states[i + 3], states[i], states[i - 1]), p=(0.2, 0.2, 0.6))
            else:
                next_state = np.random.choice((states[i + 3], states[i - 3], states[i - 1]), p=(0.2, 0.2, 0.6))
        elif action =='D':
            if self == states[0] or self == states[1] or self == states[2]:
                next_state = self
            elif self == states[3] or self == states[6]:
                next_state = np.random.choice((states[i - 3], states[i + 1], states[i]), p=(0.6, 0.2, 0.2))
            elif self == states[5] or self == states[8]:
                next_state = np.random.choice((states[i - 3], states[i - 1], states[i]), p=(0.6, 0.2, 0.2))
            else:
                next_state = np.random.choice((states[i - 3], states[i - 1], states[i + 1]), p=(0.6, 0.2, 0.2))
        return next_state
        
    def reward(self,action,next_state):
        if action == 'D':
            return -0.01
        elif next_state == self:
            return -0.01
        elif next_state == states[8]:
            return 1
        elif action == 'R':
            if self == states[0] or self == states[1]:
                return -0.1
            else:
                return 0.1
        else:
            return 0.1


# create states for a toy problem
#this is how you define an object
# states=[]

# for i in range(9):
#     states.append(Problem())
#     states[-1].name=i

# #sample dictionary
# # Generic actions possible at every state
# actions = {'U':{'nov':0,'Q':0}, 'L':{'nov':0,'Q':0}, 'R':{'nov':0,'Q':0}, 'D':{'nov':0,'Q':0}}

# #set the above actions to all the states. You can customize this depending on the problem
# for i in range(9):
#     states[i].define_actions(actions)



def FrozenlakeEnv():
    return gym.make('FrozenLake-v0').unwrapped

class FrozenlakeState(State):

    def __init__(self,env,obs=None):

        self.env = env
        if obs is None:
            self.state = env.reset()
        else:
            self.state = obs
        self.number_of_visits=1
        self.actions={}
        for i in range(env.action_space.n):
            self.actions.update({i:{'nov':0,'Q':0}})
        self.term = False
        self.policy = 0

    # def define_actions(self, actions):
    #     self.actions = actions
    #     self.untried_actions = cp.deepcopy(actions)

    def next_state_reward(self,action):
        # env.render()
        observation, reward, done, info = self.env.step(action)
        n_s = FrozenlakeState(self.env,observation)
        if done == True:
            n_s.term = 1
        return n_s, reward

    def env_reset(self):
        for x in range(50):
            if self.state==0:
                self.env.reset()
            else:
                action = self.env.action_space.sample()
                o, r, d, i = self.env.step(action)
                if o == self.state:
                    break



def RiverSwimEnv():
    return 0


class RiverSwimState(State):

    def __init__(self,env,obs=0):
        # super(RiverSwimState,self).__init__()
        self.env = env
        self.state = obs
        self.number_of_visits = 1
        self.actions = {'L':{'nov':0,'Q':0},'R':{'nov':0,'Q':0}}
        self.eps    = 0.1
        self.term = False
        self.policy = 0
        self.n_states = 5

    def get_actions(self,state):
        return ['L','R']

    def next_state_reward(self,action):
        if action == 'L':
            next = RiverSwimState(max(0,self.state-1))

        # right
        elif self.state == self.n_states - 1:
            next = RiverSwimState(np.random.choice([self.state-2, self.state - 1], p=[0.4, 0.6]))
        elif self.state > 0:
            next = RiverSwimState(np.random.choice([min(self.n_states - 1, self.state + 1), self.state, max(0, self.state-1)],
                                              p=[0.35, 0.6, 0.05]))
        else:
            next = RiverSwimState(np.random.choice([0, 1], p=[0.4, 0.6]))


        if self.state == 4:
            next.term = 1
            
        #reward calc
        if self.state==0 and action == 'L':
            reward = 0.0001
        elif self.state==self.n_states-1 and action=='R':
            reward = 1.00
        else:
            reward = 0.0
        return next,reward

    def env_reset(self):
        return RiverSwimState(self.env,obs=1)






def SingleTaxiEnv():
    return 0


class SingleTaxiState(State):
    def __init__(self,env, debug=True,
                    gridSideLength=4,
                    maxServiceTime=1,
                    maxCustomersInNbhd=1,
                    nbhdSideLength=3,obs=0):

        # super(SingleTaxiState,self).__init__()
        self.env = env
        self.obs = obs
        self.debug              = debug
        self.gridSideLength     = gridSideLength
        self.maxServiceTime     = maxServiceTime
        self.maxCustomersInNbhd = maxCustomersInNbhd
        self.nbhdSideLength     = nbhdSideLength  #neighborhood, should always be ODD
        self.costProportional   = 0.5
        self.nbhdMatrixX,self.nbhdMatrixY = self.create_neighborhood_matrices()
        self.states = self.generate_states()

        #Transition probability parameters
        np.random.seed(1234)
        self.binomialProb = .5
        self.probOfKOrigins = [scipy.stats.binom.pmf(noOfOrigins,self.maxCustomersInNbhd,self.binomialProb) for noOfOrigins in range(self.maxCustomersInNbhd+1)]
        self.probOriginInCell = np.array(scipy.stats.uniform.rvs(size=self.get_no_of_nbhd_grids()))
        self.probOriginInCell = self.probOriginInCell/sum(self.probOriginInCell)
        self.probOriginInCell.shape = (self.nbhdSideLength,self.nbhdSideLength)
        self.probDestinationInCell = np.array(scipy.stats.uniform.rvs(size=self.get_no_of_grids()))
        self.probDestinationInCell = self.probDestinationInCell/sum(self.probDestinationInCell)
        self.probDestinationInCell.shape = (self.gridSideLength,self.gridSideLength)
        
        self.state = self.states[obs]
        self.number_of_visits = 1
        self.actions = self.get_actions(self.state)
        self.term = False
        self.policy = 0
        self.n_states = len(self.generate_states())

    def get_actions(self,state):

        busyCounter = state[2]
        #Case 0: Taxi is busy
        if busyCounter > 0:
            actionList = {'[(None,None,None,None)]':{'nov':0,'Q':0}} #represents rejection action
            return actionList

        #Case 1: Taxi is not busy
        #Case 1a: No arrivals
        if state[3] == (None,None,None,None):#can compare immutable tuples
            actionList = {'[(None,None,None,None)]':{'nov':0,'Q':0}} #represents rejection action
            # print(actionList)
            return actionList

        #Case 1b: Arrivals exist. Then, get origins from grids that overlap between neighborhood 
        #and the gridworld. Filter out other customers. These are our possible actions represented by ods.
        carPosX = state[0]
        carPosY = state[1]
        actionList = {'[(None,None,None,None)]':{'nov':0,'Q':0}} #Add a default action, that rejects.
        a = self.get_valid_customers(state)
        if a!= []:
            actionList.update({repr(a):{'nov':0,'Q':0}})
        # print(a)
        return actionList


    def trans(self,state,action,statePrime):
        if action[0] == None:#equivalent to checking action==(None,None,None,None)
            if statePrime[0]==state[0] and statePrime[1]==state[1] and statePrime[2]==max(0,state[2]-1):
                return self.get_probability_of_OD(statePrime[3])
            else:
                return 0
        else:
            #print action
            if statePrime[0]==action[2] and statePrime[1]==action[3] and statePrime[2]==self.get_service_time(state,action):
                return self.get_probability_of_OD(statePrime[3])
            else:
                return 0

    def get_state(self,stateIndex):
        return self.states[stateIndex]

    def next_state_reward(self,action):
        action = tuple(eval(action))[0]
        N = self.n_states
        transitionProbability = np.zeros(N)
        for statePrimeIndex in range(N):
            transitionProbability[statePrimeIndex] = \
                self.trans(self.state, action, self.get_state(statePrimeIndex))
        obs = (np.random.choice(range(self.n_states), p=transitionProbability))
        next = SingleTaxiState(self.env,obs)

        # Compute Immediate reward[s,a]
        # rewardVector = np.zeros(N)
        # for stateIndex in range(N):
        #     rewardVector[stateIndex] = self.reward(self.get_state(stateIndex),
        #                                           self.get_action(self.get_state(stateIndex)))

        if action == (None,None,None,None):
            reward = 0 #action is the rejection action either for rejection or when car is busy. So no reward.
        else:
            ox = self.nbhdMatrixX[action[0],action[1]] + self.state[0] #Taking into account relative indexing
            oy = self.nbhdMatrixY[action[0],action[1]] + self.state[1]
            dx = action[2]
            dy = action[3]
            cost_to_go_to_customer = self.get_distance_to_go_to_customer(self.state[0:2],[ox,oy])
            revenue_from_customer = self.get_distance_traveled_with_customer([ox,oy],[dx,dy])
            reward = (revenue_from_customer - self.costProportional*cost_to_go_to_customer)/self.gridSideLength/self.gridSideLength
        return next,reward


    def reward(self,state,action):
        #Currently, there is no cost for lost business when you are busy.
        if action == (None,None,None,None):
            return 0 #action is the rejection action either for rejection or when car is busy. So no reward.
        else:
            ox = self.nbhdMatrixX[action[0],action[1]] + state[0] #Taking into account relative indexing
            oy = self.nbhdMatrixY[action[0],action[1]] + state[1]
            dx = action[2]
            dy = action[3]
            cost_to_go_to_customer = self.get_distance_to_go_to_customer(state[0:2],[ox,oy])
            revenue_from_customer = self.get_distance_traveled_with_customer([ox,oy],[dx,dy]) 
            return (revenue_from_customer - self.costProportional*cost_to_go_to_customer)/self.gridSideLength/self.gridSideLength

    def get_distance_to_go_to_customer(self,cab_position,customer_position):
        return math.sqrt((cab_position[0] - customer_position[0])*(cab_position[0] - customer_position[0]) + \
                (cab_position[1] - customer_position[1])*(cab_position[1] - customer_position[1]))

    def get_distance_traveled_with_customer(self,customer_start,customer_end):
        ox = customer_start[0]
        oy = customer_start[1]
        dx = customer_end[0]
        dy = customer_end[1]
        return math.sqrt((ox-dx)*(ox-dx) + (oy-dy)*(oy-dy))
    
    #The following are helper functions to define states, actions, rewards and transitions


    def get_service_time(self,state,action):
        #should be scaled between 0, 1, and 2
        #should be a deterministic function.
        start = (state[0],state[1])
        origin = (action[0],action[1])
        dest = (action[2],action[3])
        totalDistance = self.get_distance_to_go_to_customer(start,origin) + self.get_distance_traveled_with_customer(origin, dest)
        return min(math.floor(totalDistance/(self.gridSideLength*math.sqrt(2))),2)

    def get_probability_of_OD(self,odTuples):
        #use conditional distibution of demand given origin
        #assume a simple multinomial model for demand that does not depend on origin.
        #P(O'D') = P(O')P(D'|O')
        #P(O') = Pro(k origins)*\Pi_{i=1}^{k}Prob(origin i in cell j | k origins)

        #print "\n od tuple: ",odTuples

        #Base case: Scenario when there are no valid customer od pairs. i.e., noOfOrigins = 0
        if odTuples == (None,None,None,None):
            noOfOrigins = 0
            probNoOfOrigins = self.probOfKOrigins[noOfOrigins]
            return probNoOfOrigins #TODO

        #Scenarios when there are some OD pairs
        noOfOrigins = len(self.get_odpairs_from_odTuples(odTuples))
        probNoOfOrigins = self.probOfKOrigins[noOfOrigins]
        #print "\nprob no of origins = {0}\n".format(probNoOfOrigins)

        probOriginInCell = []
        probDestinationInCell = []
        if noOfOrigins==1:
                probOriginInCell.append(self.probOriginInCell[odTuples[0],odTuples[1]])
                probDestinationInCell.append(self.probDestinationInCell[odTuples[2],odTuples[3]])
        else:
            for k in range(noOfOrigins):
                probOriginInCell.append(self.probOriginInCell[odTuples[k][0],odTuples[k][1]])
                probDestinationInCell.append(self.probDestinationInCell[odTuples[k][2],odTuples[k][3]])
        
        return functools.reduce(operator.mul,probDestinationInCell, 
                functools.reduce(operator.mul,probOriginInCell, probNoOfOrigins))

    def get_no_of_grids(self):        
        return pow(self.gridSideLength,2)

    def get_no_of_nbhd_grids(self):
        return pow(self.nbhdSideLength,2)

    def get_valid_customers(self,state):

        assert state[3] != (None,None,None,None) #cannot pass a state with no OD pairs here.
        validCustomers = []
        #If there are OD pairs
        for odpair in self.get_odpairs_from_odTuples(state[3]):
            xPos = self.nbhdMatrixX[odpair[0],odpair[1]] + state[0]
            yPos = self.nbhdMatrixY[odpair[0],odpair[1]] + state[1]
            # print ("xPos and yPos of potential customer = ({0},{1})".format(xPos,yPos))
            if xPos < 0 or xPos >= self.gridSideLength or yPos < 0 or yPos >= self.gridSideLength:
                # print ("skipping")
                xPos = 1 #TODO: refactor
            else:
                validCustomers.append(odpair)
        return validCustomers

    def get_odpairs_from_odTuples(self,odTuples):
        if len(odTuples)==4: #This hack will not work when self.maxCustomersInNbhd >= 4
            return [odTuples]
        else:
            return list(odTuples)

    def create_neighborhood_matrices(self):
        nNbhdGrids = self.get_no_of_nbhd_grids()
        centerNbhd = ((self.nbhdSideLength-1)/2,(self.nbhdSideLength-1)/2)
        nbhdMatrixX = np.array([i-centerNbhd[0] for i in \
            range(self.nbhdSideLength)]*self.nbhdSideLength).reshape(self.nbhdSideLength,
                                                            self.nbhdSideLength)
        nbhdMatrixY = np.transpose(nbhdMatrixX.copy())
        # if self.debug==True:
        #     pprint.pprint(nbhdMatrixX)
        #     pprint.pprint(nbhdMatrixY)
        return nbhdMatrixX,nbhdMatrixY

    def get_origin_destination_tuples_list(self,originTuple):

        odTuplesList = []
        nGrids = self.get_no_of_grids()
        originMatrix = np.array(originTuple).reshape(self.nbhdSideLength,self.nbhdSideLength)

        if originMatrix.sum()==0:
            #Case 0: If there are no origins, then there are no destinations.
            #this corresponds to the state of no new arrivals.
            odTuplesList = [(None,None,None,None)] #two coordinates for origin and two for destination

        elif originMatrix.sum()==1:#if there is exactly one origin in the originTuple
            ox,oy = np.where(originMatrix==1)
            ox,oy = ox[0],oy[0]
            for dx in range(self.gridSideLength):
                for dy in range(self.gridSideLength):
                    odTuplesList.append((ox,oy,dx,dy))

        elif originMatrix.sum()==2:#if there are exactly two origins in originTuple
            #2 at the same origin location
            oxArray,oyArray = np.where(originMatrix==2)
            if len(oxArray) > 0:
                #2 at the same destination loc
                for dx in range(self.gridSideLength):
                    for dy in range(self.gridSideLength):
                        odTuplesList.append((oxArray[0],oyArray[0],dx,dy))
                #1 and 1 at different destination locs
                destMatrix = np.array([0]*self.get_no_of_grids()).reshape(self.gridSideLength,self.gridSideLength)
                dxArray,dyArray = np.where(destMatrix==0) #All indices
                for i in range(len(dxArray)):
                    for j in range(len(dxArray)):
                        if i != j:
                            odList = []
                            odList.append((oxArray[0],oyArray[0],dxArray[i],dyArray[i]))
                            odList.append((oxArray[0],oyArray[0],dxArray[j],dyArray[j]))
                            odTuplesList.append(tuple(odList))
            else:
                #1 and 1 at different origin locs
                oxArray,oyArray = np.where(originMatrix==1)
                #2 at the same destination loc
                for dx in range(self.gridSideLength):
                    for dy in range(self.gridSideLength):
                        odList = []
                        for m in range(len(oxArray)):
                            odList.append((oxArray[m],oyArray[m],dx,dy))
                        odTuplesList.append(tuple(odList))

                #1 and 1 at different destination loc, the most general case.
                # here each element of the odTuplesList looks like (o1,d1,o2,d2,o1,d2,o2,d1)
                # where o* and d* are (x,y) tuples.
                destMatrix = np.array([0]*self.get_no_of_grids()).reshape(self.gridSideLength,self.gridSideLength)
                dxArray,dyArray = np.where(destMatrix==0) #All indices
                for i in range(len(dxArray)):
                    for j in range(len(dxArray)):
                        if i != j:
                            dIdx = [i,j]
                            for oIdx in itertools.permutations(range(len(oxArray))):
                                odList = []
                                for k in range(len(oxArray)):
                                    odList.append((oxArray[oIdx[k]],oyArray[oIdx[k]],dxArray[dIdx[k]],dyArray[dIdx[k]]))
                                odTuplesList.append(tuple(odList))
        else:
            #print originTuple
            raise NotImplementedError()

        return odTuplesList

    def get_list_of_all_origins(self):
        '''
        we generate all possible origins in any arbitrary neighborhood.
        The output is a list of origin vectors of size nNbhdGrids*1.
        '''

        nNbhdGrids = self.get_no_of_nbhd_grids()
        originList = []

        for nCustomers in range(self.maxCustomersInNbhd+1):

            customerOrigins = []

            if nCustomers==0:
                
                customerOrigins.append([tuple([0]*nNbhdGrids)])

            elif nCustomers==1:

                temp = np.eye(nNbhdGrids)
                for i in range(nNbhdGrids):
                    customerOrigins.append(tuple(temp[i,:]))

            elif nCustomers==2:

                #2 at the same loc
                temp = 2*np.eye(nNbhdGrids)
                for i in range(nNbhdGrids):
                    customerOrigins.append(tuple(temp[i,:]))
                
                #1 and 1 at different locations
                temp = np.eye(nNbhdGrids)
                for i in range(nNbhdGrids):
                    for j in range(i+1,nNbhdGrids):
                        tempVec = np.array([0]*nNbhdGrids)
                        tempVec[j] = 1
                        customerOrigins.append(tuple(temp[i,:]+tempVec))
            else:
                raise NotImplementedError()

            originList.extend(customerOrigins)

        return originList

    def generate_states(self):
        indices = []
        originList = self.get_list_of_all_origins() #NOTE: origins are relative indexed. Need to be recentered.
        odTuplesList = [self.get_origin_destination_tuples_list(originTuple) for originTuple in originList]

        for carPosX in range(self.gridSideLength):
            for carPosY in range(self.gridSideLength):
                # print ("Generating indices for car position ({0},{1})".format(carPosX,carPosY))
                for busyCounter in range(self.maxServiceTime+1):
                        for odTuples in odTuplesList:
                            for od in odTuples:
                                indices.append((carPosX,carPosY,busyCounter,od))
        # print ("Number of state indices is {0}".format(len(indices)))
        #pprint.pprint(indices)
        return indices

    def env_reset(self):
        return SingleTaxiState(self.env)

    def __str__(self):
        N = self.n_states()
        ret = "Probs:\n"
        for stateIndex in range(N):
            for actionIndex in range(self.n_actions(self.get_state(stateIndex))):
                probs = np.zeros(N)
                for statePrimeIndex in range(N):
                    probs[statePrimeIndex] = self.trans(self.get_state(stateIndex),
                                                        self.get_action(self.get_state(stateIndex),actionIndex),
                                                        self.get_state(statePrimeIndex))
                ret += "%s\t %s\t %s\n" % (self.get_state(stateIndex),self.get_action(self.get_state(stateIndex),actionIndex),probs)
        ret += "Rewards:\n"
        for stateIndex in range(N):
            for actionIndex in range(self.n_actions(self.get_state(stateIndex))):
                ret += "%s\t %s\t %s\t %s\n" % (stateIndex,actionIndex,self.get_state(stateIndex), self.reward(self.get_state(stateIndex),
                                                            self.get_action(self.get_state(stateIndex),actionIndex)))
        return ret
#
# if __name__=="__main__":
#
#     debug=True
#     gridSideLength=3
#     maxServiceTime=1
#     maxCustomersInNbhd=1
#     nbhdSideLength=2
#     s = SingleTaxiState(debug, gridSideLength, maxServiceTime,
#         maxCustomersInNbhd, nbhdSideLength)
#     s.next_state()

