
import math
import numpy as np
import copy 

def parseStateActionState(state_weight_values, observation_actions):
    original = copy.deepcopy(list(state_weight_values.keys()))

    actions_values = []
    for val in observation_actions:
        if val[1] not in actions_values and val[1] != "_":
            actions_values.append(val[1])
 
    state_action_state_list_combinations = {}

    valRefDict = {}
    for index, val in enumerate(original):
        valRefDict[val] = index

    actRefDict = {}
    for index, val in enumerate(actions_values):
        actRefDict[val] = index

    with open('speechRecognition/state_action_state_weights.txt', 'r') as f:
        lineNumber = 0
        default_weights = -1
        for line in f:
            if lineNumber == 0:
                fileName = line
            elif lineNumber == 1:
                lineOne = line.split()
                triples = lineOne[0]
                unique_states = lineOne[1]
                unique_actions = lineOne[2]
                default_weights = lineOne[3]

                # Add default weights to the dictionary
                for val in original:
                    for act in actions_values:
                        for val2 in original:
                            state_action_state_list_combinations[(val,act,val2)] = int(default_weights)
            else:
                action_state_values = line.split()
                state = action_state_values[0].replace('"','')
                action = action_state_values[1].replace('"','')
                nextState = action_state_values[2].replace('"','')
                weight = int(action_state_values[3])

                #  Imp!! Add actions not in the observation
                if action not in actions_values:
                    actions_values.append(action)
                    actRefDict[action] = len(actRefDict)

                state_action_state_list_combinations[(state,action,nextState)] = weight
                
                
            lineNumber += 1

    sas_list = [[[0 for _ in range(len(original))] for _ in range(len(actions_values))] for _ in range(len(original))]
    
    for val in original:
        for act in actions_values:
            for val2 in original:
                sas_list[valRefDict[val]][actRefDict[act]][valRefDict[val2]] = state_action_state_list_combinations[(val, act, val2)]
   
    return sas_list, default_weights, actions_values, actRefDict

def parseStateObservations(state_weight_values, observation_actions):

    original = copy.deepcopy(list(state_weight_values.keys()))

    observation_values = []
    for val in observation_actions:
        if val[0] not in observation_values:
            observation_values.append(val[0])

    valRefDict = {}
    for index, val in enumerate(original):
        valRefDict[val] = index

    obsRefDict = {}
    for index, val in enumerate(observation_values):
        obsRefDict[val] = index

    state_observation_list = {}
    with open('speechRecognition/state_observation_weights.txt', 'r') as f:
        lineNumber = 0
        counter = 0 
        for line in f:
            counter += 1
            if lineNumber == 0:
                fileName = line
            elif lineNumber == 1:
                lineOne = line.split()
                numberOfPairs = lineOne[0]
                uniqueStates = lineOne[1]
                uniqueObservations = lineOne[2]
                defaultWeight = lineOne[3]

                for val in original:
                    for obs in observation_values:
                        state_observation_list[(val, obs)] = int(defaultWeight)
            else:
                state_observation_values = line.split()
                
                state = state_observation_values[0].replace('"','')
                observation = state_observation_values[1].replace('"','')
                weight = int(state_observation_values[2])

                if observation not in list(obsRefDict.keys()):            
                    obsRefDict[observation] = len(obsRefDict)
                
                state_observation_list[(state, observation)] = weight

            lineNumber += 1
    

    return state_observation_list, defaultWeight, obsRefDict

def parseObservationActions():
    observation_actions_list = []
    numOfObservations = 0
    currentObservation = 0
    with open('speechRecognition/observation_actions.txt', 'r') as f:
        lineNumber = 0

        for line in f:
            if lineNumber == 0:
                fileName = line
            elif lineNumber == 1:
                numOfObservations = line
            elif currentObservation < int(numOfObservations) - 1:
                currentObservation += 1
                observation_action_values = line.split()
                observation = observation_action_values[0].replace('"','')
                action = observation_action_values[1].replace('"','')

                observation_actions_list.append((observation,action))
            else:
                last_observation = line.replace('"','')
                last_observation = last_observation.replace("\n","")
                observation_actions_list.append((last_observation,"_"))
           

            lineNumber += 1
    return observation_actions_list

def parseStateWeights():
    state_weight_values = {}
    with open('speechRecognition/state_weights.txt', 'r') as f:
        lineNumber = 0

        for line in f:
            if lineNumber == 0:
                numberOfWeights = line
            elif lineNumber == 1:
                defaultWeight = line
            else:
                state_weights = line.split()
                state_weight_values[state_weights[0].replace('"','')] = int(state_weights[1])
            lineNumber += 1        
    
    return state_weight_values


def writeToOutputFile(finalSteps):
    with open('speechRecognition/states.txt', 'w') as f:
        f.write("states\n")
        f.write(str(len(finalSteps)) + "\n")
        for i in range(len(finalSteps)):
            f.write('"' + finalSteps[i] + '"' + "\n")

        f.close()

def calculate_transition_probabilities(state_action_state_weights, state_weights, observation_actions, actRefDict):

    original = copy.deepcopy(list(state_weights.keys()))
    numpyTransitions = np.array(state_action_state_weights)
    sumTrans = np.sum(numpyTransitions, axis=(2))
    trans_probs = numpyTransitions/sumTrans[:,:,np.newaxis]
    
    return trans_probs


def calculate_emission_matrix(state_observation_weights, state_weights, obsRefDict):

    stateList = [s for s in state_weights]
    num_states = len(state_weights)
    num_observations = len(obsRefDict)

    emission_matrix = np.zeros((num_states, num_observations))
   

    for entry in state_observation_weights:
        state, obs = entry
        emission_matrix[stateList.index(state), obsRefDict[obs]] = state_observation_weights[(state, obs)]
   
    # Normalize the emission matrix
    for i in range(num_states):
        total_weight = sum(emission_matrix[i])
        if total_weight != 0:
            emission_matrix[i] /= total_weight

    return emission_matrix


def calculate_state_weight_probabilities(state_weights):
    sum = 0
    for val in state_weights:
        sum += state_weights[val]
    
    for val in state_weights:
        state_weights[val]/=sum

    return state_weights
    

def viterbiAlgorithm(observations_actions_values, state_action_state_values, state_observation_values, state_weight_values, transition_matrix, emission_matrix, actRefDict):    
    lenOfObservationSequence = len(observations_actions_values)
    
    observation_space = []

    for obs, val in state_observation_values:
        observation_space.append(val)

    states = list(state_weight_values.keys())    

    # Convert the observation sequence to have the index of the observation in the observation space for calculations only
    observation_seq_reference = []
    for obs, action in observations_actions_values:
        if action not in actRefDict:
            actRefDict[action] = len(actRefDict)
        observation_seq_reference.append((observation_space.index(obs), actRefDict[action]))

    stored_values = [[0] * len(observation_seq_reference) for _ in range(len(states))]
    backProp = [[0] * len(observation_seq_reference) for _ in range(len(states))]

    for s in range(len(states)):
        stored_values[s][0] = state_weight_values[states[s]] * emission_matrix[s][observation_seq_reference[0][0]]

    for o in range(1, len(observation_seq_reference)):
        for s in range(len(states)):
            maxProb = -math.inf
            maxK = 0

            for k in range(len(states)):
                if observation_seq_reference[o-1][1] != len(actRefDict): # Added "_" to the last index
                    prob = stored_values[k][o-1] * transition_matrix[k][observation_seq_reference[o-1][1]][s] * emission_matrix[s][observation_seq_reference[o][0]]     
                else:
                    prob = stored_values[k][o-1] * emission_matrix[s][observation_seq_reference[o][0]]
                    
                if prob > maxProb:
                    maxProb = prob
                    maxK = k
            
            stored_values[s][o] = maxProb
            backProp[s][o] = maxK

    selectPath = []
    maxFinalProb = max(stored_values[k][len(observation_seq_reference)-1] for k in range(len(states)))

    maxIndex = max(range(len(states)), key=lambda k: stored_values[k][len(observation_seq_reference)-1])

    for o in range(len(observation_seq_reference)-1, -1, -1):
        selectPath.insert(0, states[maxIndex])
        maxIndex = backProp[maxIndex][o]

    return selectPath

    

def temporalReasoning():

    state_weight_values = parseStateWeights()

    observation_actions_values = parseObservationActions()

    state_action_state_values, default_weights_sasv, actions_values, actRefDict = parseStateActionState(state_weight_values, observation_actions_values) 
    
    state_observation_values, default_weights_sov, obsRefDict = parseStateObservations(state_weight_values, observation_actions_values)
    
    transition_matrix = calculate_transition_probabilities(state_action_state_values, state_weight_values, observation_actions_values, actRefDict)
    
    emission_matrix = calculate_emission_matrix(state_observation_values, state_weight_values, obsRefDict)
    
    state_weight_values = calculate_state_weight_probabilities(state_weight_values)
    
    finalSteps = viterbiAlgorithm(observation_actions_values, state_action_state_values, state_observation_values, state_weight_values, transition_matrix, emission_matrix, actRefDict)
    

    writeToOutputFile(finalSteps)

if __name__ == "__main__":
    temporalReasoning()