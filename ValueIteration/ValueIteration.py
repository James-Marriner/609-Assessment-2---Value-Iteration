import time
from pymonad.tools import curry
from .Utilities.Dot_Product import Dot_Product

def Expected_Reward(Probabilities:dict, 
                    Conditional_Rewards:dict)-> dict:
    # Computes the expected reward by applying the dot product to two lists
    Ex_Reward = {k : Dot_Product(Conditional_Rewards[k], Probabilities[k]) for k in Probabilities}
    return Ex_Reward

@curry(7)
def Update(Actions:dict,
           Probabilities:dict,
           Rewards:dict,
           Discount:float,
           State:str,
           ValueFunction:callable,
           Policy:bool):

    # Uses Bellman's Equation to calculate the new value function.
    Q_s_a = list(map(lambda A : Rewards[State,A] + Discount* Dot_Product(ValueFunction,Probabilities[State,A]), Actions[State]))
    Best = max(Q_s_a)
    if(Policy):
        # Used to find the policy instead of the value function
        return(Actions[State][Q_s_a.index(Best)])
    else:
        return(Best)


@curry(3)
def Convergence(epsilon:float,
                V_new:list[any],
                V_old:list[any])->bool:
    n = len(V_new)
    
    # Computes the total absolute difference in value functions
    Change = sum(abs(V_new[i] - V_old[i]) for i in range(n))

    # Computes average absolute difference vs epsilon
    if Change/n <= epsilon:
        return(False)
    else:
        return(True)

def TimeTerminate(Time_limit:float) -> callable:
    Start_time = time.time()
    
    def Term_function(*args)->bool:
        # Checks if time limit has been reached since function instance was defined.
        return (time.time() - Start_time) <= Time_limit
    
    return Term_function


def Async_VI(States:list[any],
             Actions:dict,
             Probabilities:dict,
             Rewards:dict,
             Discount:float,
             TerminationFunction:callable) -> list[dict,dict]:
    '''
    States - List of States,

    Actions - Dictionary mapping each state to available actions in that state,

    Probabilities - Dictionary mapping state action pair (s,a) to values P(s'|s,a)

    Rewards - Dictionary mapping (s,a) to list of rewards R(s',s,a).
    Reward values must be defined for all states including unreachable s'  

    Discount - Float value in (0,1), controls the effect of future rewards.

    TerminationFunction - Various functions are possible and are parameterised in their own way. 
    Termination Functions used here expect to take in arguments v_old and v_new whilst returning Boolean
    '''
    # Calculate Expected Rewards
    ExRewards = Expected_Reward(Probabilities,Rewards)

    # Parameterise update step
    Update_step = Update(Actions,Probabilities,ExRewards,Discount)

    # Specify non-terminal states to search over.
    NonTerminal = [S for S in Actions if Actions[S]]
    
    # Initialise Value function
    n = len(States)
    V_new = [0 for i in range(n)]
    Continue = True
    
    while(Continue):
        # Retain previous value function for epsilon convergence.
        V_old = V_new.copy()

        # Update value function for non-terminal states
        for s in NonTerminal:
            V_new[States.index(s)] = Update_step(s,V_new,False)

        # Check termination condition
        Continue = TerminationFunction(V_new,V_old)

    # Compute optimal policy and outputs
    Opt_policy = {s : "None" for s in States}
    Final_V = {States[i] : V_new[i] for i in range(n)}
    for s in NonTerminal: Opt_policy[s] = Update_step(s,V_new, True) 
    
    return(Opt_policy, Final_V)