import math
import time
from pymonad.tools import curry

def Dot_Product(v1,v2):
    return(sum(map(math.prod, zip(v1,v2))))

def Expected_Reward(Probabilities, Conditional_Rewards):
    Ex_Reward = {k : Dot_Product(Conditional_Rewards[k], Probabilities[k]) for k in Probabilities}
    return Ex_Reward

@curry(7)
def Update(Actions,Probabilities, Rewards, Discount, State, ValueFunction,Policy):
    Q_s_a = list(map(lambda A : Rewards[State,A] + Discount* Dot_Product(ValueFunction,Probabilities[State,A]), Actions[State]))
    Best = max(Q_s_a)
    if(Policy):
        return(Actions[State][Q_s_a.index(Best)])
    else:
        return(Best)


@curry(3)
def Convergence(epsilon, V_new,V_old):
    n = len(V_new)
    Change = sum(abs(V_new[i] - V_old[i]) for i in range(n))
    if Change/n <= epsilon:
        return(False)
    else:
        return(True)

def TimeTerminate(Time_limit):
    Start_time = time.time()
    
    @curry(2)
    def inner(v_new, v_old):
        return (time.time() - Start_time) <= Time_limit
    
    return inner


def Async_VI(States,Actions,Probabilities,Rewards,Discount,TerminationFunction):
    # States - List,

    # Actions - List,

    # Probabilities - Dictionary with key (S,A) and Values P(S'|S,A)

    # Rewards - Dictionary with key (S,A) and Value R(S'|S,A)

    # Discount - In (0,1), controls the effect of future rewards.

    # Termination Function - Takes in Old and Current Value Functions, returns Boolean.

    ExRewards = Expected_Reward(Probabilities,Rewards)
    
    Update_step = Update(Actions,Probabilities,ExRewards,Discount)
    
    NonTerminal = [S for S in Actions if Actions[S]]


    n = len(States)
    V_new = [0 for i in range(n)]
    Continue = True
    while(Continue):
        V_old = V_new.copy()
        # V_new = [Update_step(s,V_new, False) for s in NonTerminal]
        for s in NonTerminal:
            V_new[States.index(s)] = Update_step(s,V_new,False)
        Continue = TerminationFunction(V_new,V_old)

    Opt_policy = {s : "None" for s in States}
    Final_V = {States[i] : V_new[i] for i in range(n)}
    for s in NonTerminal: Opt_policy[s] = Update_step(s,V_new, True) 
    return(Opt_policy, Final_V)