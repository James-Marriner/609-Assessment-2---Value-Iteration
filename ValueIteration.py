import math
from pymonad.tools import curry

@curry(7)
def Update(Actions,Probabilities, Rewards, Discount, State, ValueFunction,Policy):
    Q_s_a = list(map(lambda A : Rewards[State,A] + Discount* sum(map(math.prod, zip(ValueFunction,Probabilities[State,A]))), Actions))
    Best = max(Q_s_a)
    if(Policy):
        return(Actions[Q_s_a.index(Best)])
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


def Async_VI(States,Actions,Probabilities,Rewards,Discount,TerminationFunction):
    # States - List,

    # Actions - List,

    # Probabilities - Dictionary with key (S,A) and Values P(S'|S,A)

    # Rewards - Dictionary with key (S,A) and Value R(S,A)
    # Unclear what happens if we need to compute the expected reward.

    # Discount - In (0,1), controls the effect of future rewards.

    # Termination Function - Takes in Old and Current Value Functions, returns Boolean.

    Update_step = Update(Actions,Probabilities,Rewards,Discount)

    n = len(States)
    V_new = [0 for i in range(n)]
    Continue = True
    while(Continue):
        V_old = V_new.copy()
        V_new = [Update_step(s,V_new, False) for s in States]
        Continue = TerminationFunction(V_new,V_old)

    Opt_policy = [Update_step(s,V_new, True) for s in States]
    return(Opt_policy, V_new)