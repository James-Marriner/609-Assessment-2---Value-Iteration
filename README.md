# Value Iteration for Markov Decision Processes.

The Value Iteration package allows for implementation of asynchronous value iteration algorithm for Markov Decision Processes (MDPs). The package allows for different termination conditions for the algorithm allowing user flexibility in usage.

## Installation

### To install from github with pip

```python 
python -m pip install 'git+https://github.com/James-Marriner/609-Assessment-2---Value-Iteration.git'
```

### Unit tests:
The associated unit tests for the Value Iteration are not installed alongside the package by default. To verify their results please do the following.
```text
git clone https://github.com/James-Marriner/609-Assessment-2---Value-Iteration.git
```
Move into the cloned repository which by default is done by
```text
cd 609-Assessment-2---Value-Iteration
```
 Then, if needed install the pytest package and run the tests by
```python
pip install pytest
pytest ValueIteration/Tests
```

## Examples
Two examples are provided including Example 9.27 and a simplified version of Example 9.28 from [1](#ref1).

## Value Iteration Implementation.
The implementation of Value Iteration is asynchronous in updating the value function and deals with maximising rewards. Hence, minimisation problems will need to be reformatted for solving.

```text
Inputs: 
	S, a set of states,
	A, the set of available actions for each state,
	P(s'|a,s), the probability of entering state s' from state s when taking action a
	R(s',a,s), the reward associated with entering state s' from state s by action a.
	Gamma, the discount factor in (0,1) for the MDP
	Termination Function, a parameterised function that determines the termination 	condition.

Outputs: 
	π[s], the approximate optimal policy for each state,
	V[s], the current value function for each state


Local: Assign V[s] = 0 for each state
repeat:
	For each state s, compute
	Q[s,a] = sum over all s' in S: P(s' | s,a)[R(s,a,s') + Gamma V(s')]
	V[s] = max over all actions a in A: Q[s,a]
Until Termination Function.
For each state s do:
	Q[s,a] = sum over all s' in S: P(s' | s,a)[R(s,a,s') + Gamma V(s')]
	π[s] = argmax over all actions a in A: Q[s,a]
Return π, V.
```

With the exceptions of specifiying the discount factor and termination condition, this implementation follows exactly that of the pseudo-code seen in Chapter 9.5.2 of [1](#ref1).


## Limitations and Further Work
Currently as seen in the pseudo code, the algorithm stores the updating value for all possible actions. For a large MDP this could be infeasible, in which case a check could be performed to only record the action if it improves upon a previously checked action.

Additionally, the current format required to input the transition probabilities and rewards is cumbersome and may put off potential package users so it is worth exploring alternative methods which improve user input experience.

Finally, whilst the termination conditions off flexibility, users may prefer that they are incorporated into the algorithm rather than being externally specified. In their current form, the termination conditions are only checked after a full sweep of updating the value function. This could mean that in the case of a large MDP, the time termination occurs far after the specified time if many states are being updated.

## Creator
James Marriner - j.marriner@lancaster.ac.uk. Author, Maintainer and Creator.

## Version
This package was created using Python 3.13.10 and package Pymonad 2.4.0.

## References
[1] <a id="ref1"></a>
[Poole, D. L., & Mackworth, A. K. (2017). *Artificial Intelligence: Foundations of Computational Agents* (2nd ed.)](https://artint.info/2e/html2e/ArtInt2e.html)
