import pytest

from ValueIteration import Async_VI, Convergence, TimeTerminate, Expected_Reward, Update

@pytest.mark.parametrize("eps,v1,v2,Expected", [
    (0.01,[0,0], [0,0], False),
    (0.01,[5,5], [1,1], True),
    (1,[1.5],[1], False),
])

def test_Convergence(eps,v1, v2, Expected):
    assert Convergence(eps,v1, v2) == Expected

@pytest.mark.parametrize("States,Actions,Probabilites,Rewards,Discount,Convergence,Expected", [
    (["Healthy","Sick"], 
     {"Healthy" : ["Relax","Party"],
       "Sick" : ["Relax", "Party"]}, 
     {("Healthy","Relax") : [0.95,0.05],
         ("Healthy","Party") : [0.7,0.3],
         ("Sick","Relax") : [0.5,0.5],
         ("Sick", "Party") : [0.1,0.9]},
     {("Healthy","Relax") : [7,7],
           ("Healthy","Party") : [10,10],
           ("Sick","Relax") : [0,0],
           ("Sick", "Party") : [2,2]},
    0.9,
    Convergence(0.01),
    {'Healthy': 'Party', 'Sick': 'Relax'}
    ),
    (["TL","TR","BL","BR"],
     {"TL" : ["D","R"], "TR" : ["L","D"], "BL" : ["U","R"], "BR" : []},
     {("TL","D") : [0,0.1,0.9,0], ("TL","R") : [0,0.9,0.1,0],
          ("TR","L") : [0.9,0,0,0.1], ("TR","D") : [0.2,0,0,0.8],
          ("BL","U") : [0.8,0,0,0.2], ("BL","R") : [0.1,0,0,0.9]},
    {("TL","D") : [0,-1,-2,0], ("TL","R") : [0,-1,-2,0],
          ("TR","L") : [-1.5,0,0,10], ("TR","D") : [-1,0,0,15],
          ("BL","U") : [-0.5,0,0,5], ("BL","R") : [-2.5,0,0,20]},
    0.8,
    Convergence(0.01),
    {'TL': 'D', 'TR': 'D', 'BL': 'R', 'BR': 'None'})
])

def test_Async_VI(States, Actions, Probabilites,Rewards,Discount,Convergence,Expected):
    assert Async_VI(States, Actions, Probabilites,Rewards,Discount,Convergence)[0] == Expected

@pytest.mark.parametrize("Probs,Rewards,Expected", [
    ({("Healthy","Relax") : [0.95,0.05],
         ("Healthy","Party") : [0.7,0.3],
         ("Sick","Relax") : [0.5,0.5],
         ("Sick", "Party") : [0.1,0.9]},
     {("Healthy","Relax") : [7,7],
           ("Healthy","Party") : [10,10],
           ("Sick","Relax") : [0,0],
           ("Sick", "Party") : [2,2]},
     {('Healthy', 'Relax'): 7.0,
 ('Healthy', 'Party'): 10.0,
 ('Sick', 'Relax'): 0.0,
 ('Sick', 'Party'): 2.0})
])

def test_Expected_Reward(Probs,Rewards,Expected):
    assert Expected_Reward(Probs,Rewards) == pytest.approx(Expected)