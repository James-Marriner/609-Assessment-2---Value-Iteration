import math

def Dot_Product(v1:list[any],
                v2:list[any])-> float:
    # Computes the dot product
    return(sum(map(math.prod, zip(v1,v2))))