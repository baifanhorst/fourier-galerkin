# Elastic properties

class Elasticity():
    
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu
        self.G = E / 2 / (1+nu)
        self.lamb = nu * E / (1+nu) / (1-2*nu)
        self.c1 = E / (1 - nu**2)
        self.c2 = nu * E / (1 - nu**2)