from mesh_and_PDE_classes import PDE, Mesh
import numpy as np

L_mesh=Mesh(0.6)
L_mesh.GenerateLShapeMesh(15)

# Création d'une PDE sur ce maillage
pde = PDE(mesh=L_mesh)

# Assemblage des matrices
pde.generate_global_matrix()

# Assemblage du terme source
def f_source(x, y, b_x, b_y, p, q, r):
    return np.exp((b_x * x + b_y * y) / 2) * np.sin(p * r * np.pi * x) * np.sin(q * r * np.pi * y)

f = lambda x, y: f_source(x,y,1,1,2,3,4)  # exemple de fonction source

# Résolution de la PDE
pde.solve(f)

# Affichage de la solution
pde.plot_approximation(pde.solution)  # ici on suppose que la solution exacte est égale à la solution numérique

# Test des matrices
print(pde.test_mass_matrix(beta=0.5))
print(pde.test_rig_matrix(beta=0.5))
