from mesh_and_PDE_classes.py import PDE, Mesh

# Création d'un maillage rectangulaire
rect_mesh = Mesh.generate_rectangle(Lx=1.0, Ly=1.0, Nx=10, Ny=10)

# Affichage du maillage
rect_mesh.plot()

# Création d'une PDE sur ce maillage
pde = PDE(mesh=rect_mesh, b=np.array([1, 1]), c=1)

# Assemblage des matrices
rig_matrix = pde.generate_rig_matrix()
second_matrix = pde.generate_second_matrix()
mass_matrix = pde.generate_mass_matrix()

# Assemblage du terme source
f = lambda x, y: np.sin(x) * np.sin(y)  # exemple de fonction source
source_term = pde.assemble_source_term(f)

# Résolution de la PDE
solution = pde.solve(p=1, q=1, r=1)

# Affichage de la solution
pde.plot_approximation(solution, solution)  # ici on suppose que la solution exacte est égale à la solution numérique

# Test des matrices
print(pde.test_mass_matrix(beta=0.5))
print(pde.test_rig_matrix(beta=0.5))
