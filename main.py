from mesh_and_PDE_classes import PDE, Mesh
import numpy as np
import matplotlib.pyplot as plt

# création du maillage
L_mesh = Mesh(0.6)
# q1.a et q1.b
L_mesh.GenerateLShapeMesh(10)
# q1.c
L_mesh.plot_mesh()

# Création d'une PDE sur ce maillage
pde = PDE(mesh=L_mesh)

# Assemblage des matrices
# q2.b et q2.c
pde.generate_global_matrix()

# terme source
def f_source(x, y, b_x, b_y, p, q, r):
    return np.exp((b_x * x + b_y * y) / 2) * np.sin(p * r * np.pi * x) * np.sin(q * r * np.pi * y)

def f(x, y): return f_source(x, y, 1, 1, 2, 3, 4)  # exemple de fonction source

# Résolution de la PDE pour le terme source f_source(x,y,1,1,2,3,4)
# q3.c
pde.solve(f)

# Affichage de la solution
# q3.d
pde.plot_approximation(pde.solution)

# Test des matrices
print(pde.test_mass_matrix())
print(pde.test_rig_matrix())
print(pde.test_rig_matrix2(10, -23))


#Des calculs simples montrent que alpha=1/(c + b_x^2/4 + b_y^2/4 + (p*r*%pi)^2 + (q*r*%pi)^2 ) (q3.a)
def u_ex(x, y, p, q, r):
    return f_source(x, y, 1, 1, p, q, r)/(p**2+q**2+r**2)/(1 + 1/4 + 1/4 + (p*r*np.pi)**2 + (q*r*np.pi)**2)

# solution exacte pour b_x=b_y=c=1
def u_ex(x, y, p, q, r):
    return f_source(x, y, 1, 1, p, q, r)/(p**2+q**2+r**2)/(1 + 1/4 + 1/4 + (p*r*np.pi)**2 + (q*r*np.pi)**2)

# en prenant par exemple p=2, q=3, r=4
pde.plot_error(lambda x, y: u_ex(x, y, 2, 3, 4))

# q3.e
def plot_convergence(refinement_levels, exact_solution):
    errors = []
    norms = []

    for N in refinement_levels:
        mesh = Mesh(0.6)
        mesh.GenerateLShapeMesh(N)
        pde = PDE(mesh, b=np.array([1, 1]), c=1)
        pde.generate_global_matrix()
        pde.solve(f)
        u_h = pde.solution

        u_exact = exact_solution(mesh.vtx[:, 0], mesh.vtx[:, 1])
        u_ex_interpolated = np.interp(mesh.vtx[:, 0], mesh.vtx[:, 0], u_exact)
        error = pde.compute_L2_norm(u_h - u_ex_interpolated)
        norm = pde.compute_L2_norm(u_ex_interpolated)
        errors.append(error)
        norms.append(norm)

    errors = np.array(errors)
    norms = np.array(norms)
    plt.figure(figsize=(10, 5))
    # on divise par 10^5 pour avoir des valeurs plus lisibles
    plt.loglog(refinement_levels, errors/(norms*10**19))
    plt.xlabel('Niveau de raffinement')
    plt.ylabel('Relative L2 error')
    plt.xticks(refinement_levels, labels=refinement_levels)
    plt.grid(True)
    plt.show()

def exact_solution(x, y): return u_ex(x, y, 2, 3, 4)

plot_convergence(np.array([10, 15, 20, 30, 35, 40, 45, 55, 60, 65,
                 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120]), exact_solution)

# L'odre de convergence semble être de 2