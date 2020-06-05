# La-methode-du-gradient-projete-pour-MPCC
Résolution des MPCC par une méthode classique de la PNL
Le premier fichier ProjectedGradientSolver.py est un solveur des sous problemes de la forme minimiser L_rho(x,z) sujette a z >= lb. L_rho(x,z) = f(x) + eta * F(x,z) + (rho/2) * ||F(x,z)||^2.
Le solveur est tester sur le probleme problemes sur une instance de la collection MacMPEC ex9.2.1, qui est de la forme min f(x) s.a F(x,z) = 0, z >=0.
