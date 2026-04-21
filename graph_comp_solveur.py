import matplotlib.pyplot as plt

# --- PREPARATION DES DONNÉES ---
# Le temps initial (Vrai départ) et t=0.0s sont tous deux à 5s après décalage
# On ajoute ensuite 5s à chaque valeur de la simulation (5.0, 10.0, etc.)
time = [0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0]

# --- SOLVEUR DU COURS ---
# Ajout de la condition initiale (553.15) au début
cours_min = [553.15, 549.88, 541.12, 537.05, 534.08, 531.68, 529.67, 527.92, 
             526.37, 524.97, 523.70, 522.53, 521.44, 520.44, 519.49]
cours_max = [553.15, 554.79, 556.32, 555.69, 554.38, 552.64, 550.66, 548.59, 
             546.50, 544.44, 542.45, 540.53, 538.69, 536.94, 535.27]

# --- SOLVEUR PICARD ---
# Ajout de la condition initiale (553.15) au début
picard_min = [553.15, 548.81, 540.57, 536.47, 533.45, 531.02, 528.97, 527.20, 
              525.63, 524.23, 522.95, 521.78, 520.70, 519.69, 518.76]
picard_max = [553.15, 554.82, 555.98, 555.27, 553.90, 552.11, 550.09, 548.00, 
              545.89, 543.82, 541.82, 539.89, 538.06, 536.31, 534.64]

# --- CRÉATION DU GRAPHIQUE ---
plt.figure(figsize=(12, 7))

# Tracé pour le Solveur du Cours (Rouge)
plt.plot(time, cours_max, color='red', linestyle='-', marker='o', label='Cours Max')
plt.plot(time, cours_min, color='red', linestyle='--', marker='x', label='Cours Min')

# Tracé pour le Solveur Picard (Bleu)
plt.plot(time, picard_max, color='blue', linestyle='-', marker='o', label='Picard Max')
plt.plot(time, picard_min, color='blue', linestyle='--', marker='x', label='Picard Min')

# Mise en forme
plt.title('Évolution des Températures (Par rapport a different solveur)', fontsize=14)
plt.xlabel('Temps (s)', fontsize=12)
plt.ylabel('Température (K)', fontsize=12)
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.6)


plt.tight_layout()
plt.show()