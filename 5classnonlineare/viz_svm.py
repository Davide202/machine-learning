# viz_svm.py
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np

def plot_decision_boundaries(model, X, Y, title="Decision Boundaries"):
    """
    Disegna i confini di decisione di un classificatore 2D.
    Utilizza la colormap 'viridis' per replicare i colori standard (viola, verde acqua, giallo).
    """
    plt.figure(figsize=(10, 6))
    
    # Crea lo sfondo colorato in base alle previsioni del modello
    disp = DecisionBoundaryDisplay.from_estimator(
        model, 
        X, 
        response_method="predict",
        alpha=0.5, # 0.9, 
        #cmap='viridis' # Mappa di colori identica all'immagine allegata
    )
    
    # Sovrappone i punti reali del dataset
    disp.ax_.scatter(
        X[:, 0], 
        X[:, 1], 
        c=Y, 
        edgecolors='k', 
        #cmap='viridis',
        s=25 # 20 # Dimensione dei pallini
    )
    
    disp.ax_.set_title(title, fontsize=14)
    plt.show()

def plot_svm_bounds(model, X_train, Y_train, X_test=None, Y_test=None, title="SVM Decision Boundary"):
    """
    Disegna i confini di decisione di una SVM.
    Permette di mostrare sia i dati di Training (Cerchi) che quelli di Test (Triangoli).
    """
    plt.figure(figsize=(8, 6))
    
    # Se abbiamo anche i dati di test, uniamo X_train e X_test solo per dire 
    # a Scikit-Learn quanto deve disegnare grande la mappa di sfondo
    if X_test is not None:
        X_all = np.vstack((X_train, X_test))
    else:
        X_all = X_train
    
    # Crea lo sfondo sfumato (gradiente)
    disp = DecisionBoundaryDisplay.from_estimator(
        model, 
        X_all, 
        response_method="decision_function", 
        cmap='viridis', 
        alpha=0.8
    )
    
    # Sovrappone i punti del TRAIN SET (Pallini classici)
    disp.ax_.scatter(
        X_train[:, 0], 
        X_train[:, 1], 
        c=Y_train, 
        edgecolors='k', 
        cmap='viridis',
        s=35,
        label='Train Set (Cerchi)'
    )
    
    # Se passati, sovrappone i punti del TEST SET (Triangoli più grandi con bordo bianco)
    if X_test is not None and Y_test is not None:
        disp.ax_.scatter(
            X_test[:, 0], 
            X_test[:, 1], 
            c=Y_test, 
            edgecolors='white', # Bordo bianco per farli risaltare 
            cmap='viridis',
            s=60,
            marker='^',         # Forma a triangolo
            label='Test Set (Triangoli)'
        )
        disp.ax_.legend(loc='best') # Mostra la legenda
    
    disp.ax_.set_title(title, fontsize=14)
    plt.show()
