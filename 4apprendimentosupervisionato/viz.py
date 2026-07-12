# viz.py
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

def showBounds(model, X, Y, labels=['Classe 0', 'Classe 1'], title='Decision Boundary', xlabel='Feature 1', ylabel='Feature 2'):
    """
    Disegna il Decision Boundary di un modello di classificazione 2D.
    """
    plt.figure(figsize=(10, 6))
    
    # Crea lo sfondo colorato
    disp = DecisionBoundaryDisplay.from_estimator(
        model, 
        X, 
        response_method="predict",
        alpha=0.5, 
        cmap=plt.cm.RdYlBu
    )
    
    # Sovrappone i punti del dataset
    scatter = disp.ax_.scatter(
        X[:, 0], 
        X[:, 1], 
        c=Y, 
        edgecolors='k', 
        cmap=plt.cm.RdYlBu
    )
    
    # Imposta testi e legenda personalizzati
    disp.ax_.set_title(title, fontsize=14)
    disp.ax_.set_xlabel(xlabel)
    disp.ax_.set_ylabel(ylabel)
    
    handles, _ = scatter.legend_elements()
    disp.ax_.legend(handles, labels, loc="best")
    
    plt.show()