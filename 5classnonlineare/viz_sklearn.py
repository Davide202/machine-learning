# viz_sklearn.py
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def salva_albero_da_modello(modello, colonne, output_filename="albero_sklearn.png", profondita_max=None):
    """
    Genera e salva l'immagine di un albero decisionale passando direttamente il modello addestrato.
    Non utilizza file .dot intermedi.
    """
    # Creiamo una figura bella grande per evitare testi sovrapposti
    plt.figure(figsize=(50,20))
    
    # Disegniamo l'albero
    plot_tree(modello, 
              feature_names=colonne, 
              class_names=['Morto', 'Sopravvissuto'],
              filled=True, 
              rounded=True, 
              fontsize=10,
              max_depth=profondita_max)
    
    # Salviamo l'immagine
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    # Chiudiamo la figura per liberare la memoria RAM
    plt.close()
    
    print(f"Immagine generata con successo: {output_filename}")