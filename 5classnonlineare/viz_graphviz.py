# viz_graphviz.py
import graphviz

def salva_albero_da_dot(dot_file_path, output_filename="albero_graphviz"):
    """
    Legge un file .dot salvato su disco e genera un'immagine PNG dell'albero.
    Richiede l'installazione di Graphviz sul sistema operativo.
    """
    try:
        # Carica il contenuto del file .dot
        graph = graphviz.Source.from_file(dot_file_path)
        
        # Salva l'immagine. cleanup=True elimina i file temporanei generati durante il processo
        graph.render(output_filename, format="png", cleanup=True)
        
        print(f"Immagine generata con successo: {output_filename}.png")
        
    except Exception as e:
        print(f"Errore durante la generazione dell'immagine: {e}")
        print("Assicurati di aver installato Graphviz anche sul tuo sistema operativo, non solo tramite pip.")