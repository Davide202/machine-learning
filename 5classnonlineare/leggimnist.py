import numpy as np
import struct

filedir = '../data/MNIST/'

def carica_immagini_mnist(percorso_file):
    """
    Legge i file binari originali di MNIST contenenti le immagini 
    e li converte in un array NumPy.
    """
    with open(filedir + percorso_file, 'rb') as f:
        # L'intestazione delle immagini è di 16 byte (4 interi a 32-bit: magic_number, num_images, rows, cols)
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        
        # Legge i restanti byte (i pixel) come interi senza segno a 8-bit
        immagini = np.frombuffer(f.read(), dtype=np.uint8)
        
        # Rimodella l'array in una matrice 2D: ogni riga è un'immagine srotolata (784 pixel)
        # Se preferisci la griglia 28x28, usa: reshape(num_images, 28, 28)
        immagini = immagini.reshape(num_images, rows * cols)
        return immagini

def carica_etichette_mnist(percorso_file):
    """
    Legge i file binari originali di MNIST contenenti le etichette 
    e li converte in un array NumPy.
    """
    with open(filedir + percorso_file, 'rb') as f:
        # L'intestazione delle etichette è di 8 byte (2 interi a 32-bit: magic_number, num_labels)
        magic, num_labels = struct.unpack(">II", f.read(8))
        
        # Legge le etichette
        etichette = np.frombuffer(f.read(), dtype=np.uint8)
        return etichette
