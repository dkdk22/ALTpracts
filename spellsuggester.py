# -*- coding: utf-8 -*-
import re

class SpellSuggester:

    """
    Clase que implementa el método suggest para la búsqueda de términos.
    """

    def __init__(self,
                 dist_functions,
                 vocab = [],
                 default_distance = None,
                 default_threshold = None):
        
        """Método constructor de la clase SpellSuggester

        Construye una lista de términos únicos (vocabulario),

        Args:
           dist_functions es un diccionario nombre->funcion_distancia
           vocab es una lista de palabras o la ruta de un fichero
           default_distance debe ser una clave de dist_functions
           default_threshold un entero positivo

        """
        self.distance_functions = dist_functions
        self.set_vocabulary(vocab)
        if default_distance is None:
            default_distance = 'levenshtein'
        if default_threshold is None:
            default_threshold = 3
        self.default_distance = default_distance
        self.default_threshold = default_threshold

    def build_vocabulary(self, vocab_file_path):
        """Método auxiliar para crear el vocabulario.

        Se tokeniza por palabras el fichero de texto,
        se eliminan palabras duplicadas y se ordena
        lexicográficamente.

        Args:
            vocab_file (str): ruta del fichero de texto para cargar el vocabulario.
            tokenizer (re.Pattern): expresión regular para la tokenización.
        """
        tokenizer=re.compile("\W+")
        with open(vocab_file_path, "r", encoding="utf-8") as fr:
            vocab = set(tokenizer.split(fr.read().lower()))
            vocab.discard("")  # por si acaso
            return sorted(vocab)

    def set_vocabulary(self, vocabulary):
        if isinstance(vocabulary,list):
            self.vocabulary = vocabulary # atención! nos quedamos una referencia, a tener en cuenta
        elif isinstance(vocabulary,str):
            self.vocabulary = self.build_vocabulary(vocabulary)
        else:
            raise Exception("SpellSuggester incorrect vocabulary value")

    def suggest(self, term, distance=None, threshold=None, flatten=True):
        """
        
         Cerca suggeriments per al terme introduït utilitzant les funcions de distància,
         agrupant-los per distància i aplanant-los si cal.

        Args:
            term (str): término de búsqueda.
            distance (str): nombre del algoritmo de búsqueda a utilizar
            threshold (int): threshold para limitar la búsqueda
        """
        if distance is None:
            distance = self.default_distance  # Si no es proporciona una distància, utilitzem la predeterminada.
        if threshold is None:
            threshold = self.default_threshold  # Si no es proporciona un llindar, utilitzem el predeterminat.

        dist_function = self.distance_functions.get(distance)  # Recuperem la funció de distància.
        suggestions = []  # Llista per emmagatzemar els suggeriments agrupats.
        grouped_suggestions = []  # Llista per emmagatzemar les paraules agrupades per distància.

        # Generem les suggerències, agrupades per distància
        for word in self.vocabulary:
            dist = dist_function(term, word, threshold)  # Calculant la distància entre el terme i la paraula.

            # Si la distància és menor o igual al llindar
            if dist <= threshold:
                # Ens assegurem que la llista de la distància existeixi
                while len(grouped_suggestions) <= dist:
                    grouped_suggestions.append([])

                # Afegim la paraula a la llista corresponent a aquesta distància
                grouped_suggestions[dist].append(word)

        # Si flatten és True, aplanem les llistes per obtenir una llista de paraules plana
        if flatten:
            suggestions = [item for sublist in grouped_suggestions for item in sublist]
        else:
            suggestions = grouped_suggestions  # Retornem les llistes agrupades per distància.

        return suggestions

