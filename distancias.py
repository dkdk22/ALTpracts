import numpy as np

def levenshtein_matriz(x, y, threshold=None):
    # esta versión no utiliza threshold, se pone porque se puede
    # invocar con él, en cuyo caso se ignora
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
            )
    return D[lenX, lenY]

def levenshtein_edicion(x, y, threshold=None):
    # a partir de la versión levenshtein_matriz
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
            )
    seqOps = []; #creem una llista buida on posarem la seqüència d'operacions
    i, j = lenX, lenY;
    while i > 0 or j > 0: #en aquest bucle recorreguem la matriu fixant-nos sempre en la diagonal que dona el resultat, per a cada element d'aquesta veiem el contingut de les cel.les adjacents i així determinem l'operació
        if i > 0 and j > 0 and x[i - 1] == y[j - 1]:
            seqOps.append((x[i - 1], y[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + 1:
            seqOps.append(( x[i - 1], y[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            seqOps.append(( x[i - 1], ''))
            i -= 1
        elif j > 0 and D[i][j] == D[i][j - 1] + 1:
            seqOps.append(( '', y[j - 1]))
            j -= 1
    return D[lenX, lenY], seqOps[::-1] #tornem tant la distància com la seqüència, aquesta última invertida per a que estiga en l'ordre correcte

def levenshtein_reduccion(x, y, threshold=None):
    #En aquest codi implementem una modificació del levenshtein original però només gastant 2 vectors en volta d'una matriu
    lenX, lenY = len(x), len(y)
    vAnt = [0] * (lenX + 1)
    vAct = [0] * (lenX + 1)
    for i in range(1, lenX + 1):
        vAnt[i] = vAnt[i - 1] + 1
    for j in range(1, lenY + 1):
        vAct[0]=j
        for i in range(1, lenX + 1):
            vAct[i] = min(
                vAct[i - 1] + 1,
                vAnt[i] + 1,
                vAnt[i - 1] + (x[i - 1] != y[j - 1]),
            )
        vAnt=vAct[:]
    
    return vAct[lenX]

def levenshtein(x, y, threshold):
# Calculem la distància de Levenshtein utilitzant vectors en comptes de matrius per optimitzar l'espai
    
    lenX, lenY = len(x), len(y)
    vAnt = [0] * (lenX + 1)  # Vector anterior
    vAct = [0] * (lenX + 1)  # Vector actual

    # Inicialitzem la primera columna
    for i in range(1, lenX + 1):
        vAnt[i] = vAnt[i - 1] + 1

    # Recorrem cada fila
    for j in range(1, lenY + 1):
        vAct[0] = vAnt[0] + 1  # Inicialitzem la primera cel·la

        paradaPorThreshold = True  # Comprovem si hem de parar pel llindar
        if(vAct[0] <= threshold): 
            paradaPorThreshold = False
        elif(vAct[0] == threshold and lenX - i == lenY - j): 
            paradaPorThreshold = False

        # Recorrem cada columna
        for i in range(1, lenX + 1):
            vAct[i] = min(vAct[i - 1] + 1,  # Inserció
                          vAnt[i] + 1,      # Eliminació
                          vAnt[i - 1] + (x[i - 1] != y[j - 1]))  # Substitució

            # Si la distància és menor que el llindar, continuem
            if(vAct[i] < threshold): 
                paradaPorThreshold = False
            elif(vAct[i] == threshold and lenX - i == lenY - j): 
                paradaPorThreshold = False

        # Si s'ha superat el llindar, parem i tornem el llindar + 1
        if(paradaPorThreshold): 
            return threshold + 1
        
        # Actualitzem els vectors per la següent iteració
        vAct, vAnt = vAnt, vAct
    
    return vAnt[lenX]  # Retornem la distància final

def levenshtein_cota_optimista(x, y, threshold):
    # S'afegeixen totes les lletres de les dues cadenes a un conjunt
    dic = set(x).union(set(y))
    
    # Inicialitzem un diccionari per emmagatzemar les sumes de les diferències
    res = {1: 0, -1: 0}

    # Recorrem el conjunt de caràcters
    for letra in dic:
        # Calculem la diferència d'aparicions de cada lletra en ambdues cadenes
        dif = x.count(letra) - y.count(letra)
        
        # Si la diferència és negativa, la sumem al comptador de caràcters faltants en y
        if dif < 0:
            res[1] += abs(dif)
        # Si la diferència és positiva, la sumem al comptador de caràcters faltants en x
        else:
            res[-1] += abs(dif)
    
    # Comprovem si el resultat final és més gran o igual que el llindar donat
    res = max(res[1], res[-1])
    
    if res > threshold:
        # Si la diferència és major que el llindar, retornem threshold + 1
        return threshold + 1
    else:
        # En cas contrari, calculem la distància de Levenshtein
        return levenshtein(x, y, threshold)


def damerau_restricted_matriz(x, y, threshold=None):
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)
    
    # Inicialización de la matriz
    for i in range(1, lenX + 1):
        D[i][0] = i
    for j in range(1, lenY + 1):
        D[0][j] = j

    # Llenado de la matriz con operaciones de edición y transposición adyacente
    for i in range(1, lenX + 1):
        for j in range(1, lenY + 1):
            cost = 0 if x[i - 1] == y[j - 1] else 1
            D[i][j] = min(
                D[i - 1][j] + 1,             # Borrado
                D[i][j - 1] + 1,             # Inserción
                D[i - 1][j - 1] + cost       # Sustitución
            )
            
            # Transposición si hay dos caracteres adyacentes intercambiables
            if i > 1 and j > 1 and x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1]:
                D[i][j] = min(D[i][j], D[i - 2][j - 2] + 1)

    return D[lenX, lenY]


def damerau_restricted_edicion(x, y, threshold=None):
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)
    
    # Inicialización de la matriz
    for i in range(1, lenX + 1):
        D[i][0] = i
    for j in range(1, lenY + 1):
        D[0][j] = j

    # Llenado de la matriz con transposición
    for i in range(1, lenX + 1):
        for j in range(1, lenY + 1):
            cost = 0 if x[i - 1] == y[j - 1] else 1
            D[i][j] = min(
                D[i - 1][j] + 1,             # Borrado
                D[i][j - 1] + 1,             # Inserción
                D[i - 1][j - 1] + cost       # Sustitución
            )
            if i > 1 and j > 1 and x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1]:
                D[i][j] = min(D[i][j], D[i - 2][j - 2] + 1)

    # Recuperación de la secuencia de edición
    i, j = lenX, lenY
    ediciones = []
    while i > 0 or j > 0:
        if i > 0 and D[i][j] == D[i - 1][j] + 1:
            ediciones.append((x[i - 1], ''))  # Borrado
            i -= 1
        elif j > 0 and D[i][j] == D[i][j - 1] + 1:
            ediciones.append(('', y[j - 1]))  # Inserción
            j -= 1
        elif i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (x[i - 1] != y[j - 1]):
            ediciones.append((x[i - 1], y[j - 1]))  # Sustitución
            i -= 1
            j -= 1
        elif i > 1 and j > 1 and x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1] and D[i][j] == D[i - 2][j - 2] + 1:
            ediciones.append((x[i - 2] + x[i - 1], y[j - 2] + y[j - 1]))  # Transposición
            i -= 2
            j -= 2

    ediciones.reverse()
    return D[lenX, lenY], ediciones

def damerau_restricted(x, y, threshold=None):
    lenX, lenY = len(x), len(y)
    if threshold is not None and abs(lenX - lenY) > threshold:
        return threshold + 1

    # Inicialización de los vectores de distancia
    prev_row = np.arange(lenY + 1)
    curr_row = np.zeros(lenY + 1, dtype=int)
    prev_prev_row = np.zeros(lenY + 1, dtype=int)

    for i in range(1, lenX + 1):
        curr_row[0] = i
        min_cost_in_row = curr_row[0]

        for j in range(1, lenY + 1):
            cost = 0 if x[i - 1] == y[j - 1] else 1
            curr_row[j] = min(
                prev_row[j] + 1,           # Borrado
                curr_row[j - 1] + 1,       # Inserción
                prev_row[j - 1] + cost     # Sustitución
            )

            # Transposición
            if i > 1 and j > 1 and x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1]:
                curr_row[j] = min(curr_row[j], prev_prev_row[j - 2] + 1)

            min_cost_in_row = min(min_cost_in_row, curr_row[j])

        # Intercambio de referencias de vectores
        prev_prev_row, prev_row, curr_row = prev_row, curr_row, prev_prev_row

        # Parada por threshold si el mínimo de la fila supera el umbral
        if threshold is not None and min_cost_in_row > threshold:
            return threshold + 1

    return prev_row[lenY]



def damerau_intermediate(x, y, threshold=None):
    lenX, lenY = len(x), len(y)
    
    # Si la diferencia de longitudes supera el umbral, no hay necesidad de calcular
    if threshold is not None and abs(lenX - lenY) > threshold:
        return threshold + 1

    # Inicialización de los vectores de distancia
    prev_row = np.arange(lenY + 1)
    curr_row = np.zeros(lenY + 1, dtype=int)
    prev_prev_row = np.zeros(lenY + 1, dtype=int)
    prev_prev_prev_row = np.zeros(lenY + 1, dtype=int)  # Necesaria para la transposición de tres caracteres

    for i in range(1, lenX + 1):
        curr_row[0] = i
        min_cost_in_row = curr_row[0]

        for j in range(1, lenY + 1):
            cost = 0 if x[i - 1] == y[j - 1] else 1
            curr_row[j] = min(
                prev_row[j] + 1,           # Borrado
                curr_row[j - 1] + 1,       # Inserción
                prev_row[j - 1] + cost     # Sustitución
            )

            # Transposición de dos caracteres adyacentes (coste 1)
            if i > 1 and j > 1 and x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1]:
                curr_row[j] = min(curr_row[j], prev_prev_row[j - 2] + 1)

            # Transposición de tres caracteres: acb ↔ ba (coste 2)
            if i > 2 and j > 1 and x[i - 3] == y[j - 1] and x[i - 1] == y[j - 2]:
                curr_row[j] = min(curr_row[j], prev_prev_prev_row[j - 2] + 2)

            # Transposición de tres caracteres: ab ↔ bca (coste 2)
            if i > 1 and j > 2 and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 3]:
                curr_row[j] = min(curr_row[j], prev_prev_row[j - 3] + 2)

            # Mantener el menor costo en la fila para comparación con el umbral
            min_cost_in_row = min(min_cost_in_row, curr_row[j])

        # Intercambio de referencias de vectores
        prev_prev_prev_row, prev_prev_row, prev_row, curr_row = prev_prev_row, prev_row, curr_row, prev_prev_prev_row

        # Parada temprana si el umbral se supera
        if threshold is not None and min_cost_in_row > threshold:
            return threshold + 1

    return prev_row[lenY]


def damerau_intermediate_edicion(x, y, threshold=None):
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)
    
    # Inicialización de la matriz
    for i in range(1, lenX + 1):
        D[i][0] = i
    for j in range(1, lenY + 1):
        D[0][j] = j

    # Llenado de la matriz con transposiciones extendidas
    for i in range(1, lenX + 1):
        for j in range(1, lenY + 1):
            cost = 0 if x[i - 1] == y[j - 1] else 1
            D[i][j] = min(
                D[i - 1][j] + 1,             # Borrado
                D[i][j - 1] + 1,             # Inserción
                D[i - 1][j - 1] + cost       # Sustitución
            )
            
            # Transposición de dos caracteres adyacentes
            if i > 1 and j > 1 and x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1]:
                D[i][j] = min(D[i][j], D[i - 2][j - 2] + 1)
            
            # Transposición de tres caracteres: acb ↔ ba (coste 2)
            if i > 2 and j > 1 and x[i - 3] == y[j - 1] and x[i - 1] == y[j - 2]:
                D[i][j] = min(D[i][j], D[i - 3][j - 2] + 2)

            # Transposición de tres caracteres: ab ↔ bca (coste 2)
            if i > 1 and j > 2 and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 3]:
                D[i][j] = min(D[i][j], D[i - 2][j - 3] + 2)

    # Recuperación de la secuencia de edición
    i, j = lenX, lenY
    ediciones = []
    
    while i > 0 or j > 0:
        if i > 0 and D[i][j] == D[i - 1][j] + 1:
            ediciones.append((x[i - 1], ''))  # Borrado
            i -= 1
        elif j > 0 and D[i][j] == D[i][j - 1] + 1:
            ediciones.append(('', y[j - 1]))  # Inserción
            j -= 1
        elif i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (x[i - 1] != y[j - 1]):
            ediciones.append((x[i - 1], y[j - 1]))  # Sustitución
            i -= 1
            j -= 1
        elif i > 1 and j > 1 and D[i][j] == D[i - 2][j - 2] + 1:
            ediciones.append((x[i - 2] + x[i - 1], y[j - 2] + y[j - 1]))  # Transposición adyacente
            i -= 2
            j -= 2
        elif i > 2 and j > 1 and D[i][j] == D[i - 3][j - 2] + 2:
            ediciones.append((x[i - 3] + x[i - 1], y[j - 2] + y[j - 1]))  # Transposición acb ↔ ba
            i -= 3
            j -= 2
        elif i > 1 and j > 2 and D[i][j] == D[i - 2][j - 3] + 2:
            ediciones.append((x[i - 2] + x[i - 1], y[j - 3] + y[j - 1]))  # Transposición ab ↔ bca
            i -= 2
            j -= 3

    ediciones.reverse()
    return D[lenX, lenY], ediciones


def damerau_intermediate_matriz(x, y, threshold=None):
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)
    
    # Inicialización de la matriz
    for i in range(1, lenX + 1):
        D[i][0] = i
    for j in range(1, lenY + 1):
        D[0][j] = j

    # Llenado de la matriz con las operaciones de edición y transposiciones extendidas
    for i in range(1, lenX + 1):
        for j in range(1, lenY + 1):
            cost = 0 if x[i - 1] == y[j - 1] else 1
            D[i][j] = min(
                D[i - 1][j] + 1,             # Borrado
                D[i][j - 1] + 1,             # Inserción
                D[i - 1][j - 1] + cost       # Sustitución
            )
            
            # Transposición de dos caracteres adyacentes
            if i > 1 and j > 1 and x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1]:
                D[i][j] = min(D[i][j], D[i - 2][j - 2] + 1)
            
            # Transposición de tres caracteres: acb ↔ ba (coste 2)
            if i > 2 and j > 1 and x[i - 3] == y[j - 1] and x[i - 1] == y[j - 2]:
                D[i][j] = min(D[i][j], D[i - 3][j - 2] + 2)

            # Transposición de tres caracteres: ab ↔ bca (coste 2)
            if i > 1 and j > 2 and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 3]:
                D[i][j] = min(D[i][j], D[i - 2][j - 3] + 2)

    return D[lenX, lenY]


opcionesSpell = {
    'levenshtein_m': levenshtein_matriz,
    'levenshtein_r': levenshtein_reduccion,
    'levenshtein':   levenshtein,
    'levenshtein_o': levenshtein_cota_optimista,
    'damerau_rm':    damerau_restricted_matriz,
    'damerau_r':     damerau_restricted,
    'damerau_im':    damerau_intermediate_matriz,
    'damerau_i':     damerau_intermediate
}

opcionesEdicion = {
    'levenshtein': levenshtein_edicion,
    'damerau_r':   damerau_restricted_edicion,
    'damerau_i':   damerau_intermediate_edicion
}

