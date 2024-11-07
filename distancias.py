import numpy as np

def levenshtein_matriz(x, y, threshold=None):
    # esta versión no utiliza threshold, se pone porque se puede
    # invocar con él, en cuyo caso se ignora
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int)
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
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int)
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
            seqOps.append(('match', x[i - 1], y[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + 1:
            seqOps.append(('substitute', x[i - 1], y[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            seqOps.append(('delete', x[i - 1], ''))
            i -= 1
        elif j > 0 and D[i][j] == D[i][j - 1] + 1:
            seqOps.append(('insert', '', y[j - 1]))
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
    # En aquesta funció calculem la distància de Levenshtein
    # utilitzant una tècnica de reducció d'espai amb vectors
    lenX, lenY = len(x), len(y)
    vAnt = [0] * (lenX + 1)  # Vector per a la fila anterior
    vAct = [0] * (lenX + 1)   # Vector per a la fila actual

    for i in range(1, lenX + 1):
        vAnt[i] = i  # Inicialitzem el vector anterior

    for j in range(1, lenY + 1):
        vAct[0] = j  # Inicialitzem la primera posició
        for i in range(1, lenX + 1):
            vAct[i] = min(
                vAct[i - 1] + 1,      # Cost d'eliminació
                vAnt[i] + 1,          # Cost d'inserció
                vAnt[i - 1] + (x[i - 1] != y[j - 1]),  # Cost de substitució
            )
        if vAct[lenX] > threshold:  # Comprovem si supera el llindar
            return threshold + 1
        vAnt = vAct[:]  # Actualitzem el vector anterior

    return vAct[lenX]  # Retornem la distància final

def levenshtein_cota_optimista(x, y, threshold):
    # Aquesta funció calcula una cota optimista
    # basada en el recompte de caràcters de les dues cadenes
    count_x = {}
    count_y = {}

    for char in x:
        count_x[char] = count_x.get(char, 0) + 1  # Comptem els caràcters de x
    for char in y:
        count_y[char] = count_y.get(char, 0) + 1  # Comptem els caràcters de y

    # Calculem la suma de les diferències en els comptatges
    total_diff = sum(abs(count_x.get(char, 0) - count_y.get(char, 0)) for char in set(count_x.keys()).union(set(count_y.keys())))

    if total_diff > threshold:
        return threshold + 1  # Retornem el llindar + 1 si es supera

    return levenshtein(x, y, threshold)  # Cridem a la funció de Levenshtein real


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




def damerau_intermediate_matriz(x, y, threshold=None):
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)

    for i in range(lenX + 1):
        D[i][0] = i
    for j in range(lenY + 1):
        D[0][j] = j

    for i in range(1, lenX + 1):
        for j in range(1, lenY + 1):
            cost = 0 if x[i - 1] == y[j - 1] else 1
            D[i][j] = min(
                D[i - 1][j] + 1,        #Borrado
                D[i][j - 1] + 1,        #Inserción
                D[i - 1][j - 1] + cost  #Sustitución
            )

            #Transposición ab ↔ ba
            if i > 1 and j > 1 and x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1]:
                D[i][j] = min(D[i][j], D[i - 2][j - 2] + 1)
            
            #Transposición ab ↔ bca
            if i > 2 and j > 1 and x[i - 3:i] == y[j - 1] + x[i - 3:i-1]:
                D[i][j] = min(D[i][j], D[i - 3][j - 1] + 2)

            #Transposición acb ↔ ba
            if i > 1 and j > 2 and x[i - 1] + x[i - 3:i-1] == y[j - 3:j-1] + y[j - 1]:
                D[i][j] = min(D[i][j], D[i - 2][j - 3] + 2)

    return D[lenX][lenY]


def damerau_intermediate_edicion(x, y, threshold=None):
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)

    for i in range(lenX + 1):
        D[i][0] = i
    for j in range(lenY + 1):
        D[0][j] = j

    for i in range(1, lenX + 1):
        for j in range(1, lenY + 1):
            cost = 0 if x[i - 1] == y[j - 1] else 1
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + cost
            )

            # Transposición ab ↔ ba
            if i > 1 and j > 1 and x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1]:
                D[i][j] = min(D[i][j], D[i - 2][j - 2] + 1)

            # Transposición ab ↔ bca
            if i > 2 and j > 1 and x[i - 3:i] == y[j - 1] + x[i - 3:i-1]:
                D[i][j] = min(D[i][j], D[i - 3][j - 1] + 2)

            # Transposición acb ↔ ba
            if i > 1 and j > 2 and x[i - 1] + x[i - 3:i-1] == y[j - 3:j-1] + y[j - 1]:
                D[i][j] = min(D[i][j], D[i - 2][j - 3] + 2)

    # Recuperar secuencia de operaciones
    seqOps = []
    i, j = lenX, lenY
    while i > 0 or j > 0:
        if i > 2 and j > 1 and D[i][j] == D[i - 3][j - 1] + 2 and x[i - 3:i] == y[j - 1] + x[i - 3:i-1]:
            seqOps.append(('transpose3', x[i - 3:i], y[j - 1] + x[i - 3:i-1]))
            i -= 3
            j -= 1
        elif i > 1 and j > 2 and D[i][j] == D[i - 2][j - 3] + 2 and x[i - 1] + x[i - 3:i-1] == y[j - 3:j-1] + y[j - 1]:
            seqOps.append(('transpose3', x[i - 3:i], y[j - 3:j]))
            i -= 2
            j -= 3
        elif i > 1 and j > 1 and x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1] and D[i][j] == D[i - 2][j - 2] + 1:
            seqOps.append(('transpose', x[i - 2:i], y[j - 2:j]))
            i -= 2
            j -= 2
        elif i > 0 and j > 0 and x[i - 1] == y[j - 1]:
            seqOps.append(('match', x[i - 1], y[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + 1:
            seqOps.append(('substitute', x[i - 1], y[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            seqOps.append(('delete', x[i - 1], ''))
            i -= 1
        elif j > 0 and D[i][j] == D[i][j - 1] + 1:
            seqOps.append(('insert', '', y[j - 1]))
            j -= 1

    return D[lenX][lenY], seqOps[::-1]

    
def damerau_intermediate(x, y, threshold=None):
    lenX, lenY = len(x), len(y)
    
    vprev3 = [0] * (lenY + 1)
    vprev2 = [0] * (lenY + 1)
    vprev = [j for j in range(lenY + 1)]
    vcurrent = [0] * (lenY + 1)
    
    for i in range(1, lenX + 1):
        vcurrent[0] = i
        min_val = float('inf')  #Valor mínimo en la fila actual
        
        for j in range(1, lenY + 1):
            cost = 0 if x[i - 1] == y[j - 1] else 1
            vcurrent[j] = min(vprev[j] + 1, vcurrent[j - 1] + 1, vprev[j - 1] + cost)
            min_val = min(min_val, vcurrent[j])
            
            # Transposición ab ↔ ba
            if i > 1 and j > 1 and x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1]:
                vcurrent[j] = min(vcurrent[j], vprev2[j - 2] + 1)
            
            # Transposición ab ↔ bca
            if i > 2 and j > 1 and x[i - 3:i] == y[j - 1] + x[i - 3:i-1]:
                vcurrent[j] = min(vcurrent[j], vprev3[j - 1] + 2)

            # Transposición acb ↔ ba
            if i > 1 and j > 2 and x[i - 1] + x[i - 3:i-1] == y[j - 3:j-1] + y[j - 1]:
                vcurrent[j] = min(vcurrent[j], vprev2[j - 3] + 2)
        
        if min_val > threshold:
            return threshold + 1

        vprev3, vprev2, vprev, vcurrent = vprev2, vprev, vcurrent, vprev3

    return vprev[lenY]


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

