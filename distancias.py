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
    # completar versión reducción coste espacial y parada por threshold
    return min(0,threshold+1) # COMPLETAR Y REEMPLAZAR ESTA PARTE

def levenshtein_cota_optimista(x, y, threshold):
    return 0 # COMPLETAR Y REEMPLAZAR ESTA PARTE

def damerau_restricted_matriz(x, y, threshold=None):
    # completar versión Damerau-Levenstein restringida con matriz
    lenX, lenY = len(x), len(y)
    # COMPLETAR
    return 0 # COMPLETAR Y REEMPLAZAR ESTA PARTE

def damerau_restricted_edicion(x, y, threshold=None):
    # partiendo de damerau_restricted_matriz añadir recuperar
    # secuencia de operaciones de edición
    return 0,[] # COMPLETAR Y REEMPLAZAR ESTA PARTE

def damerau_restricted(x, y, threshold=None):
    # versión con reducción coste espacial y parada por threshold
     return min(0,threshold+1) # COMPLETAR Y REEMPLAZAR ESTA PARTE

def damerau_intermediate_matriz(x, y, threshold=None):
    # completar versión Damerau-Levenstein intermedia con matriz
     return min(0,threshold+1) # COMPLETAR Y REEMPLAZAR ESTA PARTE

def damerau_intermediate_edicion(x, y, threshold=None):
    # partiendo de matrix_intermediate_damerau añadir recuperar
    # secuencia de operaciones de edición
    # completar versión Damerau-Levenstein intermedia con matriz
    return 0,[] # COMPLETAR Y REEMPLAZAR ESTA PARTE
    
def damerau_intermediate(x, y, threshold=None):
    # versión con reducción coste espacial y parada por threshold
    return min(0,threshold+1) # COMPLETAR Y REEMPLAZAR ESTA PARTE

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

