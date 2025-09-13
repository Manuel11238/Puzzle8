
import tkinter as tk
from tkinter import messagebox
import heapq
import random
import copy

class Nodo:
    def __init__(self, estado, padre=None, movimiento=None, profundidad=0, costo=0):
        self.estado = estado
        self.padre = padre
        self.movimiento = movimiento
        self.profundidad = profundidad
        self.costo = costo

    def __lt__(self, otro):
        return (self.costo + self.heuristica()) < (otro.costo + otro.heuristica())


    #se usa el metodo manhattan para calcular la distanciapara llegar al estado
    ##objetivo
    def heuristica(self):
        # Heurística de Manhattan
        distancia = 0
        for i in range(3):
            for j in range(3):
                val = self.estado[i][j]
                if val != 0:
                    target_x = (val - 1) // 3
                    target_y = (val - 1) % 3
                    distancia += abs(i - target_x) + abs(j - target_y)
        return distancia
    #funcion que guarda ñlos nodos de la spolucion para imprimirlos posteriormente
    def get_camino(self):
        camino, nodo_actual = [], self
        while nodo_actual.padre:
            camino.append(nodo_actual)
            nodo_actual = nodo_actual.padre
        camino.reverse()
        return camino

# Funciones del juego
objetivo = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
#encuntra el espacio vacio en el tablero, el 0
def encontrar_cero(estado):
    for i in range(3):
        for j in range(3):
            if estado[i][j] == 0:
                return i, j


#busca obtener todos los posibles movimientos del tablero
def get_vecinos(estado):
    movimientos = []
    x, y = encontrar_cero(estado)
    direcciones = {'Arriba': (-1, 0), 'Abajo': (1, 0), 'Izquierda': (0, -1), 'Derecha': (0, 1)}

    for mov, (dx, dy) in direcciones.items():
        nueva_x, nueva_y = x + dx, y + dy
        if 0 <= nueva_x < 3 and 0 <= nueva_y < 3:
            nuevo_estado = copy.deepcopy(estado)
            nuevo_estado[x][y], nuevo_estado[nueva_x][nueva_y] = nuevo_estado[nueva_x][nueva_y], nuevo_estado[x][y]
            movimientos.append((nuevo_estado, mov))
    return movimientos


##es un metodoq ue se asegura que la lista generada con
##con el estado inicial es solucionable
def tiene_Solucion(estado):
    fichas = sum(estado, [])
    inv = 0
    for i in range(8):
        for j in range(i + 1, 9):
            if fichas[i] != 0 and fichas[j] != 0 and fichas[i] > fichas[j]:
                inv += 1
    return inv % 2 == 0



##metodo de busqueda, se uso el metoido A*
def a_star(estado_inicial):
    raiz = Nodo(estado_inicial)
    cola = []
    visitados = set()
    heapq.heappush(cola, raiz)

    while cola:
        nodo = heapq.heappop(cola)
        if nodo.estado == objetivo:
            return nodo.get_camino()

        visitados.add(tuple(map(tuple, nodo.estado)))

        for new_estado, mov in get_vecinos(nodo.estado):
            if tuple(map(tuple, new_estado)) not in visitados:
                nuevo_nodo = Nodo(new_estado, padre=nodo, movimiento=mov, profundidad=nodo.profundidad + 1, costo=nodo.costo + 1)
                heapq.heappush(cola, nuevo_nodo)

    return None

# para crear la ventana donde se muestra el tablero del puzzle
class Puzzle:
    def __init__(self, root):
        self.root = root
        self.root.title("Puzzle 8")
        self.tablero = [[0]*3 for _ in range(3)]
        self.botones = [[None]*3 for _ in range(3)]
        self.movimientos=0
        self.movimientos_label = tk.Label(root, text=f"Movimientos: {self.movimientos}", font=('Arial', 12))
        self.movimientos_label.grid(row=5, column=0, columnspan=3, pady=5)

        # Primero crear los botones
        for i in range(3):
            for j in range(3):
                btn = tk.Button(
                    root, text='', width=6, height=3, font=('Arial', 24),
                    command=lambda i=i, j=j: self.mover_ficha(i, j)
                )
                btn.grid(row=i, column=j, padx=2, pady=2)
                self.botones[i][j] = btn

        # Luego generar el tablero inicial
        self.generar_tablero()

        # Botones de control
        tk.Button(root, text="Resolver", command=self.resolver,  font=('Arial', 12)).grid(
            row=3, column=0, columnspan=3, sticky='nsew', padx=2, pady=2
        )
        tk.Button(root, text="Restablecer", command=self.generar_tablero, font=('Arial', 12)).grid(
            row=4, column=0, columnspan=3, sticky='nsew', padx=2, pady=2
        )
###,metodo que actualiza el label para mostarr los movimientos en la interfaz
    def act_movimientos(self):
     self.movimientos_label["text"]= f"Movimientos: {self.movimientos}"

    ##metodo que aleatoriamente genera el esyado inicial del tablero
    def generar_tablero(self):
        numeros = list(range(9))
        while True:
            random.shuffle(numeros)
            self.tablero = [numeros[i*3:(i+1)*3] for i in range(3)]
            if tiene_Solucion(self.tablero):
                break
        self.act_tablero()
        self.movimientos=0
        self.act_movimientos()
##va actualizando la posicion de los numeros en el tablero
    def act_tablero(self):
        for i in range(3):
            for j in range(3):
                val = self.tablero[i][j]
                self.botones[i][j]['text'] = '' if val == 0 else str(val)
##
    def mover_ficha(self, i, j):
        x, y = encontrar_cero(self.tablero)
        if abs(x - i) + abs(y - j) == 1:
            self.tablero[x][y], self.tablero[i][j] = self.tablero[i][j], self.tablero[x][y]
            self.act_tablero()
            self.movimientos += 1
            self.act_movimientos()
#metodo que consiste en llamar al metodso A estralla y
#despues actualizar el campo movimientos para mostrar que movimiento esta
    def resolver(self):
        camino = a_star(self.tablero)

        self.movimientos = len(camino)
        self.act_movimientos()

        for paso in camino:

            self.tablero = paso.estado
            self.act_tablero()
            self.root.update()
            self.root.after(300)

# para iniciar la interfaz
if __name__ == "__main__":
    root = tk.Tk()
    app = Puzzle(root)
    root.mainloop()

