import numpy as np
import random
import math
import matplotlib.pyplot as plt
import statistics

BolzmanConst =  1.380649 #e-23

def GetFullEnergy(matrix, J=1):
    H = 0
    for i in range(matrix.shape[0]-1):
        for j in range(matrix.shape[1]-1):
            H += matrix[i][j] + matrix[i+1][j] + matrix[i-1][j] + matrix[i][j+1] + matrix[i][j-1]
    return H*(-J)

def GetMagnetization(matrix):
    M=0
    for row in matrix:
        for col in row:
            M+=col
    return M

def SpinFlip(m, T, J):
    matrix = m.copy()
    matrixnew = m.copy()
    i = random.randint(0, (matrix.shape[0]-1))
    j = random.randint(0, (matrix.shape[0]-1))
    if matrixnew[i][j] == 1:
        matrixnew[i][j] = -1
    else:
        matrixnew[i][j] = 1
    H = GetFullEnergy(matrix,J)
    Hnew = GetFullEnergy(matrixnew,J)
    deltaH = Hnew- H
    if deltaH<=0:
        return matrixnew
    else:
        p = math.exp(-(deltaH)/(BolzmanConst*T))
        threshold = np.random.uniform(0,1,1)[0]
        if p > threshold:
            return matrixnew
        else:
            return matrix

def IzingModel(T, J, n, isPlot = False):
    matrix =  np.random.randint(0,2,(10,10))
    matrix = np.where(matrix == 0, -1, matrix)
    Magnetization = []
    Energy = []
    if isPlot:
        print(f"Начальная матрица T = {T}")
        print(matrix)
    for i in range(n):
        matrix = SpinFlip(matrix,T,J).copy()
        Energy.append(GetFullEnergy(matrix, J))
        Magnetization.append(GetMagnetization(matrix))
    Hmean = statistics.mean(Energy[round(n*0.5):])
    Mmean = statistics.mean(Magnetization[round(n*0.5):])
    if isPlot:
        print(f"Конечная матрица T = {T}")
        print(matrix)
        plt.plot(range(n), Energy, label=f'T = {T} Энергия mean = {Hmean}')
        plt.plot(range(n), Magnetization, label=f'T = {T} Намагниченность mean = {Mmean}')
        plt.legend()
        plt.show()
    return (Hmean, Mmean)
        

def main():
    J=1
    steps = [1,2,3,5,10,20,30,40,50,80,100]
    HmeanArr = []
    MmeanArr = []
    for i in range(1,101):
        if i in steps:
            (Hmean, Mmean) = IzingModel(i, J, 1000, True)
        else:
            (Hmean, Mmean) = IzingModel(i, J, 1000)
        HmeanArr.append(Hmean)
        MmeanArr.append(Mmean)
    plt.plot(range(1,101), HmeanArr, label=f'Энергия')
    plt.legend()
    plt.figure()
    plt.plot(range(1,101), MmeanArr, label=f'Намагниченность')
    plt.legend()
    counts, weights = np.histogram(MmeanArr)
    plt.figure()
    plt.stairs(counts, weights, label=f'Намагниченность')
    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    main()