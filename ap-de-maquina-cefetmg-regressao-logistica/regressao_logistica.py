from abc import abstractmethod
import numpy as np
import math
class Gradiente():
    def __init__(self,arr_dz,arr_dw,db):
        self.arr_dz = arr_dz
        self.arr_dw = arr_dw
        self.db = db

    def __str__(self):
        return "dz: "+str(self.arr_dz)+" db: "+str(self.db)+" arr_dw: "+str(self.arr_dw)

### FUncoes de ativação #########
class FuncaoAtivacao():
    def __init__(self,funcao,dz_funcao):
        self.funcao = funcao
        self.dz_funcao = dz_funcao

# Atividade 1: Crie as funções lambda para instanciar um objeto da classe FunçãoAtivacao. Lembre-se que funcao e dz_função são funções que estão sendo passadas como parametro
funcao = lambda z: 1/np.exp(-z)+1
dz_funcao = lambda a, z, y: a-y
sigmoid = FuncaoAtivacao(funcao,dz_funcao)

class RegressaoLogistica():
    def __init__(self,func_ativacao,dz_func,num_iteracoes=100):
        self.arr_w = None
        self.b = 0
        self.func_ativacao = func_ativacao
        self.dz_func = dz_func
        self.num_iteracoes = num_iteracoes

        #deixar as linhas de baixo como None, nao foi calculado/definido ainda
        self.arr_z = None
        self.arr_a = None
        self.arr_y = None
        self.mat_x = None
        self.gradiente = None

    def __str__(self):
        return "arr_z: "+str(self.arr_z)+\
                "\narr_a:"+str(self.arr_a)+\
                "\ngradiente: "+str(self.gradiente)+\
                "\nmat_x: "+str(object=self.mat_x)
    def z(self,mat_x):
        """
        Atividade 2: Função que retorna os resultados da função z por instancia usando a matriz mat_x
        """
        return self.func_ativacao.dz_funcao(mat_x)

    def forward_propagation(self,mat_x):
        """
        Atividade 3: Implementar o forward propagation

        mat_x: matriz de atributos por instancias tamanho (n,m)
        """
        #print("MAT_X: "+str(mat_x))
        #print("arr_w: "+str(self.arr_w))
        #print("b: "+str(self.b))

        #caso nao esteja definido, inicialize o atributo self.arr_w com zero
        if(self.arr_w is None):
            self.arr_w = None
        #defina o atributo mat_x
        self.mat_x = None

        #faça o calculo do método z e armazene-o em arr_z
        self.arr_z = None

        #calcule a função de ativação (por meio do atributo) e armazene o resultado em arr_a
        self.arr_a = None

        #print("ARR_Z: "+str(self.arr_z))
        #print("ARR_A: "+str(self.arr_a))

        #o arr_a será retornado nessa função
        return self.arr_a

    def backward_propagation(self,arr_y):
        """
        Atividade 4: Implemente a backward propagation
        """
        #numero de instancias
        n_instances = len(arr_y)

        #print("arr_a:"+str(self.arr_a)+" arr_z:"+str(self.arr_z)+" arr_y:"+str(arr_y))
        #print("X: "+str(self.mat_a_ant))

        #calcule dz por meio do atributo representando a função da derivada
        arr_dz = None


        #a partir de arr_dz e mat_x, calcula arr_dw
        arr_dw = None

        #a partir de arr_dz, calcula db
        db = None

        #print("DZ: "+str(arr_dz))
        #print("arr_dw: "+str(arr_dw))
        #print("db: "+str(db))

        #define o gradiente (instancie um objeto da classe Gradiente apropriadamente)
        self.gradiente = None

        return self.gradiente
    def loss_function(self,arr_y):
        """
        Atividade 5: Calcule a loss function usando entropia cruzada (cross entropy)
        """
        return None


    def atualiza_pesos(self,learning_rate):
        """
        Atividade 6: Atualize os pesos arr_w e b por meio do gradiente e o learning_rate (float)
        """
        self.arr_w = None
        self.b = None

    def fit(self,mat_x,arr_y,learning_rate=1.1):
        """
        Atividade 7: Cria o modelo de regressão logistica por meio de num_iteracoes épocas
        imprime, a cada 10 épocas, a loss function obtida
        """
        for i in range(self.num_iteracoes):
            None

            #print("A: "+str(self.arr_a))
            #print("Y:"+str(arr_y))
            if (i%10 == 0):
                print("Iteração: "+str(i)+" Loss: "+str(loss))



        #print("PESOS: "+str(self.arr_w)+" b:"+str(self.b))
        #print("A: "+str(self.arr_a))
        #print("Y:"+str(arr_y))


    def predict(self,mat_x):
        """
        Atividade 8 -usando as instancias mat_x, representando a matriz  𝑋  das instâncias que queremos prever,
        calcula-se o forward_propagation do modelo para, logo após, retornar o vetor de predições
        """

        return None
