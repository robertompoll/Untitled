# Regressão Linear

## Sobre os exercícios
O exercício proposto supõe que somos um residente da cidade Ficção e queremos vender nossa casa, mas não sabemos exatamente o preço a ser escolhido. Somos apresentados uma tabela com a área, número de quartos e o preço de uma gama de casas na cidade. Iremos treinar o algoritmo para identificar a correlação entre essas informações.

```python
import pandas as pd
pathtodata = 'Exercise_Data/ex2_Data.txt'
data = pd.read_csv(pathtodata,delimiter = ',',header=None)
x = data[[0,1]] # Área da Casa (X0) e número de quartos (X1)
y = data[[2]] # Preço das casas
```
## Função Custo
A função custo **(J)** ou função de perda representa a distância entre a hipótese linear e os dados reais. Portanto, quanto maior a função custo, menor é a correlação entre a hipótese e os dados. A função custo é representada por **equation (1)**:

<html>
<p>
<body>
    <div align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=$$J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2$$" title="$$J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2$$"/></a> (1) 
    </div>
</body>
</html>

O sobrescrito _(i)_ representa o número da linha dos dados. Enquanto a hipótese<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;$h_\theta$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;$h_\theta$" title="$h_\theta$" /></a> é dada como a equação linear (2):

<html>
<p>
<body>
    <div align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;$$h_\theta=\theta^T&space;x&space;=&space;\theta_0&space;x_0&plus;&space;\theta_n&space;x_n$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;$$h_\theta=\theta^T&space;x&space;=&space;\theta_0&space;x_0&plus;&space;\theta_n&space;x_n$$" title="$$h_\theta=\theta^T x = \theta_0 x_0+ \theta_n x_n$$" /></a> (2)
    </div>
</body>
</html>

O arquivo _**computeCost**_ demonstra a implementação do passo-a-passo dessas equações.

## Gradiente
O gradiente é um dos métodos utilizados para otimizar a hipótese e reduzir a função custo. O programa escolherá a melhor hipótese através da derivação da função custo. **Figure** 1 mostra o valor da função custo de acordo com os valores do parâmetros <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\theta_0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta_0" title="\theta_0" /></a> e <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\theta_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta_1" title="\theta_1" /></a>. **Equation (3)** mostra como a otimização dos parâmetros aconteceu. O arquivo _**gradientDescent**_ mostra como ocorreu o método.

<html>
<p>
<body>
    <div align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\theta_j:=\theta_j&space;&minus;\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta_j:=\theta_j&space;&minus;\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}" title="\theta_j:=\theta_j -\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}" /></a> (3)
    </div>
</body>
</html>

<html>
<p>
<body>
    <div align="center">
        <img src="Image/Cost_Function.png" 
        width = "350" /><p>
    </div>
</body>
<body>
    <div align="center">
   Figure 1 - Função custo para diferentes valores de Theta 0 e Theta 1<p>
    </div>
</body>
</html>

O subescrito _(j)_ é o número da coluna

## Normalização
A normalização dos dados é importante, não só para a visualização como para reduzir as interações quando aplicado o método gradiente. Ou seja, convergência dos dados em valores melhores de <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\theta_0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta_0" title="\theta_0" /></a> <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\theta_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta_1" title="\theta_1" /></a>...<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\theta_n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta_n" title="\theta_n" /></a>. A equação de normalização (equation (4)) é:

<html>
<p>
<body>
    <div align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x^{norm}_j&space;=\frac{x_j^{(i)}-\mu^{(i)}}{S^{}(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;x^{norm}_j&space;=\frac{x_j^{(i)}-\mu^{(i)}}{S^{}(i)}" title="x^{norm}_j =\frac{x_j^{(i)}-\mu^{(i)}}{S^{}(i)}" /></a> (4)
    </div>
</body>
</html>

onde <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mu^{(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\mu^{(i)}" title="\mu^{(i)}" /></a> é o meio da coluna _(i)_ e <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;S^{(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;S^{(i)}" title="S^{(i)}" /></a> é a diferença entre os valores mínimo e máximo das colunas _(i)_. O arquivo _**featureNormalize**_ mostra o desenvolvimento do processo.

## A taxa de aprendizado <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\alpha" title="\alpha" /></a>
O parâmetro <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\alpha" title="\alpha" /></a> dentro do gradiente de regressão é importante para ajustar a velocidade da convergência da função custo. Um valor errado de <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\alpha" title="\alpha" /></a> pode impedir a mesma de escolhas exatas. **Figure 2** mostra a função custo com diferentes valores de <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\alpha" title="\alpha" /></a>.

<html>
<p>
<body>
    <div align="center">
        <img src="Image/Learn_Curve_0.001.png" width = 255/><img src="Image/Learn_Curve_0.01.png" width = 250/><img src="Image/Learn_Curve_1.32.png" width = 250/>
    </div>
</body>
<body>
    <div align="center">
   Figure 2 - Curva de Aprendizado de diferentes valores de alfa<p>
    </div>
</body>
</html>

- Se <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\alpha" title="\alpha" /></a> for pequeno: velocidade de convergência baixa
- Se <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\alpha" title="\alpha" /></a> for grande: Aumento da função custo ou sem convergência

## Usando a biblioteca scikit
Iremos importar os seguintes:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
```

Com o _**preprocessamento**_ feito acima:

```python
x_norm = pd.DataFrame(preprocessing.scale(x))
```
Iremos usar a função _**train_test_split**_, onde dividiremos os dados em treino e teste. O parâmetro _**test_size**_ seleciona o tamanho dos dados a serem testados (0.3 ou 30%).

```python
[X_train, X_test, y_train, y_test] = train_test_split(x_norm,y,test_size=0.3,
random_state=101)
```
Usando os dados de treino, aplicamos a regressão linear e verificamos os valores <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\theta_0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta_0" title="\theta_0" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\theta_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta_1" title="\theta_1" /></a> e <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\theta_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta_2" title="\theta_2" /></a>.

```python
# dados treino
lm.fit(X_train,y_train)

# Processando os parâmetros (theta_0, theta_1 and theta_2)
theta0 = pd.DataFrame(lm.intercept_)
theta = pd.DataFrame(lm.coef_)
```
