# Boxing with Reinforcement Learning
Trabalho desenvolvido para a disciplina de Reinforcement Learning 2023.2 da FGV EMAp.


## O jogo
O jogo faz parte dos ambientes Atari disponibilizados no Gymnasium, mas originalmente foi lançado pela Activision para o Atari 2600. O objetivo é atingir o oponente até completar 100 pontos (ou estar com a maior pontuação no final do round, que dura dois minutos) em um espaço limitado pelo ringue de boxe.

<div align="center">
	<img src = "img/Boxing_Coverart.png" width=25%> 
</div>

No desenvolvimento do trabalho, buscamos treinar o jogador preto utilizando aprendizado por reforço para que ele passe a vencer o jogador branco. 

## Ambiente
### Espaço de observação

Definido no Gymnasium como `Box(0, 255, (210, 160, 3), uint8)`
O espaço de observação é uma matriz 210x160x3, em que o terceiro canal define o canal RGB da imagem. Na prática, nem todo pixel da matriz é utilizável, porque o ringue de luta é limitado, então realizamos um crop removendo os 30 pixels mais externos de cada margem.  

<div align="center">
	<img src = "img/boxing.jpg" width=30%> 
</div>

### Espaço de ações

Tratamos de um espaço discreto contendo 18 possíveis ações (todas as possíveis dos ambientes Atari). É possível não realizar nenhuma ação (0), deferir o soco (1), se mover em alguma direção (2-9) ou se mover em alguma direção enquanto deferimos um soco (10-17). 

<div align="center">
	<img src = "img/action.png" width=50%> 
</div>

### Recompensas
Recompensamos o agente (jogador preto) utilizando +1 ponto quando um soco deferido atinge o oponente, e -1 quando o agente é atingido pelo oponente.

## Nossa abordagem
Buscando observar aprendizado por parte do agente em sua performance contra o adversário, decidimos utilizar a abordagem do Deep Q-learning, isto é, utilizar a combinação de aprendizado por reforço e aprendizado profundo produzindo uma NN que busque um valor de Optimal State-Action para o agente, isto é, aprendemos a matriz Q. Utilizamos diferentes modelos de aprendizado profundo para testar e comparar desempenhos.

Para todos os modelos, utilizamos:
* _batch size_: 64
* otimizador: Adam
* _loss function_: Mean Squared Error
* _epochs_: 10, 100, 500
* _steps per epoch_: 200

### CNN Simples
A estrutura que determinamos para essa rede foi a seguinte:

```
CONV2D(32, activation=relu)
BATCH NORMALIZATION
CONV2D(32, activation=relu)
FLATTEN 
DENSE(64, activation=relu)
DENSE(18, activation=linear)
```

Para esse modelo, definimos _inputs_ simples, que representam uma única imagem, e utilizamos _samples_ aleatórias de um conjunto de dados gerado pré-treinamento. Assim sendo, não há uma noção de "movimentação" ou sequência/temporalidade para este modelo.

<div align="center">
	<img src = "img/cnn1.png" width=40%> 
</div>

Sua performance 

### CNN Simples com Stack de imagens

Para esse modelo, utilizamos a mesma estrutura de CNN que a anterior, mas realizamos um stack de 5 imagens para cada _input_ da rede. A quantidade de parâmetros treináveis aumenta pouco em relação à rede anterior, mas o objetivo era ter inputs mais significativos, com uma noção de "movimento" do agente dentro de cada input, já que capturamos uma janela de 5 possibilidades de ação. 
