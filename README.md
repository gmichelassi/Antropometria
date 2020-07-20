# Projeto de Iniciação Científica
##### Escola de Artes, Ciências e Humanidades da Universidade de São Paulo (EACH-USP)

Este projeto tem como objetivo cumprir as seguintes tarefas principais:
~~~
1. Carregar um conjunto de dados qualquer;
2. Realizar o balanceamento das classes por superamostragem (SMOTE), subamostragem (aleatoria) e mista (Tomek);
3. Realizar a redução de dimensionalidade/seleção de característiscas do conjunto carregado;
4. Realizar a calibração dos parâmetros para os classificadores desejados;
5. Salvar o modelo de Machine Learning obtido para que possa ser executado em um projeto futuro;
~~~
Além dessas tarefas principais, este trabalho também lida com as seguintes tarefas secundárias:
~~~
- Carregar conjuntos de dados extremamente grandes;
- Executar individualmente cada classificador, utilizando uma combinação de parâmetros qualquer;
- Gerar dados importantes para a avaliação de um classificador, como matrizes de confusão, _scores_ específicos;
- A partir do dataset principal utilizado para o projeto, pode ser gerado um conjunto de dados que utiliza as razões entre as distâncias;
- Processar este dataset para executá-lo de forma mais fluída.
~~~


#### Como executar?

##### 1. Calibração de Parâmetros
~~~
1. O primeiro passo é certificar-se que está usando a versão 3.6 ou superior do Python, preferencialmente em um sistema operacional UNIX;
2. Crie um ambiente virtual -venv- (veja aqui como fazer isso) e clone este repositório dentro do seu venv;
3. Dentro de seu ambiente virtual, execute o comando "pip install -r requirements.txt" para instalar todas as dependencias necessárias;
4. Feito isso, caso deseje incluir novos conjuntos de dados, eles devem ser incluídos em um diretório dentro da pasta "data/";
5. Caso você julgue que não seja necessário realizar qualquer modificação nos códigos OU não saiba o que modificar, o único arquivo que deve ser editado é o "mainParameterCalibration.py";
5.1 Neste arquivo, no fim do código, realize uma chamada ao método run_gridSearch(): os parâmetros passados devem ser o subdiretório de data/ e o nome do seu arquivo .csv que contém seu conjunto de dados;
5.1.1 Uma restrição deste projeto no momento se dá que para novos conjuntos de dados, todos os dados tem que estar em um arquivo .csv só com uma coluna denominada 'labels' para definir as classes de cada instância;
5.2 Agora, a única coisa a se definir deve ser quais testes serão executados:
5.2.1 No mesmo arquivo, no primeiro método são definidos 4 vetores e 1 dicionário... cada elemento dessas coleções indica um atributo do teste a ser executado, modifique-os conforme sua necessidade;
5.2.2 Caso não saiba como modificar, deixe somente os vetores de classificadores e redução de dimensionalidades preenchidos
6. Finalmente, execute o pipeline digitando "python mainParameterCalibration.py" em seu terminal
~~~

##### 2. Salvar o melhor modelo definido via calibração de parâmetros
~~~
~~~