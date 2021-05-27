![image](./config/header.png)

[![image](https://img.shields.io/badge/gabrielmichelassi@usp.br-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:gabrielmichelassi@usp.br) \
[![image](https://img.shields.io/badge/GabrielMichelassi-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/gabrielmichelassi/)

### Este projeto implementa as seguintes tarefas:
~~~
1. Carregar um conjunto de dados qualquer
2. Realizar o pré processamento dos dados carregados
    2.1. Remover características altamente correlacionadas dos dados (filtro de correlação de Pearson)
    2.2. Aplicar normalização dos valores via re-escala linear (normalização min_max)
    2.3. Aplicar redução de dimensionalidade/seleção de características
    2.4. Aplicar balanceamento das classes por superamostragem (SMOTE), subamostragem (aleatoria) e mista (Tomek)
3. Realizar a calibração dos parâmetros para os classificadores disponíveis (RF, SVM, KNN, NB e Redes Neurais)
4. Salvar o modelo de Machine Learning obtido para que possa ser re-utilizado em um projeto futuro
~~~

### Preparativos
~~~
# Instalação do repositório
1. Certifique-se que está usando a versão 3.6 ou superior do Python, preferencialmente em um sistema operacional UNIX
2. Crie um ambiente virtual -venv- (veja abaixo como fazer isso) e clone este repositório dentro do seu venv
3. Dentro de seu ambiente virtual, execute o comando "pip install -r requirements.txt" para instalar todas as dependencias necessárias;

# Formato dos dados
Todos os dados devem ser incluídos dentro da pasta data/ e devem estar organizados de uma das formas a seguir

1. Um arquivo .csv para cada classe
2. Um único arquivo .csv para todas as classes com uma coluna para as labels chamada 'label'

Abaixo está descrito como devem ser organizados os arquivos

data/
 |
 |____ exemplo1
 |      |____ A_mydata.csv
 |      |____ B_mydata.csv
 |
 |____ exemplo2
        |____ mydata.csv
~~~
Veja [aqui](https://www.treinaweb.com.br/blog/criando-ambientes-virtuais-para-projetos-python-com-o-virtualenv/) sobre criação de ambientes virtuais para Python.

### Como executar?

##### 1. Calibração de Parâmetros
~~~
A calibração de parâmetros é feita em várias etapas e vários processos podem ou não ser aplicados, para controlar quais processos serão aplicados:
1. No arquivo mainParameterCalibration.py localize a função testToRun()
2. Os vetores da linha 27 até a linha 31 indicam quais algoritmos serão aplicados durante o processo de calibração
3. Remova ou insira valores nos vetores para controlar quais testes serão executados

Para indicar em qual conjunto de dados será realizada a calibração, vá no arquivo mainParameterCalibration.py e defina os argumentos _folder_, _dataset name_ e _classes_
O argumento classes deve ser um vetor no qual em cada posição há uma string com o nome da classe OU deve ser None, indicando que todas as classes estão em um mesmo arquivo

Finalmente, execute o pipeline digitando "python mainParameterCalibration.py" em seu terminal
A calibração de parâmetros irá gerar um grande volume de arquivos na pasta output, sendo o primeiro deles um arquivo .csv contendo o melhor resultado para cada teste realizado
~~~

##### 2. Salvar o melhor modelo definido via calibração de parâmetros
~~~
Tendo em mãos o arquivo com os resultados, localize o melhor resultado (aconselha-se usar algum software auxiliar para essa tarefa, como por exemplo o Excel)
Sabendo o melhor resultado, no arquivo SaveModels.py, encontre o dicionário relativo ao classificador e altere os parâmetros conforme a combinação que gerou os melhores resultados
Finalmente, vá até a linha 110 do arquivo SaveModels.py e indique no último parâmetro, qual dicionário deverá ser utilizado

Se tudo ocorrer bem, um arquivo chamado output.sav será salvo na pasta de outputs!
~~~

### Para usuários

1. Não altere os códigos caso não tenha conhecimento de Machine Learning, eles já estão com uma configuração padrão para rodarem sem problemas.
2. O arquivo mainSplitDataFrame.py foi criado única e exclusivamente para lidar com um caso específico que ocorreu durante a Iniciação Científica.

Caso queira aprender melhor sobre machine learning ou este projeto, consulte:
- o relatório dessa IC (disponível por e-mail)
- a ementa e a bibliografia das disciplinas [Inteligência Artificial - ACH2016](https://uspdigital.usp.br/jupiterweb/obterDisciplina?sgldis=ACH2016) e [Reconhecimento de Padrões - SIN5007](https://uspdigital.usp.br/janus/componente/disciplinasOferecidasInicial.jsf?action=3&sgldis=SIN5007)


### Para desenvolvedores

Sinta-se livre para criar uma `branch` deste projeto, `pull requests` serão cuidadosamente avaliados.

As nomenclaturas devem seguir as seguintes instrucões:

Classes, métodos e variáveis
- Nomes de classe devem estar em _camel case_. Ex: `ClassExample`;
- Nome de métodos devem estar em _lower case_ e as palavras separadas por _. Ex: `call_method()`;
- Nomes de variaveis devem estar em _lower case_ e as palavras separadas por _. Ex: `num_of_features = 10`;
- Nomes de constantes devem estar em _upper case_ e as palavras separadas por _. Ex: `CONSTANT = 10`;
  
Arquivos e diretórios
- Arquivos de classe devem estar em _camel case_;
- Arquivos no diretório do projeto devem iniciar com _main_ seguido do seu propósito. Ex: `mainParameterCalibration.py`;  
- Arquivos `.py` que não estão nas categorias listadas devem estar em _lower case_;
- Diretorios de projeto devem estar em _lower case_

Commits
- Durante seus _commits_, as mensagens devem ser claras e explicativas do que foi alterado.

Testes
- ...

OBS: O arquivo `mainSplitDataFrame.py` não tende a seguir a organização do resto do projeto por realizar uma tarefa muito específica e não comum a maioria dos demais conjuntos de dados;

Deve-se ter em mente que tudo que for escrito deve estar claro em seu propósito.
Comentários devem ser evitados.
