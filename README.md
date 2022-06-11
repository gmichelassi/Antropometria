![image](antropometria/assets/header.png)

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
1. Certifique-se que está usando a versão 3.6 ou superior do Python, preferencialmente em um sistema operacional UNIX;
2. Crie um ambiente virtual -venv- (veja abaixo como fazer isso) e clone este repositório dentro do seu venv;
3. Dentro de seu ambiente virtual, execute o comando "pip install -r requirements.txt" para instalar todas as dependencias necessárias;

# Formato dos dados
Todos os dados devem ser incluídos dentro da pasta data/ e devem estar organizados de uma das formas a seguir

1. Um arquivo .csv para cada classe
2. Um único arquivo .csv para todas as classes com uma coluna para as labels chamada 'class_label'

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
Para executar o pipeline é preciso realizar dois passos.

##### Configurar os parâmetros de testes
Em `antropometria/classifiers/` existe um arquivo para cada classificador implementado. É possível (e recomendado) alterar os parâmetros a serem calibrados para que se adequem ao problema que está sendo estudado.

Em `antropometria/config/constants/training/` estão definidas as configurações de testes que serão executados durante os pipelines.

##### Executar
Na raiz do projeto `python3 app.py`.


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
