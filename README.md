![image](antropometria/assets/header.png)

[![image](https://img.shields.io/badge/gabrielmichelassi@usp.br-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:gabrielmichelassi@usp.br) \
[![image](https://img.shields.io/badge/GabrielMichelassi-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/gabrielmichelassi/)

Esse projeto tem o intuito de suportar todas as etapas de desenvolvimento de um classificador binário ou multiclasse para um determinado problema.
- Pré processamento (aplicação de filtros, transformações nos dados, balanceamento de classes, etc);
- Redução de Dimensionalidade (ou seleção de características);
- Hyperparameter Tuning e Grid Search;
- Avaliação das métricas do classificador;
- Avaliação estatística dentre todos testes realizados.

### Como utilizar?
~~~
# Instalação do repositório
1. Certifique-se que está usando a versão 3.9 ou superior do Python, preferencialmente em um sistema operacional UNIX;
2. Crie um ambiente virtual -venv- (veja abaixo como fazer isso) e clone este repositório dentro do seu venv;
3. Dentro de seu ambiente virtual vá até a raiz do projeto execute o comando "pip install -r requirements.txt" para instalar todas as dependencias necessárias;

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

##### Configurar o teste
Em `antropometria/classifiers/` existe um arquivo para cada classificador implementado. É possível (e recomendado) alterar os parâmetros a serem calibrados para que se adequem ao problema que está sendo estudado.

Em `antropometria/config/training_parameters.py` estão definidas as configurações de testes que serão executados durante os pipelines.

##### Executar
Na raiz do projeto execute `python3 app.py`.
- Não se esqueça de alterar o arquivo `app.py` para que seus dados sejam carregados e os testes sejam executados.
```
DEFAULT_VALUES = [
    ('exemplo1', 'mydata', ['A', 'B']), # 'A' e 'B' são as classes que serão carregadas e cada uma está em um arquivo .csv separado
    ('exemplo2', 'mydata', None) # todos os dados estão em um mesmo arquivo .csv
]
```


##### Executar
Outros módulos podem ser invocados separadamente, basta fazer a chamada deles no arquivo `main.py`. Um exemplo, para plotar um gráfico de barras com barras de erro faça:

```
from antropometria.plots import bar_plots_with_error_bars

bar_plots_with_error_bars(x=x, y=y, errors=errors, output_file='./output.csv')
```

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
- Arquivos `.py` que não estão nas categorias listadas devem estar em _lower case_;
- Diretorios de projeto devem estar em _lower case_ e seguir o padrão de módulos em python (com os arquivos que devem ser expostos pra fora declarados no `__init__.py`).

Commits
- Durante seus _commits_, as mensagens devem ser claras e explicativas do que foi alterado.
- Utilize os prefixos `feat: | refactor: | chore: | fix:` e outros para deixar mais claro a intençao do seu _commit_.

### Testes, qualidade de código e CI/CD

Os seguintes comandos são obrigatórios antes de abrir um _pull request_:
- `pytest` para rodar os testes unitários;
- `propector` para verificar a qualidade do código;
- `isort .` para organizar os imports;

Esses comandos são verificados ao abrir um PR, e o _pipeline_ de CI/CD só é aprovado se todos os testes passarem.

