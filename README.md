# AiFinalWork

Esse foi um projeto feito com o intuito de servir como trabalho final da matéria de Inteligência Artificial. Esse Agente de IA foi criada com o intuito de servir como uma secretária para uma clínica, que tem como objetivo cadrastrar clientes em uma planilha e criar agendamentos destes clientes no google Calendar.


## Como executar o projeto localmente

Primeiro tenha o Python instalado na sua máquina.

Clone o projeto

```bash
  git clone https://github.com/Justanothervitor/AiFinalWork
```

Vá ao diretório do projeto.

```bash
  cd AiFinalWork
```

Instale as dependências atráves dos scripts.

Se tiver usando o Windows use:
```bash
./install_virtual_environment.bat
```
Se tiver usando o Linux use:
```bash
.install_virtual_environment.zsh
```

Os scripts acima vão baixar as dependências do projeto e vão criar a pasta de ambiente virtual do Python.
Após a execução do script você precisará criar um arquivo .env com uma variável com a chave de API da Open Ai, outra com um id da planilha do Google Planilhas e um variável com o nome da planilha. Você irá precisar de um arquivo json com as credenciais do google. Use o tutorial a seguir para poder adquirir ele:https://www.youtube.com/watch?v=wBXUuWNFOu4
Tendo feito isso, renomeie o arquivo .json para credentials.json e coloque-o na pasta. Após isso execute.

```bash
  python main.py
```

Se quiser fazer um deploy usando o ngrok, abra o arquivo ngrok.yml e coloque a sua authKey nele, e execute o exec-infra.bash.
