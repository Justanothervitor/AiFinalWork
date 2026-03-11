from datetime import datetime

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from sqlalchemy.sql.functions import current_date

import tools
from tools import GoogleAPIManager

load_dotenv()

class AgentConfig:
    
    MODEL_NAME = "gpt-4o-mini"
    TEMPERATURE = 0.5
    SYSTEM_MESSAGE= f"""
               Você é um assistente de reservas de IA útil, com acesso ao Google Agenda e ao Google Sheets.

REGRA IMPORTANTE: 
1.Os usuários DEVEM estar registrados no sistema antes de poderem fazer reservas!
2. Quando os usuários quiserem realizar ações, você DEVE responder APENAS com o formato do comando, nada mais.
3. NÃO PEÇA identificação novamente se já a recebeu.

**AUTH WORKFLOW:**
- Se o usuário mencionar seu nome/e-mail use: CHECK_USER|[nome ou e-mail]
- Se o usuário quiser fazer uma reserva, pergunte o seu nome ou email e espere a resposta do usuário, após o usuário responder procure extrair o nome ou o email e use: CHECK_USER|[nome ou e-mail]
- Após o sucesso do CHECK_USER, o usuário é autenticado para a sessão
- NÃO solicite CHECK_USER novamente na mesma conversa após a autenticação bem-sucedida


**Para verificar o registro do usuário (APENAS UMA VEZ por conversa):**
Quando o usuário quiser fazer uma reserva OU pedir para verificar o registro:
CHECK_USER|[nome ou e-mail]

Exemplo: O usuário diz “Eu sou John” ou “Meu e-mail é john@email.com”.
Resposta: CHECK_USER|john@email.com

Exemplo: O usuário diz “Quero marcar uma consulta”
Resposta: Você poderia me informar o seu nome ou email?

** Para registrar novos usuários:**
Quando o usuário fornecer informações de contato para registro:
SAVE_DATA|[nome]|[e-mail]|[telefone]|[endereço][notas]

Exemplo: “Meu nome é John Doe, e-mail john@email.com, telefone 555-1234”
Resposta: SAVE_DATA|John Doe|john@email.com|555-1234|Rua Apê Aloe Vera,Residencial Paula Vadao|Observações

**Para reservas no calendário (o usuário já está autenticado):**
CALENDAR_BOOKING|[Nome do médico]|[AAAA-MM-DDTHH:MM:SS]|[descrição da consulta]|[e-mail do participante]

Exemplo: Usuário já autenticado diz "Eu quero marcar uma consulta"
Resposta: Me informe o nome do médio, a data e hora no qual você quer ser atendido,me diga se é uma consulta de retorno.

Exemplo: Dr.Hammilton,2024-11-12T14:00:00,É uma consulta de retorno
Resposta: CALENDAR_BOOKING|Dr.Hammilton|2024-11-12T14:00:00||É uma consulta de retorno|

IMPORTANTE: Se o usuário já tiver sido verificado/registrado nesta conversa, prossiga DIRETAMENTE para a criação de reservas sem verificar novamente!

**Para listar eventos:**
LIST_EVENTS|[número máximo]

Exemplo: “Tenho alguma consulta marcada?”
Resposta: LIST_EVENTS|10

**Para conversas gerais:**
Responda normalmente e de forma prestativa.

IMPORTANTE: 
- CHECK_USER apenas UMA VEZ quando ele se identificar pela primeira vez
- Após a autenticação/registro bem-sucedido, prossiga diretamente com as ações solicitadas
- Nunca crie registros duplicados - o sistema verifica automaticamente
- Para comandos de ação, exiba APENAS a linha de comando, sem explicações
- Sempre use o formato ISO de data e hora: AAAA-MM-DDTHH:MM:SS
- Calcule datas relativas a hoje: {current_date}
- A duração padrão é de 60 minutos, se não for especificada
- Seja amigável e oriente os usuários durante o processo de registro, se necessário
- Caso seja perguntado algo que não tenha relação á marcar as consultas, redirecione o cliente para outros departamentos.
    """

class ChatAgent:

    def __init__(self):

        try:
            self.google_api = GoogleAPIManager()
            self.calendar_tool = tools.GoogleCalendarTool(self.google_api)
            self.sheets_tool = tools.GoogleSheetsTool(self.google_api)
            self.tools_available = True

        except Exception as e:
            print(f"Atenção a Api do google não foi inicializada, logo não vai ser possível fazer agendamentos: {str(e)}")
            self.calendar_tool = None
            self.sheets_tool = None
            self.tools_available = False

        self.llm = self._initialize_llm()
        self.chain = self._create_chain()
        self.chat_histories = {}
        self.chain_with_history = self._create_chain_with_history()
        self.session_to_user = {}

    def _initialize_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=AgentConfig.MODEL_NAME,
            temperature = AgentConfig.TEMPERATURE
        )

    def _process_response(self, response: str, gradio_session_id: str = "default") -> str:
        """Process the AI response and execute actions if needed."""
        if not self.tools_available:
            return response

        # Clean up the response
        response = response.strip()

        # Debug: print what we received
        print(f"\n{'=' * 60}")
        print(f"DEBUG - Raw LLM Response:")
        print(response)
        print(f"DEBUG - Gradio Session: {gradio_session_id}")
        print(f"{'=' * 60}\n")

        # Check for action commands (handle both with and without extra text)
        if "CHECK_USER|" in response:
            print("→ Detected CHECK_USER command")
            return self._handle_check_user(response, gradio_session_id)
        elif "CALENDAR_BOOKING|" in response:
            print("→ Detected CALENDAR_BOOKING command")
            return self._handle_calendar_booking(response, gradio_session_id)
        elif "LIST_EVENTS|" in response or "LIST_EVENTS" in response:
            print("→ Detected LIST_EVENTS command")
            return self._handle_list_events(response, gradio_session_id)
        elif "SAVE_DATA|" in response:
            print("→ Detected SAVE_DATA command")
            return self._handle_save_data(response, gradio_session_id)
        else:
            print("→ No command detected, returning as normal conversation")
            return response

    def _handle_calendar_booking(self, response: str, gradio_session_id: str = "default") -> str:
        """Handle calendar booking action - only if user is authenticated."""
        try:
            # Check if user is authenticated for this gradio session
            if gradio_session_id not in self.session_to_user:
                return ("🚫 **Você precisa se identificar primeiro!**\n\n"
                        "Para fazer agendamentos, forneça seu nome ou email cadastrado.\n\n"
                        "Exemplo: *Meu email é joao@email.com*")

            user_email = self.session_to_user[gradio_session_id]

            # Fetch user data from spreadsheet (sempre busca dados atuais)
            user_result = self.sheets_tool.check_user_exists(user_email)

            if not user_result['success'] or not user_result['exists']:
                return ("❌ **Erro ao verificar seu cadastro.**\n\n"
                        "Por favor, identifique-se novamente.")

            user_data = user_result['user_data']

            # Extract the command line from the response
            command_line = None

            if 'CALENDAR_BOOKING|' in response:
                for line in response.split('\n'):
                    if 'CALENDAR_BOOKING|' in line:
                        command_line = line.strip()
                        break

                if not command_line:
                    command_line = response.strip()

            if not command_line:
                return "❌ Não consegui processar o comando de agendamento. Tente novamente com: nome do evento, data/hora, duração."

            print(f"Command line: {command_line}")

            if 'CALENDAR_BOOKING|' in command_line:
                command_data = command_line.split('CALENDAR_BOOKING|', 1)[1]
            else:
                return "❌ Formato de comando inválido."

            parts = command_data.split('|')
            print(f"Parsed parts: {parts}")

            if len(parts) < 3:
                return "❌ Preciso de: nome do Doutor, data/hora (YYYY-MM-DDTHH:MM:SS)."

            summary = parts[0].strip()
            start_time = parts[1].strip()
            duration_str = parts[2].strip()
            duration = int(duration_str) if duration_str and duration_str.isdigit() else 60
            description = parts[3].strip() if len(parts) > 3 else ""
            attendee = parts[4].strip() if len(parts) > 4 and parts[4].strip() else None

            # Add user info to description
            if description:
                description += f" | Cliente: {user_data['name']} ({user_data['email']})"
            else:
                description = f"Cliente: {user_data['name']} ({user_data['email']})"

            print(f"Creating booking for {user_data['name']}: {summary} at {start_time} for {duration} min")

            result = self.calendar_tool.create_booking(
                summary=summary,
                start_time=start_time,
                duration_minutes=duration,
                description=description,
                attendee_email=attendee or user_data['email']
            )

            if result['success']:
                return (f"✅ **Agendamento criado com sucesso!**\n\n"
                        f"📅 {summary}\n"
                        f"🕐 {start_time}\n"
                        f"⏱️ Duração: {duration} minutos\n"
                        f"👤 Cliente: {user_data['name']}\n\n"
                        f"🔗 Ver consulta: {result['link']}")
            else:
                return f"❌ {result['message']}"

        except Exception as e:
            error_msg = f"❌ Erro ao criar agendamento: {str(e)}"
            print(error_msg)
            return error_msg

    def _handle_list_events(self, response: str, gradio_session_id: str = "default") -> str:
        """Handle list events action."""
        try:
            # Check if user is authenticated
            if gradio_session_id not in self.session_to_user:
                return ("🚫 **Você precisa se identificar primeiro!**\n\n"
                        "Para ver as consultas marcadas, forneça seu nome ou email cadastrado.")

            user_email = self.session_to_user[gradio_session_id]

            # Fetch user data
            user_result = self.sheets_tool.check_user_exists(user_email)
            if user_result['success'] and user_result['exists']:
                user_name = user_result['user_data']['name']
            else:
                user_name = "Usuário"

            # Extract the command line
            lines = response.split('\n')
            command_line = None
            for line in lines:
                if 'LIST_EVENTS|' in line:
                    command_line = line.strip()
                    break

            if not command_line:
                max_results = 10
            else:
                command_line = command_line.split('LIST_EVENTS|')[1]
                parts = command_line.split('|')
                max_results = int(parts[0].strip()) if parts and parts[0].strip() else 10

            result = self.calendar_tool.list_upcoming_events(max_results)

            if result['success']:
                if result['count'] == 0:
                    return f"📅 {user_name}, você não tem consultas próximas na agenda."

                events_text = f"📅 **{user_name}, suas próximas consultas:**\n\n"
                for i, event in enumerate(result['events'], 1):
                    events_text += f"{i}. **{event['summary']}** - {event['start']}\n"

                return events_text
            else:
                return f"❌ Não foi possível listar as consultas: {result.get('error', 'Erro desconhecido')}"

        except Exception as e:
            return f"❌ Erro ao listar consultas: {str(e)}"

    def _handle_check_user(self, response: str, gradio_session_id: str = "default") -> str:
        try:
            # Extract identifier from command
            command_line = None
            if 'CHECK_USER|' in response:
                for line in response.split('\n'):
                    if 'CHECK_USER|' in line:
                        command_line = line.strip()
                        break

                if not command_line:
                    command_line = response.strip()

            if not command_line or 'CHECK_USER|' not in command_line:
                return ("Para usar o sistema, preciso identificar você.\n\n"
                        "**Por favor, forneça:** seu nome completo OU email\n\n"
                        "Exemplo: *Meu email é joao@email.com*")

            identifier = command_line.split('CHECK_USER|', 1)[1].strip()

            if not identifier or len(identifier) < 2:
                return ("Para usar o sistema, preciso identificar você.\n\n"
                        "**Por favor, forneça:** seu nome completo OU email\n\n"
                        "Exemplo: *Meu nome é João Silva*")

            print(f"Checking user: '{identifier}'")

            # Check in spreadsheet
            result = self.sheets_tool.check_user_exists(identifier)

            if result['success'] and result['exists']:
                # User found - map gradio session to user email for history
                user_data = result['user_data']
                user_email = user_data['email']

                # Store mapping for this gradio session
                self.session_to_user[gradio_session_id] = user_email

                print(f"✓ User verified: {user_data['name']} ({user_email})")
                print(f"✓ Session mapping: {gradio_session_id} -> {user_email}")

                return (f"✅ **Bem-vindo(a), {user_data['name']}!**\n\n"
                        f"📧 Email: {user_data['email']}\n"
                        f"📱 Telefone: {user_data.get('phone', 'Não informado')}\n\n"
                        f"✨ Você está autenticado e pode fazer agendamentos agora! 📅")
            else:
                # User not found - explain registration process
                return (f"🔍 **Usuário não encontrado.**\n\n"
                        f"Você ainda não está cadastrado no sistema.\n\n"
                        f"**Para se cadastrar, forneça:**\n"
                        f"✓ Nome completo\n"
                        f"✓ Email\n"
                        f"✓ Telefone (opcional)\n\n"
                        f"✓ Endereço\n"
                        f"Exemplo: *Meu nome é João Silva, email joao@email.com, telefone 11-99999-8888, Rua do Aloe Vera Residencial Paula Vadao*")

        except Exception as e:
            error_msg = f"❌ Erro ao verificar usuário: {str(e)}"
            print(error_msg)
            return error_msg

    def _handle_save_data(self, response: str, gradio_session_id: str = "default") -> str:
        """Handle save user data action - with duplicate check."""
        try:
            # Extract the command line
            command_line = None

            if 'SAVE_DATA|' in response:
                for line in response.split('\n'):
                    if 'SAVE_DATA|' in line:
                        command_line = line.strip()
                        break

                if not command_line:
                    command_line = response.strip()

            if not command_line:
                return "❌ Preciso de pelo menos nome e email para cadastrar."

            print(f"Command line: {command_line}")

            if 'SAVE_DATA|' in command_line:
                command_data = command_line.split('SAVE_DATA|', 1)[1]
            else:
                return "❌ Formato de comando inválido."

            parts = command_data.split('|')
            print(f"Parsed parts: {parts}")

            if len(parts) < 2:
                return "❌ Preciso de pelo menos nome e email para cadastrar."

            name = parts[0].strip()
            email = parts[1].strip()
            phone = parts[2].strip()
            address = parts[3].strip()
            notes = parts[4].strip() if len(parts) > 4 else ""

            print(f"Attempting to save: {name}, {email}")

            # IMPORTANT: Check if user already exists before saving
            print(f"Checking if user already exists...")
            check_result = self.sheets_tool.check_user_exists(email)

            if check_result['success'] and check_result['exists']:
                # User already exists!
                existing_user = check_result['user_data']

                # Authenticate this gradio session with existing user
                self.session_to_user[gradio_session_id] = existing_user['email']

                print(f"✓ User already registered, authenticated session")

                return (f"ℹ️ **Este email já está cadastrado!**\n\n"
                        f"👤 Nome: {existing_user['name']}\n"
                        f"📧 Email: {existing_user['email']}\n"
                        f"📱 Telefone: {existing_user.get('phone', 'Não informado')}\n\n"
                        f"✅ Você foi autenticado e pode fazer agendamentos!")

            # User doesn't exist, proceed with registration
            print(f"User not found, proceeding with registration...")
            result = self.sheets_tool.write_user_data(
                user_name=name,
                email=email,
                phone=phone,
                address= address,
                notes=notes
            )

            if result['success']:
                # Authenticate this gradio session with new user
                self.session_to_user[gradio_session_id] = email

                print(f"✓ New user registered and authenticated")

                return (f"✅ **Cadastro realizado com sucesso!**\n\n"
                        f"👤 Nome: {name}\n"
                        f"📧 Email: {email}\n"
                        f"📱 Telefone: {phone if phone else 'Não informado'}\n\n"
                        f"🎉 Bem-vindo(a)! Agora você pode fazer agendamentos! 📅")
            else:
                return f"❌ {result['message']}"

        except Exception as e:
            error_msg = f"❌ Erro ao salvar dados: {str(e)}"
            print(error_msg)
            return error_msg


    def _create_chain(self):
        current_date = datetime.now().strftime('%Y-%m-%d')

        system_msg = AgentConfig.SYSTEM_MESSAGE.format(current_date=current_date)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])


        chain = prompt | self.llm | StrOutputParser()

        return chain

    def _get_chat_history(self,session_id:str) -> InMemoryChatMessageHistory:
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = InMemoryChatMessageHistory()
        return self.chat_histories[session_id]
    
    def _create_chain_with_history(self) -> RunnableWithMessageHistory:
        return RunnableWithMessageHistory(
            self.chain,
            self._get_chat_history,
            input_messages_key="input",
            history_messages_key="chat_history")

    def chat_with_agent(self,message:str ,session_id:str = "default") -> str:
        try:
            is_authenticated = session_id in self.session_to_user

            if is_authenticated:
                user_email = self.session_to_user[session_id]
                conversation_session_id = f"user_{user_email}"
                print(f"User authenticated: {user_email}")
            else:
                conversation_session_id = f"temp_{session_id}"
                print(f"User not authenticated")

            if is_authenticated:
                context_message = f"[CONTEXT: User is ALREADY AUTHENTICATED in this conversation. Proceed directly with their request without asking for identification again.]"
                enhanced_message = f"{context_message}\n\nUser message: {message}"
            else:
                enhanced_message = message

            response = self.chain_with_history.invoke({"input": enhanced_message},
                                                      config={"configurable":{"session_id":conversation_session_id}})

            processed_response = self._process_response(response,session_id)
            return processed_response

        except Exception as e:
            return f"Error:{str(e)}"

    def clear_history(self, gradio_session_id: str = "default") -> None:
        if gradio_session_id in self.session_to_user:
            user_email = self.session_to_user[gradio_session_id]
            conversation_session_id = f"user_{user_email}"

            # Clear conversation history
            if conversation_session_id in self.chat_histories:
                self.chat_histories[conversation_session_id].clear()

            # Remove authentication
            del self.session_to_user[gradio_session_id]
            print(f"✓ Cleared session for user: {user_email}")
        else:
            # Clear temporary session
            temp_session_id = f"temp_{gradio_session_id}"
            if temp_session_id in self.chat_histories:
                self.chat_histories[temp_session_id].clear()
            print(f"✓ Cleared temporary session")
    
    def get_session_message_count(self, session_id:str = "default") -> int:
        if session_id in self.chat_histories:
            return len(self.chat_histories[session_id].messages)
        return 0
