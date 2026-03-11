import gradio as gr
from ai import ChatAgent

class  ChatbotUI:

    def __init__(self,agent:ChatAgent):
        self.agent = agent
        self.session_id = "gradio-session-1"

    def chat_handler(self,message:str,history:list) -> str:
        return self.agent.chat_with_agent(message,self.session_id)
    
    def clear_handler(self):
        self.agent.clear_history(self.session_id)
        return None
    
    def build_interface(self) -> gr.Blocks:
        with gr.Blocks() as demo:
            gr.Markdown("Ai Agent")
            chatbot = gr.ChatInterface(self.chat_handler,type="messages")

            with gr.Row():
                clear = gr.Button("Clear chat")

            clear.click(fn=self.clear_handler,outputs=[chatbot])


        return demo

    def launch(self):
        demo = self.build_interface()
        demo.launch()

def main():

    try:
        agent = ChatAgent()
        print("O agente do Chat foi inicializado com sucesso!")

        ui = ChatbotUI(agent)
        print("Iniciando a interface do Gradio")
        ui.launch()

    except ValueError as e:
        print(f"Erro de configuração: {e}")
        print("\nPlease set your OPENAI_API_KEY environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()