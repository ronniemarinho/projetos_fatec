import streamlit as st
from nltk.chat.util import Chat, reflections

# Definindo os pares de padrões e respostas
pairs = [
    [
        r"meu nome é (.*)",
        ["Olá %1, como posso ajudá-lo hoje?"]
    ],
    [
        r"qual é o seu nome ?",
        ["Meu nome é Chatbot.", "Você pode me chamar de Chatbot."]
    ],
    [
        r"como você está ?",
        ["Estou bem, obrigado por perguntar.", "Estou ótimo! E você?"]
    ],
    [
        r"qual é a sua função ?",
        ["Eu sou um chatbot criado para responder perguntas simples."]
    ],
    [
        r"como posso criar um chatbot ?",
        ["Você pode usar bibliotecas como NLTK em Python para criar um chatbot."]
    ],
    [
        r"quais cursos são oferecidos na fatec?",
        ["Oferecemos os cursos de Ciência de Dados, Gestão Comercial e Logística."]
    ],
    [
        r"quais são as instalações disponíveis na faculdade ?",
        [
            "A faculdade possui bibliotecas, laboratórios modernos, quadras esportivas, residências estudantis, e refeitórios."]
    ],
    [
        r"qual é o endereço da faculdade ?",
        ["A faculdade está localizada na Avenida das Universidades, 123, Cidade Universitária."]
    ],
    [
        r"quais são os horários de visita ?",
        ["As visitas podem ser feitas de segunda a sexta, das 9h às 17h."]
    ],
    [
        r"como posso me inscrever no vestibular?",
        ["Você pode se inscrever pelo nosso site ou entrando em contato com a secretaria da faculdade."]
    ],
    [
        r"quais são os requisitos para admissão ?",
        [
            "Os requisitos variam conforme o curso, mas geralmente incluem um bom desempenho no vestibular ou no ENEM, além de documentação acadêmica."]
    ],
    [
        r"há oportunidades de bolsas de estudo ?",
        ["Sim, oferecemos diversas bolsas de estudo baseadas em mérito acadêmico e necessidades financeiras."]
    ],
    [
        r"qual é o custo das mensalidades ?",
        [
            "O custo das mensalidades varia conforme o curso. Para mais detalhes, por favor visite nosso site ou entre em contato com a secretaria."]
    ],
    [
        r"qual é a duração dos cursos ?",
        ["A duração dos cursos varia, mas geralmente é entre 4 a 6 anos para cursos de graduação."]
    ],
    [
        r"tchau|adeus",
        ["Até mais!", "Tchau, até logo!"]
    ],
    [
        r"(.*)",
        ["Desculpe, não entendi. Pode reformular a pergunta?"]
    ]
]

# Reflections são usadas para substituir algumas palavras em uma frase por outras palavras
reflections = {
    "eu sou": "você é",
    "eu era": "você era",
    "eu": "você",
    "eu vou": "você vai",
    "meu": "seu",
    "minha": "sua",
    "me": "te",
    "eu estou": "você está"
}

# Criar uma instância do chatbot
chatbot = Chat(pairs, reflections)

# Função principal do Streamlit
def main():
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>Chatbot de Informações da Fatec Adamantina</h1>
        </div>
        """, unsafe_allow_html=True
    )
    # Centralizando a imagem usando colunas
    col1, col2, col3 = st.columns([1, 2, 1])  # Definindo uma estrutura de colunas com proporções 1:2:1

    with col2:
        st.image('/Users/ronnieshida/PycharmProjects/novoprojeto/img.png', width=350)

    # Instruções para o usuário
    st.write("Olá! Eu sou um chatbot criado para responder suas perguntas sobre a Fatec de Adamantina.")
    st.write("Pergunte-me algo e eu farei o meu melhor para ajudar!")

    # Inicializar o estado da sessão para o input do usuário
    if 'user_input' not in st.session_state:
        a = st.session_state.user_input = ""

    # Caixa de entrada de texto do usuário
    user_input = st.text_input("Você:", st.session_state.user_input)
    st.session_state.user_input = ""
    if user_input and st.session_state.user_input != user_input:
        # Obter a resposta do chatbot
        response = chatbot.respond(user_input)
        st.text_area("Chatbot:", value=response, height=200, max_chars=None, key=None)

        # Atualizar o estado do input do usuário para limpar o texto
        st.session_state.user_input = ""

if __name__ == "__main__":
    main()
