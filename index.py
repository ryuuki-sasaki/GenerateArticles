import time
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import (
    load_tools,
    initialize_agent,
    AgentType,
)
from langchain.document_loaders import SeleniumURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# .env ファイルをロードして環境変数へ反映
from dotenv import load_dotenv
load_dotenv()
import os
import re

TEXT_SIZE_LIMIT = 4000

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if OPENAI_API_KEY is None:
    raise Exception("OPENAI_API_KEY is not set.")

openAI = OpenAI(temperature=0)
openAIChat = ChatOpenAI(temperature=0)

# urlsを受け取って、Seleniumでページを開いて、ページの内容を返す。
def GetPageContent(urls):
    loader = SeleniumURLLoader(urls=urls)
    data = loader.load()
    return data[0].page_content

# 余計な空白と改行を一つの空白に置き換える関数
# GPT-3は空白もトークンとしてカウントされるらしいので、余計な空白や改行を削除する。
def remove_space_and_newline(text):
    text = re.sub(r'[\s]+', ' ', text)
    return text

# GPT-3による文章要約
def SummarizeText(docs):
    #  Documentsを並列で要約するmap_reduceの方が早いが、今回は要約の精度重視でrefineを使う。
    chain = load_summarize_chain(openAI, chain_type="refine")
    summary = chain.run(docs)
    return summary

def GenerateArticle():
    urls = [
        "https://prtimes.jp/main/html/rd/p/000000042.000045103.html",
    ]
    content = GetPageContent(urls)
    text = remove_space_and_newline(content) 
    text_splitter = CharacterTextSplitter(        
        separator = " ",
        chunk_size = 1000,
    )
    docs = text_splitter.create_documents([text])

    # テキストのサイズが定数文字を超えるなら文章を要約する
    summary = ""
    if len(text) > TEXT_SIZE_LIMIT:
        summary = SummarizeText(docs)

    # systemメッセージプロンプトテンプレートの準備
    template="これからテキストを複数に分割して渡しますので、私が「これで全ての文章を渡しました。」というまでは、作業を始めないでください。代わりに「次の入力を待っています」とだけ出力してください。"
    # chatプロンプトテンプレートの準備
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    # メモリの準備(トークン上限は2000,超過した分は要約される)
    memory = ConversationSummaryBufferMemory(llm=openAI, max_token_limit=2000, return_messages=True)
    conversation_with_summary = ConversationChain(
        llm=openAIChat, 
        prompt=prompt,
        memory=memory,
        verbose=True,
    )
    
    if summary != "":
        conversation_with_summary.predict(input=summary)
    else:
        for doc in docs:
            conversation_with_summary.predict(input=doc.page_content + "\n\n上記の文章は全体のテキストの一部です。まだまとめないでください")
    conversation_with_summary.predict(input="これで全ての文章を渡しました。この文章からタイトルとまとめを生成してください。")
    res = conversation_with_summary.predict(input="この文章のキーワードを重要度の高い順に3つ挙げてください。")
    print(res)
    
    tools = load_tools(["google-search"], llm=openAI)
    agent = initialize_agent(tools, openAI, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    ret = agent.run(
        f"{res}に関連する情報を検策してまとめてください。検策を行う際は検策先URLを覚えておいてください。最後に検索に使用したURLをリストで教えてください。"
    )
    print(ret)
