from langchain import OpenAI
from langchain.document_loaders import SeleniumURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# .env ファイルをロードして環境変数へ反映
from dotenv import load_dotenv
load_dotenv()
import os
import re

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if OPENAI_API_KEY is None:
    raise Exception("OPENAI_API_KEY is not set.")

llm = OpenAI(temperature=0)

urls = [
    "https://prtimes.jp/main/html/rd/p/000000042.000045103.html",
]

loader = SeleniumURLLoader(urls=urls)

data = loader.load()

# 余計な空白と改行を一つの空白に置き換える関数
# GPT-3は空白もトークンとしてカウントされるらしいので、余計な空白や改行を削除する。
def remove_space_and_newline(text):
    text = re.sub(r'[\s]+', ' ', text)
    return text

# GPT-3による文章要約
def SummarizeText(docs):
    #  Documentsを並列で要約するmap_reduceの方が早いが、今回は要約の精度重視でrefineを使う。
    chain = load_summarize_chain(llm, chain_type="refine")
    summary = chain.run(docs)
    return summary

def GenerateArticle():
    text = remove_space_and_newline(data[0].page_content) 
    text_splitter = CharacterTextSplitter(        
        separator = " ",
        chunk_size = 1000,
    )
    docs = text_splitter.create_documents([text])

    # 文章を要約する
    summary = SummarizeText(docs)
    print(summary)

GenerateArticle()