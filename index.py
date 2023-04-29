from langchain.document_loaders import SeleniumURLLoader

urls = [
    "https://prtimes.jp/main/html/rd/p/000000042.000045103.html",
]

loader = SeleniumURLLoader(urls=urls)

data = loader.load()

print(data)