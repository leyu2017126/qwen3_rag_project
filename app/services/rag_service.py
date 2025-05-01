import os
from pathlib import Path

import pandas as pd
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings


class RAGService:

    def __init__(self):
        # 1. 找到项目根目录（rag_service.py 的上上一级）
        BASE_DIR = Path(__file__).resolve().parent.parent.parent

        self.docs_path = f"{BASE_DIR}/product_docs/"
        self.persist_dir = f"{BASE_DIR}/chroma_dbs/"
        self.chunk_size = 1000
        self.chunk_overlap = 100
        self.embedding_model = f"{BASE_DIR}/models/embedding"
        self.k = 3
        self._vectordb = None
        self._initialize()

    def _initialize(self):
        # 如果已存在持久化目录，直接加载，否则重新构建
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            self._load_vectorstore()
        else:
            self._build_vectorstore()

    def _load_vectorstore(self):
        """
        从持久化目录加载 Chroma 向量数据库
        """
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self._vectordb = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=embeddings
        )

    def _build_vectorstore(self):
        """
        加载文档、分割并生成向量数据库，并持久化
        """
        # 分别加载 txt 和 xlsx 文件，避开单一 loader_cls dict
        txt_loader = DirectoryLoader(
            str(self.docs_path),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}  # 关键参数
        )

        xlsx_docs = []
        for xlsx_file in Path(self.docs_path).glob("**/*.xlsx"):
            # 读 Excel（默认就支持 .xlsx、.xls）
            df = pd.read_excel(xlsx_file, sheet_name=None)  # sheet_name=None 会返回 dict，把所有 sheet 一并读入
            # 如果你只关心第一个 sheet，可以写：df = pd.read_excel(xlsx_file)

            # 如果读的是 dict（多个 sheet）
            if isinstance(df, dict):
                for sheet_name, sheet_df in df.items():
                    for _, row in sheet_df.iterrows():
                        # 把每行拼成一大段文本
                        content = "\n".join(f"{col}: {row[col]}" for col in sheet_df.columns)
                        xlsx_docs.append(Document(page_content=content))
            else:
                # 单个 DataFrame
                for _, row in df.iterrows():
                    content = "\n".join(f"{col}: {row[col]}" for col in df.columns)
                    xlsx_docs.append(Document(page_content=content))

        txt_docs = txt_loader.load()
        documents = txt_docs + xlsx_docs
        print(f"加载文档数：{len(documents)}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        split_docs = splitter.split_documents(documents)
        print(f"分割后文档片段数：{len(split_docs)}")

        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self._vectordb = Chroma.from_documents(
            split_docs,
            embeddings,
            persist_directory=self.persist_dir,
        )

    def retrieve_docs(self, user_input: str) -> str:
        """
        只做语义检索，返回相关文档内容拼接（不调用 LLM）
        """
        docs = []

        # 优先检索城市信息
        city_docs = self._vectordb.similarity_search(f"city:{user_input}", k=self.k)
        docs.extend(city_docs)

        # 然后检索县区信息
        county_docs = self._vectordb.similarity_search(f"county:{user_input}", k=self.k)
        docs.extend(county_docs)

        # 处理多重结果并去重
        docs = list({doc.page_content: doc for doc in docs}.values())

        return "\n".join([doc.page_content for doc in docs])

    def reload(self):
        """
        重新构建向量数据库，例如当文档更新后调用
        """
        self._build_vectorstore()
