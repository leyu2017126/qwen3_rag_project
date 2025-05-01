from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.services.rag_service import RAGService


class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-1.7B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
        )
        self.history = []

    def generate_response(self, user_input, rag: RAGService):
        # 搜索相近内容
        related_content = rag.retrieve_docs(user_input)
        # 提示词模板
        PROMPT_TEMPLATE = f"""
                        基于以下已知信息，简洁和专业的来回答用户的问题。不允许在答案中添加编造成分。
                        已知内容:
                        {related_content}
                        问题:
                        {user_input}
                    """
        messages = self.history + [{"role": "user", "content": PROMPT_TEMPLATE}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response
