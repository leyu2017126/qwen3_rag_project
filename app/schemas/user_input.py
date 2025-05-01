from pydantic import BaseModel


# 用户输入
class UserInput(BaseModel):
    query: str
