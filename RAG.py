from typing import List
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from modelscope import snapshot_download
from langdetect import detect, LangDetectException
from googletrans import Translator

# 下载和加载嵌入模型
model_dir = snapshot_download("AI-ModelScope/bge-small-zh-v1.5", cache_dir='.')

class EmbeddingModel:
    """
    Class for EmbeddingModel
    """
    def __init__(self, path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, legacy=False)
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda()
        print(f'Loading EmbeddingModel from {path}.')

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Calculate embeddings for a list of texts
        """
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.cpu().tolist()

print("> Create embedding model...")
embed_model_path = './AI-ModelScope/bge-small-zh-v1.5'
embed_model = EmbeddingModel(embed_model_path)

# 定义向量库索引类
class VectorStoreIndex:
    """
    Class for VectorStoreIndex
    """
    def __init__(self, document_path: str, embed_model: EmbeddingModel) -> None:
        self.documents = [line.strip() for line in open(document_path, 'r', encoding='utf-8')]
        self.embed_model = embed_model
        self.vectors = self.embed_model.get_embeddings(self.documents)
        print(f'Loading {len(self.documents)} documents for {document_path}.')

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        """
        vector1 = np.array(vector1)
        vector2 = np.array(vector2)
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        return dot_product / magnitude if magnitude != 0 else 0

    def query(self, question: str, k: int = 1) -> List[str]:
        question_vector = self.embed_model.get_embeddings([question])[0]
        result = np.array([self.get_similarity(question_vector, vector) for vector in self.vectors])
        sorted_indices = result.argsort()[-k:][::-1]
        return [self.documents[i] for i in sorted_indices]

print("> Create index...")
document_path = "Audio_Generator/knowledge.txt"
# theme= 'classic music'
index = VectorStoreIndex(document_path, embed_model)

# 定义大语言模型类
class LLM:
    """
    Class for Yuan2.0 LLM
    """
    def __init__(self, model_path: str) -> None:
        print("Create tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
        self.tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>', '<commit_before>', '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>', '<jupyter_code>', '<jupyter_output>', '<empty_output>'], special_tokens=True)

        print("Create model...")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()

        print(f'Loading Yuan2.0 model from {model_path}.')
        self.translator = Translator()

    def generate(self, question: str, theme: List[str], temperature: float = 1.0):
        if theme:
            prompt = f'拼接{question}和{theme}'
        else:
            prompt = question

        prompt += "<sep>"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        
        # 生成 attention_mask
        if self.tokenizer.pad_token_id is not None:
            attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id).long().cuda()
        else:
            attention_mask = torch.ones_like(input_ids).cuda()  # 处理 pad_token_id 为 None 的情况
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=True,  # 开启采样以应用温度
            max_length=1024,
            temperature=temperature  # 设置温度
        )
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 输出生成的文本
        print(output)
        return output


print("> Create Yuan2.0 LLM...")
model_path = 'IEITYuan\Yuan2-2B-Mars-hf'
llm = LLM(model_path)

# 示例问题
# theme ='合成器流行（Synth Pop）'
# question = '水流'
def prompt_enhance(question):
    return llm.generate(question, [], temperature=1.0)
    

def prompt_enhance_RAG(question,theme):
    _theme = index.query(theme)
    return llm.generate(question, _theme, temperature=1.0)
# print('> With RAG:')
# print(_theme)
# # print(f'Theme from index: {_theme}')  # Print the retrieved theme for debugging
