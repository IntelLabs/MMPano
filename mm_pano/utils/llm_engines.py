import copy
from typing import Optional, Dict
from abc import ABC, abstractmethod

from text_generation import Client
from utils.model_utils import load_llm

_VALIDATED_MODELS = [
    "gpt-4", "gpt-3.5-turbo",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "tgi",
]


class BaseLLMEngine(ABC):
    @abstractmethod
    def chat(self, user_content:str, system_content: str, history: Optional[str] = None):
        pass

    def extract_output(self, output: str) -> str:
        return output


class OpenAILLMEngine(BaseLLMEngine):
    def __init__(self,
                 model_engine: str = None,
                 openai=None,
                 openai_key: str = None):
        self.model_engine = model_engine
        self.openai = openai
        self.openai.api_key = openai_key
        print(f"Using model engine {self.model_engine} to generate text")

    def chat(self,
             user_content: str,
             system_content: str = "You are a helpful assistant.",
             history: str = None) -> str:

        message = self.openai.ChatCompletion.create(
            model=self.model_engine,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]).choices[0]['message']['content']

        # For now the history always = None
        return message, None


class QwenLLMEngine(BaseLLMEngine):
    def __init__(self,
                 tokenizer,
                 model):
        self.tokenizer = tokenizer
        self.model = model

    def chat(self,
             user_content: str,
             system_content: str = "You are a helpful assistant.",
             history: str = None):
        message, history = self.model.chat(self.tokenizer, user_content, history=history)
        return message, history


class MistralLLMEngine(BaseLLMEngine):
    def __init__(self,
                 tokenizer,
                 model,
                 default_generate_kwargs: Optional[Dict] = None):
        self.tokenizer = tokenizer
        self.model = model
        self.default_generate_kwargs = {} if default_generate_kwargs is None else default_generate_kwargs

    def chat(self,
             user_content: str,
             system_content: str = "You are a helpful assistant.",
             history: str = None,
             generate_kwargs: Optional[Dict] = None):

        _generate_kwargs = copy.deepcopy(self.default_generate_kwargs)
        if generate_kwargs is not None:
            _generate_kwargs.update(generate_kwargs)

        messages = [
            {"role": "user", "content": user_content},
        ]
        model_inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(model_inputs, **_generate_kwargs)
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # TODO(Tien Pei Chou): Find a better way to only output the new tokens.
        return decoded[0][decoded[0].rfind("[/INST]") + len("[/INST] "):], None


class Llama3LLMEngine(BaseLLMEngine):
    def __init__(self,
                 tokenizer,
                 model,
                 default_generate_kwargs: Optional[Dict] = None):
        self.tokenizer = tokenizer
        self.model = model
        self.default_generate_kwargs = {} if default_generate_kwargs is None else default_generate_kwargs

    def extract_output(self, output: str) -> str:
        return output[output.rfind("assistant\n\n") + len("assistant\n\n"):]

    def chat(self,
             user_content: str,
             system_content: str = "You are a helpful assistant.",
             history: str = None,
             generate_kwargs: Optional[Dict] = None):

        _generate_kwargs = copy.deepcopy(self.default_generate_kwargs)
        if generate_kwargs is not None:
            _generate_kwargs.update(generate_kwargs)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        model_inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(model_inputs, **_generate_kwargs)
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # TODO(Tien Pei Chou): Find a better way to only output the new tokens.
        return self.extract_output(decoded[0]), None


class TGILLMEngine(BaseLLMEngine):
    def __init__(self,
                 tgi_url: Optional[str] = "http://127.0.0.1:8080",
                 default_generate_kwargs: Optional[Dict] = None):
        self.client = Client(tgi_url)
        self.default_generate_kwargs = {} if default_generate_kwargs is None else default_generate_kwargs

    def chat(self,
             user_content: str,
             system_content: str = "You are a helpful assistant.",
             history: str = None,
             generate_kwargs: Optional[Dict] = None):

        _generate_kwargs = copy.deepcopy(self.default_generate_kwargs)
        if generate_kwargs is not None:
            _generate_kwargs.update(generate_kwargs)

        response = self.client.generate(user_content, **_generate_kwargs, return_full_text=False)

        return self.extract_output(response.generated_text), None


def get_llm_engine(model_name: str,
                   dtype: Optional[str] = "float32",
                   device: Optional[str] = "hpu",
                   openai_key: Optional[str] = None,
                   hf_token: Optional[str] = None,
                   tgi_url: Optional[str] = "http://127.0.0.1:8080"):
    if model_name in ["gpt-4", "gpt-3.5-turbo"]:
        import openai
        assert openai_key is not None, "Please set the `openai_key` when using OpenAI API"
        print(f"Using OpenAI {model_name} API for text generaton ...")
        return OpenAILLMEngine(model_engine=model_name, openai=openai, openai_key=openai_key)
    elif model_name == "mistralai/Mistral-7B-Instruct-v0.2":
        tokenizer, model = load_llm(
            model_name=model_name,
            device=device,
            dtype=dtype,
            trust_remote_code=True,
            hf_token=hf_token)
        default_generate_kwargs = {
            "do_sample": True,
            "temperature": 0.7,
            "max_new_tokens": 256
        }
        print(f"Using {model_name} for text generaton ...")
        return MistralLLMEngine(tokenizer, model, default_generate_kwargs=default_generate_kwargs)
    elif "Llama" in model_name:  # Ex: "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer, model = load_llm(
            model_name=model_name,
            device=device,
            dtype=dtype,
            trust_remote_code=True,
            hf_token=hf_token)
        default_generate_kwargs = {
            "do_sample": True,
            "temperature": 0.6,
            "max_new_tokens": 256
        }
        print(f"Using {model_name} for text generaton ...")
        return Llama3LLMEngine(tokenizer, model, default_generate_kwargs=default_generate_kwargs)
    elif "tgi" in model_name:
        assert tgi_url is not None, "Must pass a url to the client when using TGI-Gaudi"
        default_generate_kwargs = {
            "do_sample": True,
            "temperature": 0.6,
            "max_new_tokens": 256
        }
        return TGILLMEngine(tgi_url, default_generate_kwargs)
    else:
        raise NotImplementedError(f"Got unsupported model {model_name}")
