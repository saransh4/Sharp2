from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class ModelHandler:
    _instance = None  # Singleton instance
    # _loaded = False   # Flag for is_loaded check

    def __init__(self, model_name="google/flan-t5-base"):

        self._loaded=False
        if not self._loaded:  # Only load once
            print(f"🔄 Loading model: {model_name} ...")
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name).to(self.device)
            self._loaded = True
            ModelHandler._instance = self
            print("✅ Model loaded successfully!")

    @classmethod
    def get_instance(cls, model_name="google/flan-t5-base"):
        if cls._instance is None:
            cls(model_name=model_name)
        return cls._instance

    def is_loaded(self):
        return self._loaded

    def generate_responses(self, prompts, max_new_tokens=150):
        if not prompts:
            return []

        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append({"prompt": prompt, "response": text})
        return results
