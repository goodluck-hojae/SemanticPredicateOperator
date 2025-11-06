class PromptConstructor:

    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.first_part = None
        
    def _caching_part(self, a):
        return f"Document [{a}]"

    def construct_prompt(self, a, b=None):
        if b is not None:
            caching_part = self._caching_part(a)
            statement = self.placeholder.format(a=caching_part, b=b)
            prompt = Prompt(statement, caching_part)
            return prompt
        
        statement = self.placeholder.format(a)
        prompt = Prompt(statement)
        return prompt


class Prompt:
    def __init__(self, statement, caching_part=None):
        self.statement = statement
        self.caching_part = caching_part
        self.input_ids = None

    def set_input_ids(self, input_ids):
        self.input_ids = input_ids




if __name__ == '__main__':

    import os, sys

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../transformers/src"))
    print(project_root)
    sys.path.insert(0, project_root) 

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    backend_path = os.path.join(project_root, "backend")
    calibration_path = os.path.join(project_root, "calibration")
    calibration_path = os.path.join(project_root, "semantic_ops")
    sys.path.extend([backend_path, calibration_path])

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = '/datasets/ai/llama3/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08'
    model_name = "/datasets/ai/llama3/hub/models--meta-llama--Meta-Llama-3-70B/snapshots/c82494877ce7f6d7d317c56ec081328e382c72fe"
    tok = AutoTokenizer.from_pretrained(model_name)



    predicate_placeholder = "{a} is aligned with Document [{b}]?" # Predicate statement
    prompt_constructor = PromptConstructor(placeholder=predicate_placeholder)
    a = 'test1'
    b = 'test2'
    prompt = prompt_constructor.construct_prompt(a, b)
    print(prompt.statement)
    print(prompt.caching_part)
