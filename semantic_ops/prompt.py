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




if __name__ == '__main__':
    predicate_placeholder = "{a} is aligned with Document [{b}]?" # Predicate statement
    prompt_constructor = PromptConstructor(placeholder=predicate_placeholder)
    a = 'test1'
    b = 'test2'
    prompt = prompt_constructor.construct_prompt(a, b)
    print(prompt.statement)
    print(prompt.caching_part)
