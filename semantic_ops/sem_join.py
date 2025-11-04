def estimate_page_size(table):
    return 5

def load_pages(table, page_size):
    for i in range(0, len(table), page_size):
        yield table[i:i + page_size]

def load_blocks(table, block_pages, page_size):
    block_size = block_pages * page_size
    for i in range(0, len(table), block_size):
        yield table[i:i + block_size]


class SemanticFilter:
    pass


class JoinAlgorithm:
    def join(self, A, B):
        raise NotImplementedError
    

class SemanticJoin:
    def __init__(self,  join_impl: JoinAlgorithm):
        self.join_ops = join_impl

    def join(self, A, B):
        return self.join_impl.join(A, B)



class BlockNestedLoopJoinImpl(JoinAlgorithm):
    def __init__(self, prompt, page_size=4, block_pages=2):
        self.page_size = page_size
        self.block_pages = block_pages
        self.prompt = prompt

    def join(self, A, B):
        result = []
        for blockA in load_blocks(A, self.block_pages, self.page_size):
            for pageB in load_pages(B, self.page_size):
                for a in blockA:
                    for b in pageB:
                        print(self.prompt.construct_statement(a, b))

        return result



class PageNestedLoopJoinImpl(JoinAlgorithm):
    def __init__(self, prompt_constructor, page_size=4):
        self.page_size = page_size
        self.prompt_constructor = prompt_constructor


    def join(self, A, B):
        result = []
        for pageA in load_pages(A, self.page_size):
            for pageB in load_pages(B, self.page_size):
                for a in pageA:
                    for b in pageB:
                        prompt = self.prompt_constructor.construct_prompt(a, b)
                        print('full statement: ', prompt.statement)
                        print('cachine part: ', prompt.caching_part)


        return result


if __name__ == '__main__':
    from prompt import Prompt, PromptConstructor
    predicate_placeholder = "{a} is aligned with Document [{b}]?" # Predicate statement
    prompt_constructor = PromptConstructor(placeholder=predicate_placeholder)

    tableA = ['test1'] * 5
    tableB = ['test2'] * 3

    bnlj = PageNestedLoopJoinImpl(prompt_constructor)
    print(bnlj.join(tableA, tableB))