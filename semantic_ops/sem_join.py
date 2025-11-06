
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






if __name__ == '__main__':
    from prompt import PromptConstructor
    from data import BlockPairLoader
    predicate_placeholder = "{a} is aligned with Document [{b}]?" 
    prompt_constructor = PromptConstructor(placeholder=predicate_placeholder)
    
    tableA = []
    tableB = []
    for i in range(100):
        tableA.append(f'A-{i}'+ "test " *500)

    for i in range(100):
        tableB.append(f'B-{i}')

    avg_seq_len = 500
    bnlj = BlockPairLoader(int(50 ** 0.5), 3)


    for idx, block_pair in enumerate(bnlj.next(tableA, tableB)):
        blockA, blockB = block_pair
        for pageA in blockA.page_list:
            for pageB in blockB.page_list:
                for tupleA in pageA.prompt_list:
                    for tupleB in pageB.prompt_list:
                        prompt = prompt_constructor.construct_prompt(tupleA.statement, tupleB.statement)
                        print(prompt.statement)