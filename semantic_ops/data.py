
from prompt import Prompt


def load_page(table, page_size):
    for i in range(0, len(table)-1, page_size):
        prompt_list = []
        for j in range(i, min(i + page_size, len(table))):
            prompt_list.append(Prompt(statement=table[j]))
        yield Page(prompt_list)


def load_block(table, block_size, page_size):
    pages = list(load_page(table, page_size)) # Materialize page by page
    for i in range(0, len(pages), block_size):
        yield Block(pages[i:i + block_size])


class Page:
    def __init__(self, prompt_list):
        self.prompt_list = prompt_list
        # self.seq_len = self.avg_seq_len()

    def avg_seq_len(self):
        total = 0
        for prompt in self.prompt_list:
            total += len(prompt.input_ids[0])
        return int(total/len(self.prompt_list))

            
class Block:
    def __init__(self, page_list):
        self.page_list = page_list
        # self.seq_len = self.avg_seq_len()

    def avg_seq_len(self):
        total = 0
        for page in self.page_list:
            for prompt in page.prompt_list:
                total += len(prompt.input_ids[0])
        return int(total/len(self.prompt_list))
    

    def __iter__(self):
        return iter(self.page_list)

    def next_page(self): 
        for page in self.page_list:
            yield page

class PagePairLoaders:
    def __init__(self, prompt_constructor, page_size=4):
        self.page_size = page_size
        self.prompt_constructor = prompt_constructor
    
    def next(self, tableA, tableB):
        for pageA in load_page(tableA, self.page_size):
            for pageB in load_page(tableB, self.page_size):
                yield (pageA, pageB)


class BlockPairLoader:
    def __init__(self, page_size=4, block_size=2):
        self.page_size = page_size
        self.block_size = block_size

    def next(self, A, B):
        result = []
        for blockA in load_block(A, self.block_size, self.page_size):
            for blockB in load_block(B, self.block_size, self.page_size):
                yield (blockA, blockB)

        return result



if __name__ == '__main__':
    from prompt import PromptConstructor

    predicate_placeholder = "{a} is aligned with Document [{b}]?" + "test " *500 # Predicate statement
    prompt_constructor = PromptConstructor(placeholder=predicate_placeholder)
    
    tableA = []
    tableB = []
    for i in range(100):
        tableA.append(f'A-{i}')

    for i in range(100):
        tableB.append(f'B-{i}')

    avg_seq_len = 500
    bnlj = BlockPairLoader(int(50 ** 0.5), 3)

    for idx, block_pair in enumerate(bnlj.next(tableA, tableB)):
        blockA, blockB = block_pair
        for pageA in blockA.next_page():
            for pageB in blockB.next_page():
                for promptA in pageA.prompt_list:
                    for promptB in pageB.prompt_list:
                        print(pageA, len(pageA.prompt_list), pageB, len(pageB.prompt_list))