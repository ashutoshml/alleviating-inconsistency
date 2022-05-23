class Tokenizer_nltk:
    def __init__(self, voc_path=None):
        if not voc_path:
            raise ValueError('Please Provide vocabulary path built using create_vocab_nltk.py')

        with open(voc_path, 'rb') as f:
            voc = pickle.load(f)

        self.voc = voc

    def __call__(self, text_batch, padding=False, truncation=False, max_len=512):
        if isinstance(text_batch, str) or len(text_batch) == 1:
            if not padding or not truncation:
                raise ValueError("Unable to tokenize, please set padding/truncation True for multiple sentences")

        if isinstance(text_batch, str):
            text_batch = [text_batch]

        batch_size = len(text_batch)
        maxl = max([len(line.split()) for line in text_batch]) + 2

        if truncation:
            if maxl > max_len:
                maxl = max_len

        input_ids = torch.zeros((2,), dtype=torch.int32)
        input_ids.new_full((batch_size, maxl), self.voc.w2id['<pad>'])

        attention_mask = torch.zeros((batch_size, maxl), dtype=torch.int32)

        for i, line in enumerate(tqdm(text_batch, desc='Tokenizing strings')):
            line = self.voc.clean_str(line)
            words = '<s> {} </s>'.format(" ".join(word_tokenize(line))).split()[:maxl]
            for j, word in enumerate(words):
                input_ids[i][j] = self.voc.w2id[word] if word in self.voc.w2id else self.voc.w2id['UNK']
                attention_mask[i][j] = 1

        final_tokenized = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


    def convert_ids_to_tokens(self, input_ids):
        shape = input_ids.size()
        tokens = [['<pad>' for _ in range(shape[1].item())] for _ in range(shape[0].item())]
        for row in range(shape[0].item()):
            for col in range(shape[1].item()):
                tokens[row][col] = self.voc.id2w[input_ids[row][col].item()]

        return tokens

    def convert_tokens_to_string(self, tokens, special_tokens=False):
        strings = []
        if special_tokens:
            stoken = ['<s>', '</s>', '<pad>']
        else:
            stoken = ['']
        for token in tokens:
            currstr = " ".join(token)

            for s in stoken:
                currstr = currstr.replace(s, '')

            strings.append(currstr)

        return strings
