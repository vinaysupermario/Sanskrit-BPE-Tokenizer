class SanskritBPETokenizer:
    def __init__(self, tokenizer_path):
        with open(tokenizer_path, "r") as f:
            data = json.load(f)
        
        # Convert back to tuple pairs and create vocab maps
        self.merges = [(tuple(pair), idx) for pair, idx in data["merges"]]
        self.vocab = {idx: tuple(pair) for pair, idx in self.merges}
        self.orig_vocab_size = data["orig_vocab_size"]
        
        # Create byte->token and token->byte mappings for base vocabulary
        self.byte_to_token = {i: i for i in range(self.orig_vocab_size)}
        self.token_to_byte = {v: k for k, v in self.byte_to_token.items()}

    def _merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def encode(self, text):
        # Convert text to bytes and map to reduced vocabulary
        bytes_ = text.encode("utf-8")
        tokens = [b % self.orig_vocab_size for b in bytes_]  # Wrap bytes to 0-127
        ids = list(tokens)
        
        # Apply merges
        for pair, idx in self.merges:
            ids = self._merge(ids, pair, idx)
        return ids

    def decode(self, ids):
        # Expand merged tokens
        expanded = []
        for idx in reversed(ids):  # Reverse for stack processing
            stack = [idx]
            while stack:
                current = stack.pop()
                if current in self.vocab:
                    a, b = self.vocab[current]
                    stack.append(b)
                    stack.append(a)
                else:
                    if current >= self.orig_vocab_size:
                        raise ValueError(f"Invalid token id: {current}")
                    # Convert back to original byte value
                    expanded.append(self.token_to_byte[current])
        
        # Convert bytes to text
        return bytes(reversed(expanded)).decode("utf-8", errors="replace")
